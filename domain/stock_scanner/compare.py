"""Stock vs Option trade route comparator.

For any ticker flagged by the stock scanner, this module:
  1. Fetches the live option chain via the existing data source fallback chain.
  2. Projects both a direct share purchase and an option contract purchase
     forward to the user's price target and stop using Black-Scholes.
  3. Returns a TradeComparison with a clear verdict and human-readable reason.

Design notes:
  - IV is held constant in the BS projection (best-case for long premium;
    real moves typically involve some IV crush after a directional run).
  - Contracts are filtered to DTE in [hold_days, hold_days+30] to give the
    trade cushion past the expected hold horizon.
  - Budget cap: single option position limited to 30% of account_balance.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.factory import default_sources

log = logging.getLogger(__name__)

# Suppress py_vollib deprecation noise at import time
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.greeks.analytical import delta as bs_delta


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StockRoute:
    """Buying shares directly."""
    ticker: str
    shares: int                # whole shares only
    entry_price: float
    cost: float                # shares * entry_price
    target_pnl: float          # $ profit if spot hits target
    stop_pnl: float            # $ loss if spot hits stop (negative)
    target_pct: float          # % gain relative to cost
    stop_pct: float            # % loss relative to cost (negative)
    rr_ratio: float            # |target_pnl / stop_pnl|


@dataclass(frozen=True)
class OptionRoute:
    """Buying a single options contract."""
    ticker: str
    option_type: Literal['call', 'put']
    strike: float
    expiration: str            # YYYY-MM-DD
    dte_at_entry: int
    entry_mid: float           # price per share (mid of bid/ask)
    cost: float                # entry_mid * 100
    iv_at_entry: float
    delta: float
    bid: float
    ask: float
    projected_target_value: float    # estimated mid at target spot + hold_days
    projected_stop_value: float      # estimated mid at stop spot + hold_days
    target_pnl: float                # (target_value - entry_mid) * 100
    stop_pnl: float                  # (stop_value - entry_mid) * 100
    breakeven_spot: float            # spot at expiration needed for P&L = 0
    rr_ratio: float                  # |target_pnl / stop_pnl|
    leverage: float                  # target_pnl_pct / stock_target_pnl_pct
    notes: str                       # warnings (deep OTM, tight spread, etc.)


@dataclass(frozen=True)
class TradeComparison:
    """Full comparison for one ticker."""
    ticker: str
    direction: Literal['bullish', 'bearish']
    spot: float
    target: float
    stop: float
    hold_days: int
    account_balance: float
    stock_route: StockRoute
    option_routes: list[OptionRoute]     # top 3 ranked by R:R, within budget
    verdict: Literal['STOCK_PREFERRED', 'OPTION_PREFERRED', 'CLOSE_CALL']
    verdict_reason: str


# ---------------------------------------------------------------------------
# Black-Scholes helper
# ---------------------------------------------------------------------------

def _project_option_value(
    strike: float,
    spot_future: float,
    days_to_expiry: int,
    iv: float,
    option_type: str,
    risk_free_rate: float = 0.053,
) -> float:
    """Use Black-Scholes to project option mid at a future spot + remaining DTE.

    Assumes IV stays constant (best-case for long premium; reality usually
    involves slight IV crush after a directional move proves out).

    Args:
        strike:          Contract strike price.
        spot_future:     Projected spot at the future date.
        days_to_expiry:  Calendar days remaining to expiration at that future date.
        iv:              Implied volatility (annualised, e.g. 0.28 = 28%).
        option_type:     'call' or 'put'.
        risk_free_rate:  Risk-free rate (default 5.3%).

    Returns:
        Projected option price per share (multiply by 100 for contract value).
        Returns 0.0 if the option has zero or negative DTE remaining.
    """
    if days_to_expiry <= 0:
        # At or past expiration — use intrinsic value only
        if option_type == 'call':
            return max(0.0, spot_future - strike)
        return max(0.0, strike - spot_future)

    t = days_to_expiry / 365.0
    # Clamp IV to avoid BS domain errors (degenerate near-zero vol)
    iv_clamped = max(iv, 0.01)
    flag = 'c' if option_type == 'call' else 'p'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            price = black_scholes(flag, spot_future, strike, t, risk_free_rate, iv_clamped)
        return max(0.0, float(price))
    except Exception as exc:
        log.debug("BS pricing failed (S=%.2f K=%.2f t=%.4f iv=%.3f): %s",
                  spot_future, strike, t, iv_clamped, exc)
        # Fall back to intrinsic value
        if option_type == 'call':
            return max(0.0, spot_future - strike)
        return max(0.0, strike - spot_future)


def _compute_delta(
    strike: float,
    spot: float,
    days_to_expiry: int,
    iv: float,
    option_type: str,
    risk_free_rate: float = 0.053,
) -> float:
    """Compute Black-Scholes delta. Returns 0.0 on failure."""
    if days_to_expiry <= 0:
        return 1.0 if (option_type == 'call' and spot > strike) else 0.0
    t = days_to_expiry / 365.0
    iv_clamped = max(iv, 0.01)
    flag = 'c' if option_type == 'call' else 'p'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(bs_delta(flag, spot, strike, t, risk_free_rate, iv_clamped))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Chain filtering
# ---------------------------------------------------------------------------

def _select_candidate_contracts(
    chain: list,
    spot: float,
    direction: str,
    target: float,
    hold_days: int,
    account_balance: float,
) -> list:
    """Filter chain to relevant, affordable contracts for the trade direction.

    Filters applied (in order):
      1. Option type matches direction: bullish → calls, bearish → puts.
      2. DTE in [hold_days, hold_days + 30] — needs cushion past the hold.
      3. Mid price in [0.20, account_balance * 0.30 / 100] per share.
      4. For bullish: strikes in [spot, spot * 1.10]; bearish: [spot * 0.90, spot].

    Returns contracts sorted by distance from ATM (closest first).
    """
    option_type = 'call' if direction == 'bullish' else 'put'
    # Allow up to 7 days short of hold_days so the nearest standard expiry isn't
    # excluded by a few days (e.g. 28-day hold with 25-DTE expiry is fine).
    dte_min = max(1, hold_days - 7)
    dte_max = hold_days + 30
    max_mid_per_share = (account_balance * 0.30) / 100.0  # 30% of account ÷ 100 shares

    if direction == 'bullish':
        strike_lo, strike_hi = spot, spot * 1.10
    else:
        strike_lo, strike_hi = spot * 0.90, spot

    candidates = []
    for contract in chain:
        if contract.option_type != option_type:
            continue
        if not (dte_min <= contract.dte <= dte_max):
            continue
        if contract.bid <= 0 or contract.ask <= 0:
            continue
        mid = (contract.bid + contract.ask) / 2.0
        if mid < 0.20:
            continue
        if mid > max_mid_per_share:
            continue
        if not (strike_lo <= contract.strike <= strike_hi):
            continue
        candidates.append(contract)

    # Sort by distance from ATM (closest first)
    candidates.sort(key=lambda c: abs(c.strike - spot))
    return candidates


# ---------------------------------------------------------------------------
# Route builders
# ---------------------------------------------------------------------------

def _build_stock_route(
    ticker: str,
    spot: float,
    target: float,
    stop: float,
    account_balance: float,
) -> StockRoute:
    """Build the share purchase route."""
    shares = max(1, int(account_balance / spot))  # floor to whole shares
    cost = shares * spot
    target_pnl = shares * (target - spot)
    stop_pnl = shares * (stop - spot)          # negative for bullish (stop < spot)
    target_pct = target_pnl / cost
    stop_pct = stop_pnl / cost
    # Guard against zero denominator (stop == entry)
    rr_ratio = abs(target_pnl / stop_pnl) if stop_pnl != 0 else 0.0
    return StockRoute(
        ticker=ticker,
        shares=shares,
        entry_price=round(spot, 2),
        cost=round(cost, 2),
        target_pnl=round(target_pnl, 2),
        stop_pnl=round(stop_pnl, 2),
        target_pct=round(target_pct, 4),
        stop_pct=round(stop_pct, 4),
        rr_ratio=round(rr_ratio, 4),
    )


def _build_option_route(
    contract,
    spot: float,
    target: float,
    stop: float,
    hold_days: int,
    risk_free_rate: float,
    stock_target_pct: float,
) -> OptionRoute:
    """Project the contract forward and return an OptionRoute.

    Args:
        contract:           RawContract from the data layer.
        spot:               Current spot price.
        target:             Price target.
        stop:               Stop level.
        hold_days:          Expected hold duration in calendar days.
        risk_free_rate:     Risk-free rate for BS.
        stock_target_pct:   % gain from stock route (for leverage calc).
    """
    mid = (contract.bid + contract.ask) / 2.0
    iv = contract.implied_volatility
    dte = contract.dte
    dte_at_target = max(0, dte - hold_days)

    target_value = _project_option_value(
        contract.strike, target, dte_at_target, iv, contract.option_type, risk_free_rate
    )
    stop_value = _project_option_value(
        contract.strike, stop, dte_at_target, iv, contract.option_type, risk_free_rate
    )

    target_pnl = (target_value - mid) * 100
    stop_pnl = (stop_value - mid) * 100
    cost = mid * 100

    rr_ratio = abs(target_pnl / stop_pnl) if stop_pnl != 0 else 0.0

    # Breakeven at expiration
    if contract.option_type == 'call':
        breakeven_spot = contract.strike + mid
    else:
        breakeven_spot = contract.strike - mid

    # Leverage: how many times the option % gain exceeds the stock % gain.
    # Use absolute values so bearish trades (negative pct moves) yield a
    # positive leverage ratio — it represents capital efficiency, not direction.
    option_target_pct = abs(target_pnl / cost) if cost > 0 else 0.0
    leverage = (option_target_pct / abs(stock_target_pct)) if stock_target_pct != 0 else 0.0

    # Compose notes / warnings
    notes_list = []
    otm_pct = abs(contract.strike - spot) / spot
    if otm_pct > 0.07:
        notes_list.append(f"deep OTM ({otm_pct*100:.1f}% from spot) — low probability")
    if iv > 0.50:
        notes_list.append(f"high IV ({iv*100:.0f}%) — elevated vol crush risk")
    spread_pct = (contract.ask - contract.bid) / mid if mid > 0 else 0
    if spread_pct > 0.15:
        notes_list.append(f"wide spread ({spread_pct*100:.0f}% of mid)")
    if dte_at_target <= 7:
        notes_list.append("short DTE at target date — theta accelerates")

    delta_val = _compute_delta(
        contract.strike, spot, dte, iv, contract.option_type, risk_free_rate
    )

    expiry_str = (
        contract.expiration.strftime('%Y-%m-%d')
        if hasattr(contract.expiration, 'strftime')
        else str(contract.expiration)
    )

    return OptionRoute(
        ticker=contract.ticker,
        option_type=contract.option_type,
        strike=contract.strike,
        expiration=expiry_str,
        dte_at_entry=dte,
        entry_mid=round(mid, 4),
        cost=round(cost, 2),
        iv_at_entry=round(iv, 4),
        delta=round(delta_val, 4),
        bid=contract.bid,
        ask=contract.ask,
        projected_target_value=round(target_value, 4),
        projected_stop_value=round(stop_value, 4),
        target_pnl=round(target_pnl, 2),
        stop_pnl=round(stop_pnl, 2),
        breakeven_spot=round(breakeven_spot, 2),
        rr_ratio=round(rr_ratio, 4),
        leverage=round(leverage, 4),
        notes='; '.join(notes_list) if notes_list else '',
    )


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _decide_verdict(
    stock: StockRoute,
    options: list[OptionRoute],
) -> tuple[str, str]:
    """Return (verdict, reason) based on relative R:R, IV, and leverage."""
    if not options:
        return ('STOCK_PREFERRED', 'No affordable option contracts found within budget.')

    best_option = options[0]   # top by R:R after ranking

    # Option leverage check — significantly better return at fraction of capital.
    # Use absolute values so bearish trades (negative stock P&L) compare correctly.
    abs_stock_pnl = abs(stock.target_pnl)
    abs_opt_pnl   = abs(best_option.target_pnl)
    if (abs_opt_pnl > abs_stock_pnl * 2.5
            and best_option.cost <= stock.cost * 0.5):
        return (
            'OPTION_PREFERRED',
            f'Option offers {abs_opt_pnl / abs_stock_pnl:.1f}x leverage '
            f'at {best_option.cost / stock.cost * 100:.0f}% the capital.'
        )

    # High IV: vol crush risk dominates directional gain
    if best_option.iv_at_entry > 0.50:
        return (
            'STOCK_PREFERRED',
            f'Option IV {best_option.iv_at_entry*100:.0f}% is elevated — '
            f'vol crush risk eats directional gain.'
        )

    # Small target relative to option cost
    if abs(stock.target_pct) < 0.05 and best_option.cost > stock.cost * 0.4:
        return (
            'STOCK_PREFERRED',
            f'Target only {abs(stock.target_pct)*100:.1f}% away — '
            f'option cost too high vs move size.'
        )

    # Option R:R worse than stock
    if best_option.rr_ratio < stock.rr_ratio:
        return (
            'STOCK_PREFERRED',
            f'Option R:R {best_option.rr_ratio:.2f} worse than stock R:R {stock.rr_ratio:.2f}.'
        )

    # Option R:R materially better than stock
    if best_option.rr_ratio >= stock.rr_ratio * 1.5:
        return (
            'OPTION_PREFERRED',
            f'Option R:R {best_option.rr_ratio:.2f} materially better than stock {stock.rr_ratio:.2f}.'
        )

    return (
        'CLOSE_CALL',
        f'Both routes viable. Stock R:R {stock.rr_ratio:.2f}, '
        f'option R:R {best_option.rr_ratio:.2f}. Trader preference.'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_routes(
    ticker: str,
    direction: Literal['bullish', 'bearish'],
    target: float,
    stop: float,
    hold_days: int,
    account_balance: float,
    risk_free_rate: float = 0.053,
    sources: Optional[list] = None,
) -> TradeComparison:
    """Pull option chain, project stock and option routes, return comparison.

    Args:
        ticker:           Equity ticker (e.g. 'KLAC').
        direction:        'bullish' or 'bearish'.
        target:           Price target.
        stop:             Stop loss level.
        hold_days:        Expected hold duration in calendar days.
        account_balance:  Available capital for the trade in dollars.
        risk_free_rate:   Risk-free rate for Black-Scholes (default 5.3%).
        sources:          Optional list of DataSource objects; if None, uses
                          default_sources() from data.sources.factory.

    Returns:
        TradeComparison with stock_route, option_routes (top 3), verdict, and
        human-readable verdict_reason.

    Raises:
        DataFetchError: If all data sources fail to return a spot price.
    """
    if sources is None:
        sources = default_sources()

    # ---- Fetch spot -------------------------------------------------------
    try:
        spot = fetch_with_fallback(sources, 'fetch_spot', ticker)
    except DataFetchError:
        spot = None

    # ---- Fetch chain -------------------------------------------------------
    try:
        chain = fetch_with_fallback(sources, 'fetch_option_chain', ticker)
    except DataFetchError as exc:
        log.warning("Could not fetch option chain for %s: %s", ticker, exc)
        chain = []

    # Fall back: derive spot from chain if fetch_spot failed
    if spot is None:
        if chain:
            spot = chain[0].spot_price
            log.info("Using spot from chain for %s: %.2f", ticker, spot)
        else:
            raise DataFetchError(
                f"Cannot determine spot price for {ticker} — all sources failed."
            )

    spot = float(spot)

    # ---- Stock route -------------------------------------------------------
    stock_route = _build_stock_route(ticker, spot, target, stop, account_balance)

    # ---- Option candidates -------------------------------------------------
    candidates = _select_candidate_contracts(
        chain, spot, direction, target, hold_days, account_balance
    )

    option_routes_raw: list[OptionRoute] = []
    for contract in candidates:
        try:
            route = _build_option_route(
                contract=contract,
                spot=spot,
                target=target,
                stop=stop,
                hold_days=hold_days,
                risk_free_rate=risk_free_rate,
                stock_target_pct=stock_route.target_pct,
            )
            option_routes_raw.append(route)
        except Exception as exc:
            log.debug("Skipping contract %s@%s: %s", contract.strike, contract.expiration, exc)

    # Rank by R:R descending, take top 3
    option_routes_raw.sort(key=lambda r: -r.rr_ratio)
    top_options = option_routes_raw[:3]

    # ---- Verdict -----------------------------------------------------------
    verdict_str, reason = _decide_verdict(stock_route, top_options)
    verdict: Literal['STOCK_PREFERRED', 'OPTION_PREFERRED', 'CLOSE_CALL'] = verdict_str  # type: ignore[assignment]

    return TradeComparison(
        ticker=ticker,
        direction=direction,
        spot=round(spot, 2),
        target=round(target, 2),
        stop=round(stop, 2),
        hold_days=hold_days,
        account_balance=account_balance,
        stock_route=stock_route,
        option_routes=top_options,
        verdict=verdict,
        verdict_reason=reason,
    )
