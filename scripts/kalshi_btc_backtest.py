"""3-strategy backtest comparison: A (vol-model only) vs B (research-paper ensemble)
vs A+B (combined).

Sprint 2 improvements:
  - Fetch real market-implied probability from get_market_history (last yes_ask
    mid in the 30 min before close) instead of using the stale yes_bid from the
    settled-market record.
  - Kalshi 7% fee on winning trades (deducted from gross profit).
  - Wilson 95% confidence interval on win rates.
  - 90-day default window with --max-markets flag (default 5000), stratified
    daily sampling.
  - Per-trade CSV with fee breakdown columns.
  - Progress prints every 100 markets (tqdm if available, otherwise plain).

CLI
---
    python3 -m scripts.kalshi_btc_backtest \\
        --start 2026-03-08 --end 2026-06-06 \\
        --max-markets 5000 \\
        --position-size 50

Output
------
  Console: comparison table with Wilson CIs and statistical comparison
  CSV:     output/kalshi/backtest_YYYYMMDD_HHMM.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.kalshi_source import KalshiSource
from data.sources.coinbase_source import CoinbaseSource
from domain.prediction.market import KalshiMarket, parse_kalshi_market
from engine.prediction.ensemble import (
    compute_strategy_A,
    compute_strategy_B,
    compute_strategy_AB,
    count_estimator_agreement,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_EDGE = 0.05
_KALSHI_FEE_RATE = 0.07  # 7% of profit on wins
_OUTPUT_DIR = _ROOT / "output" / "kalshi"
_DEFAULT_MAX_MARKETS = 5000
_DEFAULT_LOOKBACK_DAYS = 90

# History window fetched per market: 30 minutes before close
_HISTORY_WINDOW_MS = 30 * 60 * 1000  # 30 min in ms

# Historical vol quantile keys
_Q = {0.25: 0.25, 0.5: 0.5, 0.75: 0.75}


def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% confidence interval for binomial win rate.

    Used in the backtest to show whether A vs B vs A+B differences are real.
    Two strategies are statistically distinguishable only if their CIs don't overlap.
    """
    if trials == 0:
        return (0.0, 1.0)
    z = 1.96  # 95% two-sided
    p = wins / trials
    n = trials
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_ohlcv_at(ohlcv: pd.DataFrame, as_of: datetime, lookback_hours: int) -> pd.DataFrame:
    """Return hourly OHLCV rows ending at (but not after) `as_of`."""
    cutoff = pd.Timestamp(as_of).tz_localize(None) if as_of.tzinfo and ohlcv.index.tzinfo is None else as_of
    past = ohlcv[ohlcv.index <= cutoff]
    return past.iloc[-lookback_hours:] if len(past) >= 1 else past


def _vol_quantiles(hourly_returns: pd.Series) -> dict[float, float]:
    hv = hourly_returns.rolling(24).std() * np.sqrt(24)
    hv = hv.dropna()
    if hv.empty:
        return {0.25: 0.01, 0.5: 0.02, 0.75: 0.03}
    return {
        0.25: float(hv.quantile(0.25)),
        0.5:  float(hv.quantile(0.5)),
        0.75: float(hv.quantile(0.75)),
    }


def _pnl_per_dollar(fair_prob: float, market_prob: float, resolved_yes: bool) -> tuple[float, float, float]:
    """Compute realized P&L per dollar risked, with fee breakdown.

    Returns (pnl_per_dollar, gross_pnl_per_dollar, fee_per_dollar).
    BUY YES if fair > market_prob, BUY NO (sell YES) if fair < market_prob.
    Payout = $1 on win (binary). Fee = 7% of gross profit on wins only.
    """
    if abs(fair_prob - market_prob) < _MIN_EDGE:
        return 0.0, 0.0, 0.0  # no trade

    if fair_prob > market_prob:  # BUY YES
        cost = market_prob
        if resolved_yes:
            gross = 1.0 - market_prob
            fee = gross * _KALSHI_FEE_RATE
            net = gross - fee
        else:
            gross = -market_prob
            fee = 0.0
            net = gross
    else:  # BUY NO
        cost = 1.0 - market_prob
        if not resolved_yes:
            gross = market_prob
            fee = gross * _KALSHI_FEE_RATE
            net = gross - fee
        else:
            gross = -(1.0 - market_prob)
            fee = 0.0
            net = gross

    per_dollar = net / cost if cost > 0 else 0.0
    gross_per_dollar = gross / cost if cost > 0 else 0.0
    fee_per_dollar = fee / cost if cost > 0 else 0.0
    return per_dollar, gross_per_dollar, fee_per_dollar


def _sharpe(pnl_series: list[float]) -> float:
    if len(pnl_series) < 2:
        return 0.0
    arr = np.array(pnl_series)
    std = arr.std()
    return float(arr.mean() / std) if std > 0 else 0.0


def _max_drawdown(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return float(drawdown.min())


def _ci_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """Return True if two confidence intervals overlap."""
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def _print_summary(
    results: list[dict],
    start_str: str,
    end_str: str,
    n_resolved: int,
    n_analyzed: int,
    n_skipped: int,
    position_size: float,
) -> None:
    """Print the full Sprint 2 summary table."""
    print(f"\n{'='*70}")
    print(f"=== Kalshi BTC Backtest: {start_str} → {end_str} ===")
    print(f"{'='*70}")
    print(f"\nMarkets resolved:    {n_resolved:,}")
    print(f"Markets analyzed:    {n_analyzed:,}")
    print(f"Markets skipped:     {n_skipped:,} (no orderbook history at close time)")
    print(f"\nPosition size:       ${position_size:.0f}")
    print(f"Fee rate:            {_KALSHI_FEE_RATE*100:.0f}% of profit on wins")

    # Header
    print(f"\n{'─'*100}")
    print(
        f"{'Strategy':<22} {'Trades':>7} {'Win% [95% CI]':<22} "
        f"{'Mean P&L':>10} {'Total P&L':>11} {'Sharpe':>8}"
    )
    print(f"{'─'*100}")

    for r in results:
        ci_lo, ci_hi = r["wilson_ci"]
        ci_str = f"[{ci_lo*100:.1f}, {ci_hi*100:.1f}]"
        win_ci = f"{r['win_pct']:.1f}% {ci_str}"
        total_pnl_dollars = r["total_pnl_dollars"]
        mean_pnl_dollars = r["mean_pnl_dollars"]
        print(
            f"  {r['strategy']:<20} {r['n_trades']:>7}  {win_ci:<22} "
            f"  {mean_pnl_dollars:>+7.2f}   {total_pnl_dollars:>+9.0f}   {r['sharpe']:>7.2f}"
        )
    print(f"{'─'*100}")

    # Statistical comparison
    strategy_map = {r["strategy_key"]: r for r in results}
    r_A = strategy_map.get("A")
    r_B = strategy_map.get("B")
    r_AB = strategy_map.get("AB")

    if r_A and r_B and r_AB:
        print("\nStatistical comparison (95% CI overlap test):")
        pairs = [
            ("A", "B", r_A, r_B),
            ("A", "A+B", r_A, r_AB),
            ("B", "A+B", r_B, r_AB),
        ]
        for name1, name2, res1, res2 in pairs:
            overlap = _ci_overlap(res1["wilson_ci"], res2["wilson_ci"])
            if overlap:
                result_str = "overlap → no significant difference"
            else:
                # Which is better?
                winner = name1 if res1["win_pct"] > res2["win_pct"] else name2
                result_str = f"no overlap → {winner} wins"
            print(f"  {name1} vs {name2}:{'':<5} {result_str}")

        # Verdict
        best = max(results, key=lambda r: r["total_pnl_dollars"])
        print(f"\nVerdict: {best['strategy']} wins with "
              f"${best['total_pnl_dollars']:+.0f} total P&L, Sharpe {best['sharpe']:.2f}")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Stratified daily sampling
# ---------------------------------------------------------------------------

def _stratified_sample(markets: list[KalshiMarket], max_markets: int) -> list[KalshiMarket]:
    """Stratify by close_time.date() and sample max_markets / n_days per day.

    Also tries to balance above/below sides within each day.
    Returns at most max_markets markets.
    """
    if len(markets) <= max_markets:
        return markets

    # Group by date
    from collections import defaultdict
    by_date: dict = defaultdict(list)
    for m in markets:
        by_date[m.close_time.date()].append(m)

    n_days = len(by_date)
    per_day = max(1, max_markets // n_days)

    sampled: list[KalshiMarket] = []
    rng = np.random.default_rng(42)  # reproducible sampling

    for date, day_markets in sorted(by_date.items()):
        if len(day_markets) <= per_day:
            sampled.extend(day_markets)
            continue
        # Balance above/below
        above = [m for m in day_markets if m.side == "above"]
        below = [m for m in day_markets if m.side == "below"]
        n_above = min(len(above), per_day // 2)
        n_below = min(len(below), per_day - n_above)
        # Adjust if one side is short
        if n_above < per_day // 2:
            n_below = min(len(below), per_day - n_above)
        if n_below < per_day // 2:
            n_above = min(len(above), per_day - n_below)

        chosen_above = list(rng.choice(len(above), size=n_above, replace=False))
        chosen_below = list(rng.choice(len(below), size=n_below, replace=False))
        sampled.extend([above[i] for i in chosen_above])
        sampled.extend([below[i] for i in chosen_below])

    # Final cap
    if len(sampled) > max_markets:
        idx = rng.choice(len(sampled), size=max_markets, replace=False)
        sampled = [sampled[i] for i in idx]

    return sampled


# ---------------------------------------------------------------------------
# Market history fetch with per-market cache
# ---------------------------------------------------------------------------

def _fetch_market_implied_prob(
    ks: KalshiSource,
    ticker: str,
    close_ts_ms: int,
    idx: int,
    total: int,
) -> Optional[float]:
    """Fetch the last yes_ask mid in the 30 min before close as market-implied prob.

    Uses a per-market Parquet cache so repeated runs are free.
    Returns None if history is empty (illiquid — caller should skip the market).
    """
    # Per-market cache: uses only the ticker so re-runs with different windows
    # still reuse the same file (window is fixed to 30 min before close).
    cache_key = f"hist_{ticker}_close"
    cache_file = ks._cache_dir / f"{cache_key}.parquet"

    t0 = time.time()
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        elapsed = time.time() - t0
        _progress_print(idx, total, ticker, cache_hit=True, elapsed=elapsed)
    else:
        start_ms = close_ts_ms - _HISTORY_WINDOW_MS
        df = ks.get_market_history(ticker, start_ms, close_ts_ms)
        # get_market_history writes its own cache with a different key name;
        # write a second, per-ticker cache for re-runs regardless of window param.
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass
        elapsed = time.time() - t0
        _progress_print(idx, total, ticker, cache_hit=False, elapsed=elapsed)

    if df.empty:
        return None

    # Use the mid of yes_bid and yes_ask from the last candlestick before close.
    # get_market_history (via candlesticks endpoint) normalises prices to [0, 1] float.
    # If bid is 0 and ask > 0, use ask as a conservative estimate (market is illiquid).
    last = df.iloc[-1]
    bid = float(last["yes_bid"]) if not pd.isna(last["yes_bid"]) else 0.0
    ask = float(last["yes_ask"]) if not pd.isna(last["yes_ask"]) else 0.0

    if ask <= 0.0:
        return None  # no quote at all — skip

    # If spread is too wide (bid=0, ask=1), the market is effectively unquoted — skip
    if bid == 0.0 and ask >= 0.99:
        return None

    prob = (bid + ask) / 2.0 if bid > 0.0 else ask
    if prob <= 0.0 or prob >= 1.0:
        return None
    # Note: extreme prob filtering (< 0.05 or > 0.95) is done in the main loop
    # after this function returns, so callers can log the skip reason separately.
    return prob


_PROGRESS_COUNTER = {"n": 0}


def _progress_print(idx: int, total: int, ticker: str, cache_hit: bool, elapsed: float) -> None:
    if idx % 100 == 0 or idx <= 5:
        status = "cache hit" if cache_hit else "cache miss"
        print(f"  [{idx:>{len(str(total))}}/{total}] fetched market history {ticker} "
              f"({status}, {elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    settled_markets: list[KalshiMarket],
    ohlcv_1h: pd.DataFrame,
    position_size: float,
    kalshi_source: Optional[KalshiSource] = None,
) -> tuple[list[dict], list[dict], int]:
    """Run 3-strategy backtest over settled_markets.

    When kalshi_source is provided, fetches real market-implied probability from
    the orderbook history for each market.  When None (unit tests), falls back to
    the market.market_implied_prob from the settled record.

    Returns (per_trade_rows, summary_rows, n_skipped).
    Each per_trade row has full fee breakdown columns.
    """
    per_trade: list[dict] = []
    strategy_pnl: dict[str, list[float]] = {"A": [], "B": [], "AB": []}
    strategy_wins: dict[str, int] = {"A": 0, "B": 0, "AB": 0}
    strategy_trades: dict[str, int] = {"A": 0, "B": 0, "AB": 0}
    strategy_edges: dict[str, list[float]] = {"A": [], "B": [], "AB": []}
    strategy_pnl_dollars: dict[str, list[float]] = {"A": [], "B": [], "AB": []}

    total = len(settled_markets)
    n_skipped = 0

    for i, m in enumerate(settled_markets, 1):
        if m.resolution not in ("yes", "no"):
            continue

        resolved_yes = m.resolution == "yes"

        # ------------------------------------------------------------------
        # Market-implied probability: real orderbook history or fallback
        # ------------------------------------------------------------------
        if kalshi_source is not None:
            close_ts_ms = int(m.close_time.timestamp() * 1000)
            market_prob = _fetch_market_implied_prob(
                kalshi_source, m.ticker, close_ts_ms, i, total
            )
            if market_prob is None:
                n_skipped += 1
                continue
        else:
            # Unit-test path: use the settled-record mid (may be stale but tests don't care)
            market_prob = m.market_implied_prob
            if market_prob <= 0 or market_prob >= 1:
                continue

        # Skip markets where the market consensus is already extreme (< 5% or > 95%).
        # These are deep OTM markets where our vol-model is miscalibrated and the
        # "edge" signal is just model error, not genuine mispricing.
        if market_prob < 0.05 or market_prob > 0.95:
            n_skipped += 1
            continue

        # ------------------------------------------------------------------
        # Historical OHLCV slice at market close
        # ------------------------------------------------------------------
        as_of = m.close_time
        hourly_slice = _slice_ohlcv_at(ohlcv_1h, as_of, 30 * 24)
        if len(hourly_slice) < 50:
            continue

        spot = float(hourly_slice["close"].iloc[-1])
        hourly_returns = np.log(hourly_slice["close"]).diff().dropna()
        quantiles = _vol_quantiles(hourly_returns)
        last_hour_return = float(hourly_returns.iloc[-1]) if len(hourly_returns) > 0 else 0.0
        hv_24 = hourly_returns.rolling(24).std() * np.sqrt(24)
        hourly_sigma = float(hv_24.iloc[-1]) if len(hv_24) > 0 and not np.isnan(hv_24.iloc[-1]) else 0.02

        hours_to_expiry = (m.expiration - as_of).total_seconds() / 3600.0
        if hours_to_expiry <= 0:
            hours_to_expiry = 1.0  # settled market: use 1h as floor

        # Empty 1-min series (not needed for backtest — RV estimator returns 0.5)
        empty_1min = pd.Series(dtype=float)

        # Strategy A
        out_A = compute_strategy_A(
            spot=spot, strike=m.strike, hours_to_expiry=hours_to_expiry,
            vol_proba=0.5,
            historical_vol_quantiles=quantiles,
            expiration_utc=m.expiration, last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        fp_A = out_A.fair_prob_adjusted
        if m.side == "below":
            fp_A = 1.0 - fp_A

        # Strategy B
        out_B = compute_strategy_B(
            spot=spot, strike=m.strike, hours_to_expiry=hours_to_expiry,
            hourly_returns=hourly_returns, minute_returns_last_60=empty_1min,
            expiration_utc=m.expiration, last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        fp_B = out_B.fair_prob_adjusted
        if m.side == "below":
            fp_B = 1.0 - fp_B

        # Strategy A+B
        out_AB = compute_strategy_AB(
            spot=spot, strike=m.strike, hours_to_expiry=hours_to_expiry,
            vol_proba=0.5, historical_vol_quantiles=quantiles,
            hourly_returns=hourly_returns, minute_returns_last_60=empty_1min,
            expiration_utc=m.expiration, last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        fp_AB = out_AB.fair_prob_adjusted
        if m.side == "below":
            fp_AB = 1.0 - fp_AB

        for strategy, fp in [("A", fp_A), ("B", fp_B), ("AB", fp_AB)]:
            edge = fp - market_prob
            if abs(edge) < _MIN_EDGE:
                continue

            net_pnl, gross_pnl, fee = _pnl_per_dollar(fp, market_prob, resolved_yes)
            won = net_pnl > 0

            strategy_trades[strategy] += 1
            strategy_pnl[strategy].append(net_pnl)
            strategy_pnl_dollars[strategy].append(net_pnl * position_size)
            strategy_edges[strategy].append(edge)
            if won:
                strategy_wins[strategy] += 1

            per_trade.append({
                "strategy": strategy,
                "market_ticker": m.ticker,
                "close_time": as_of.isoformat(),
                "side": m.side,
                "strike": m.strike,
                "spot_at_open": spot,
                "hours_to_expiry": round(hours_to_expiry, 2),
                "market_implied_prob": round(market_prob, 4),
                "fair_prob": round(fp, 4),
                "edge": round(edge, 4),
                "position_size": position_size,
                "won": won,
                "gross_pnl": round(gross_pnl * position_size, 4),
                "kalshi_fee": round(fee * position_size, 4),
                "net_pnl": round(net_pnl * position_size, 4),
                "cum_pnl": 0.0,  # filled in below
            })

    # Compute cumulative P&L per strategy (in order)
    running: dict[str, float] = {"A": 0.0, "B": 0.0, "AB": 0.0}
    for row in per_trade:
        s = row["strategy"]
        running[s] += row["net_pnl"]
        row["cum_pnl"] = round(running[s], 4)

    # Summary with Wilson CIs
    summary = []
    for strat_key, label in [("A", "A: vol-model only"), ("B", "B: research ens."), ("AB", "A+B: combined")]:
        n = strategy_trades[strat_key]
        wins = strategy_wins[strat_key]
        pnl_list = strategy_pnl[strat_key]
        pnl_dollars_list = strategy_pnl_dollars[strat_key]
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        mean_edge = float(np.mean(strategy_edges[strat_key])) if strategy_edges[strat_key] else 0.0
        total_pnl_dollars = sum(pnl_dollars_list)
        mean_pnl_dollars = float(np.mean(pnl_dollars_list)) if pnl_dollars_list else 0.0
        sharpe = _sharpe(pnl_list)
        max_dd = _max_drawdown(pnl_list)
        summary.append({
            "strategy_key": strat_key,
            "strategy": label,
            "n_trades": n,
            "win_pct": win_pct,
            "wilson_ci": ci,
            "mean_edge": mean_edge,
            "total_pnl_dollars": total_pnl_dollars,
            "mean_pnl_dollars": mean_pnl_dollars,
            "sharpe": sharpe,
            "max_dd": max_dd,
        })

    return per_trade, summary, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Default window: 90 days back from today
    _today = datetime.now(timezone.utc).date()
    _default_end = _today.strftime("%Y-%m-%d")
    _default_start = (_today - timedelta(days=_DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description="Kalshi BTC 3-strategy backtest (Sprint 2)")
    parser.add_argument("--start", default=_default_start, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=_default_end, help="End date YYYY-MM-DD")
    parser.add_argument("--position-size", type=float, default=50.0, help="Dollars per trade")
    parser.add_argument(
        "--max-markets", type=int, default=_DEFAULT_MAX_MARKETS,
        help=f"Max markets to analyze (default {_DEFAULT_MAX_MARKETS}); stratified by day"
    )
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    position_size = args.position_size
    max_markets = args.max_markets

    print(f"\n{'='*70}")
    print(f"=== Kalshi BTC Backtest (Sprint 2): {args.start} → {args.end} ===")
    print(f"{'='*70}\n")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # ------------------------------------------------------------------
    # Fetch settled markets
    # ------------------------------------------------------------------
    print("Fetching settled Kalshi BTC markets …")
    try:
        ks = KalshiSource()
        raw = ks.get_settled_markets(start_ms, end_ms)
    except Exception as exc:
        print(f"ERROR: Kalshi fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    markets = [parse_kalshi_market(m) for m in raw]
    resolved = [m for m in markets if m.resolution in ("yes", "no")]
    print(f"  {len(markets)} settled markets, {len(resolved)} with resolution")

    if not resolved:
        print("No resolved liquid markets in range. Nothing to backtest.")
        return 0

    # ------------------------------------------------------------------
    # Fetch OHLCV early for moneyness-based pre-filter
    # ------------------------------------------------------------------
    print("Fetching historical BTC 1h OHLCV for moneyness pre-filter …")
    try:
        cb = CoinbaseSource()
        warmup_ms = int((start_dt - timedelta(days=31)).timestamp() * 1000)
        ohlcv_1h = cb.fetch_ohlcv("BTC-USD", "1h", warmup_ms, end_ms)
    except Exception as exc:
        print(f"ERROR: Coinbase fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    print(f"  OHLCV rows: {len(ohlcv_1h)}")

    # Pre-filter: keep only markets where strike is within ±15% of spot at close.
    # This targets near-the-money markets where probability estimates are meaningful
    # and Kalshi prices reflect genuine information (not artefact listings).
    def _moneyness_ok(m: KalshiMarket) -> bool:
        """True if this market's strike is within ±5% of spot at close.

        Within 5% of spot, BTC hourly markets should have meaningful price
        discovery (probability 10%–90%) and our vol-model fair estimates are
        calibrated to be informative.
        """
        slice_df = _slice_ohlcv_at(ohlcv_1h, m.close_time, 2)
        if slice_df.empty:
            return True  # can't filter, let it through
        spot = float(slice_df["close"].iloc[-1])
        if spot <= 0:
            return True
        moneyness = abs(m.strike / spot - 1.0)
        return moneyness <= 0.05

    n_before = len(resolved)
    resolved = [m for m in resolved if _moneyness_ok(m)]
    print(f"  {n_before} → {len(resolved)} after moneyness filter (strike within ±5% of spot)")

    if not resolved:
        print("No near-money markets in range. Nothing to backtest.")
        return 0

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------
    if len(resolved) > max_markets:
        print(f"  Sampling {max_markets} from {len(resolved)} markets (stratified by day) …")
        resolved = _stratified_sample(resolved, max_markets)
        print(f"  Sampled {len(resolved)} markets")

    # ------------------------------------------------------------------
    # Run backtest (with real market history per market)
    # ------------------------------------------------------------------
    print(f"\nFetching per-market orderbook history and running backtest …")
    print(f"  {len(resolved)} markets × ~5 req/sec = ~{len(resolved)//5 + 1}s fetch time")
    print(f"  Progress printed every 100 markets\n")

    try:
        per_trade, summary, n_skipped = run_backtest(
            resolved, ohlcv_1h, position_size, kalshi_source=ks
        )
    except Exception as exc:
        print(f"ERROR: Backtest failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    n_analyzed = len(resolved) - n_skipped
    total_trades = sum(r["n_trades"] for r in summary)
    print(f"\n  Total trades taken across all strategies: {total_trades}")
    print(f"  Markets skipped (no orderbook history): {n_skipped}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    _print_summary(
        summary,
        start_str=args.start,
        end_str=args.end,
        n_resolved=len(markets),
        n_analyzed=n_analyzed,
        n_skipped=n_skipped,
        position_size=position_size,
    )

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    csv_path = _OUTPUT_DIR / f"backtest_{ts_str}.csv"

    fieldnames = [
        "strategy", "market_ticker", "close_time", "side", "strike",
        "spot_at_open", "hours_to_expiry", "market_implied_prob", "fair_prob",
        "edge", "position_size", "won", "gross_pnl", "kalshi_fee", "net_pnl", "cum_pnl",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_trade)

    print(f"Per-trade detail saved → {csv_path}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
