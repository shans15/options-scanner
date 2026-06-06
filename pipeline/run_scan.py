from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import logging
import numpy as np

from data.sources.base import DataSource
from data.sources.yahooquery_source import YahooQuerySource
from data.sources.yfinance_source import YfinanceSource
from data.sources.stooq_source import StooqSource
from data.fallback import fetch_with_fallback, DataFetchError
from data.adapters import to_contract

from domain.signals import compute_regime
from domain.strategy import ALL_STRATEGIES, NakedPut, NakedCall, LongPut, LongCall
from domain.technical_signals import TechnicalSetup
from domain.market_context import compute_market_context
from pipeline.technical_filter import filter_by_technicals

from engine.pop_models import pop_delta, pop_black_scholes, pop_historical, pop_garch_mc, blend_pop
from engine.stress import compute_stress
from engine.risk_filters import apply_filters
from engine.scorer import composite_score, label_from, ScoredCandidate

from pipeline.universe_builder import build_universe_cached, UniverseFilters
from pipeline.earnings import has_earnings_within


log = logging.getLogger(__name__)


def _strategy_matches_setup(strategy, setup_dirs: set[str]) -> bool:
    """Map strategy → required setup direction.
    - LongCall, NakedPut are bullish-direction bets.
    - LongPut, NakedCall are bearish-direction bets.
    """
    if isinstance(strategy, (LongCall, NakedPut)):
        return 'bullish' in setup_dirs
    if isinstance(strategy, (LongPut, NakedCall)):
        return 'bearish' in setup_dirs
    return False


def _strongest_matching_setup(setups: list[TechnicalSetup], strategy) -> TechnicalSetup | None:
    target_dir = 'bullish' if isinstance(strategy, (LongCall, NakedPut)) else 'bearish'
    matches = [s for s in setups if s.direction == target_dir]
    if not matches:
        return None
    return max(matches, key=lambda s: s.strength)


@dataclass
class ScanConfig:
    risk_free_rate: float = 0.053
    earnings_blackout_days: int = 5
    n_monte_carlo_paths: int = 10_000
    today: date = field(default_factory=date.today)
    universe_filters: UniverseFilters = field(default_factory=UniverseFilters)
    use_technical_filter: bool = True
    as_of: Optional[date] = None     # NEW


@dataclass
class ScanResult:
    timestamp: datetime
    config: ScanConfig
    candidates: list[ScoredCandidate]
    skipped: dict[str, str]


def _expected_profit_when_itm_long(terminal_prices: np.ndarray, contract, strategy) -> float:
    matches = np.array([strategy.profit_condition(float(p), contract) for p in terminal_prices])
    if not matches.any():
        return 0.0
    if contract.option_type == 'call':
        intrinsic = np.maximum(terminal_prices[matches] - contract.strike, 0.0)
    else:
        intrinsic = np.maximum(contract.strike - terminal_prices[matches], 0.0)
    return float(intrinsic.mean() - contract.mid)


def run_scan(config: ScanConfig, sources_override: Optional[list[DataSource]] = None) -> ScanResult:
    sources = sources_override if sources_override is not None else [
        YahooQuerySource(), YfinanceSource(), StooqSource()
    ]
    universe = build_universe_cached(config.universe_filters, sources)
    candidates: list[ScoredCandidate] = []
    skipped: dict[str, str] = {}

    if config.use_technical_filter:
        setups_by_ticker = filter_by_technicals(universe, sources, as_of=config.as_of)
        ticker_iter = list(setups_by_ticker.keys())
    else:
        setups_by_ticker = {t: [] for t in universe}
        ticker_iter = universe

    for ticker in ticker_iter:
        try:
            if has_earnings_within(ticker, config.earnings_blackout_days):
                skipped[ticker] = 'earnings_blackout'
                continue
            history = fetch_with_fallback(sources, 'fetch_price_history', ticker, 365)
            if config.as_of is not None and len(history) > 0:
                history = history[history.index.date <= config.as_of]
            spot = fetch_with_fallback(sources, 'fetch_spot', ticker)
            if config.as_of is not None and len(history) > 0:
                spot = float(history.iloc[-1])
            raw_chain = fetch_with_fallback(sources[:2], 'fetch_option_chain', ticker)
        except DataFetchError as e:
            skipped[ticker] = f'fetch_failed: {e}'
            continue

        if history is None or len(history) < 30 or spot <= 0:
            skipped[ticker] = 'insufficient_history'
            continue

        log_returns = np.log(history / history.shift(1)).dropna()
        regime = compute_regime(ticker, log_returns, raw_chain)
        setups = setups_by_ticker.get(ticker, [])
        setup_dirs = {s.direction for s in setups}

        context = compute_market_context(history)

        for strategy in ALL_STRATEGIES:
            if config.use_technical_filter:
                if not _strategy_matches_setup(strategy, setup_dirs):
                    continue
            else:
                if strategy.direction not in regime.favored:
                    continue
            for raw in raw_chain:
                if raw.option_type != strategy.option_type:
                    continue
                contract = to_contract(raw, spot, config.risk_free_rate, config.today)
                if contract is None or not strategy.applies_to(contract):
                    continue

                p_d = pop_delta(contract, strategy)
                p_bs = pop_black_scholes(contract, strategy, config.risk_free_rate)
                p_hist = pop_historical(contract, strategy, log_returns)
                p_garch = pop_garch_mc(contract, strategy, log_returns, n_paths=config.n_monte_carlo_paths)
                p_blend = blend_pop(p_d, p_bs, p_hist, p_garch)

                stress = compute_stress(contract, strategy, contract.implied_volatility)
                max_adverse = abs(min(stress.stress_2sd, 0.0))

                if strategy.direction == 'sell':
                    ev = strategy.expected_value(contract, p_blend, max_adverse)
                else:
                    np.random.seed(42)
                    daily_vol = float(log_returns.std())
                    z = np.random.standard_normal((2000, contract.dte))
                    cumulative = (daily_vol * z).sum(axis=1)
                    terminal = contract.spot_price * np.exp(cumulative)
                    expected_profit_itm = _expected_profit_when_itm_long(terminal, contract, strategy)
                    ev = strategy.expected_value(contract, p_blend, max_adverse, expected_profit_when_itm=expected_profit_itm)

                fr = apply_filters(contract, strategy, p_blend, ev, stress)
                score = composite_score(
                    pop_blended=p_blend, ev=ev, max_adverse_loss=max_adverse,
                    strategy=strategy, regime=regime, mid=contract.mid,
                )
                label = label_from(score, fr.passed)

                chosen = _strongest_matching_setup(setups, strategy) if setups else None

                reason_for = (
                    f"PoP={p_blend:.0%}, EV=${ev:.2f}, regime ratio={regime.rv_iv_ratio:.2f} "
                    f"favors {strategy.direction}"
                    + (f", setup={chosen.setup_name} ({chosen.direction})" if chosen else "")
                )
                reason_against = (
                    f"Filters failed: {', '.join(fr.failed_filters)}" if fr.failed_filters else ""
                )

                candidates.append(ScoredCandidate(
                    contract=contract, strategy=strategy,
                    pop_blended=p_blend, pop_delta=p_d, pop_bs=p_bs,
                    pop_historical=p_hist, pop_garch_mc=p_garch,
                    stress=stress, ev=ev, max_adverse_loss=max_adverse,
                    margin_estimate=strategy.margin_estimate(contract),
                    filter_result=fr, composite_score=score, label=label,
                    reason_for=reason_for, reason_against=reason_against,
                    setup_name=chosen.setup_name if chosen else None,
                    setup_direction=chosen.direction if chosen else None,
                    setup_strength=chosen.strength if chosen else None,
                    weekly_trend=context.weekly_trend,
                    consecutive_close_streak=context.consecutive_close_streak,
                    pct_change_1w=context.pct_change_1w,
                    pct_change_2w=context.pct_change_2w,
                    pct_change_4w=context.pct_change_4w,
                    weekly_ribbon_agreement=context.weekly_ribbon_agreement,
                ))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return ScanResult(timestamp=datetime.now(), config=config,
                      candidates=candidates, skipped=skipped)
