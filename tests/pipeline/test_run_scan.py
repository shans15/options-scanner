from datetime import date, datetime
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from data.sources.base import DataSource, RawContract
from pipeline.run_scan import run_scan, ScanConfig, ScanResult


class _FakeSource(DataSource):
    def __init__(self, history: pd.Series, spot: float, chain: list[RawContract], ohlcv=None):
        self._history = history; self._spot = spot; self._chain = chain; self._ohlcv = ohlcv
    def fetch_spot(self, ticker): return self._spot
    def fetch_price_history(self, ticker, lookback_days): return self._history
    def fetch_option_chain(self, ticker): return self._chain
    def fetch_price_history_ohlcv(self, ticker, lookback_days):
        if self._ohlcv is None:
            return pd.DataFrame()
        return self._ohlcv


def _make_raw(option_type='put', strike=100.0, dte=21, iv=0.20, mid=2.0):
    return RawContract(
        ticker='X', expiration=date(2026, 5, 30) + pd.Timedelta(days=dte).to_pytimedelta(),
        strike=strike, option_type=option_type, bid=mid-0.05, ask=mid+0.05,
        volume=1000, open_interest=1000, implied_volatility=iv, dte=dte, spot_price=100.0,
    )


def test_run_scan_returns_scan_result_with_candidates():
    bull_df = _bullish_ohlcv_history()
    chain = [_make_raw(option_type='put', strike=98, mid=1.0),
             _make_raw(option_type='call', strike=102, mid=1.0)]
    src = _FakeSource(history=bull_df['close'], spot=100.0, chain=chain, ohlcv=bull_df)

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[src])

    assert isinstance(result, ScanResult)
    assert result.skipped == {} or 'X' not in result.skipped
    assert isinstance(result.candidates, list)
    assert len(result.candidates) > 0


def test_run_scan_skips_tickers_with_earnings():
    history = pd.Series([100.0] * 100)
    src = _FakeSource(history, spot=100.0, chain=[])

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=True):
        result = run_scan(ScanConfig(today=date(2026, 5, 30), use_technical_filter=False), sources_override=[src])

    assert result.skipped == {'X': 'earnings_blackout'}
    assert result.candidates == []


def test_run_scan_records_fetch_failure_per_ticker():
    class _Broken(DataSource):
        def fetch_spot(self, t): raise RuntimeError('no')
        def fetch_price_history(self, t, lb): raise RuntimeError('no')
        def fetch_option_chain(self, t): raise RuntimeError('no')
        def fetch_price_history_ohlcv(self, t, lb): raise RuntimeError('no')

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(ScanConfig(today=date(2026, 5, 30), use_technical_filter=False), sources_override=[_Broken()])

    assert 'X' in result.skipped
    assert 'fetch_failed' in result.skipped['X']


def _bullish_ohlcv_history() -> pd.DataFrame:
    np.random.seed(99)
    volatile = 100 + np.cumsum(np.random.normal(0, 2.0, 100))
    trend = np.linspace(volatile[-1], volatile[-1] + 20, 30)
    flat = np.full(120, trend[-1]) + np.random.normal(0, 0.05, 120)
    close = np.concatenate([volatile, trend, flat])
    s = pd.Series(close)
    return pd.DataFrame({
        'open':   s.shift(1).fillna(s.iloc[0]),
        'high':   s * 1.003,
        'low':    s * 0.997,
        'close':  s,
        'volume': pd.Series(1_000_000, index=s.index),
    })


def _flat_ohlcv_history() -> pd.DataFrame:
    # Sine wave with no monotonic stack (mirrors Task 8's `_ohlcv_flat`)
    t = np.linspace(0, 8 * np.pi, 250)
    close = pd.Series(100.0 + 0.5 * np.sin(t))
    return pd.DataFrame({
        'open':   close.shift(1).fillna(close.iloc[0]),
        'high':   close * 1.003,
        'low':    close * 0.997,
        'close':  close,
        'volume': pd.Series(1_000_000, index=close.index),
    })


def test_run_scan_with_technical_filter_only_fires_aligned_strategies():
    bull_df = _bullish_ohlcv_history()
    chain = [
        _make_raw(option_type='put', strike=98, mid=1.0, iv=0.22),
        _make_raw(option_type='call', strike=102, mid=1.0, iv=0.22),
    ]
    src = _FakeSource(
        history=bull_df['close'], spot=100.0, chain=chain, ohlcv=bull_df,
    )

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=True),
            sources_override=[src],
        )

    # Bullish setup → LongCall (and NakedPut) eligible; LongPut / NakedCall excluded.
    strategy_names = {c.strategy.name for c in result.candidates}
    assert 'long_put' not in strategy_names
    assert 'naked_call' not in strategy_names
    if strategy_names:
        assert strategy_names.issubset({'long_call', 'naked_put'})
    _valid_bullish_setups = {'compression_breakout', 'pullback_in_trend', 'stage_2_breakout', 'failed_breakdown_reversal'}
    for c in result.candidates:
        assert c.setup_name in _valid_bullish_setups
        assert c.setup_direction == 'bullish'


def test_run_scan_with_no_signal_ticker_produces_no_candidates_when_filter_on():
    flat_df = _flat_ohlcv_history()
    chain = [_make_raw(option_type='put', strike=98, mid=1.0)]
    src = _FakeSource(history=flat_df['close'], spot=100.0, chain=chain, ohlcv=flat_df)

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=True),
            sources_override=[src],
        )

    assert result.candidates == []


def test_run_scan_with_as_of_uses_sliced_close_as_spot():
    """When as_of is set, the spot comes from the close of the as_of bar,
    not from fetch_spot's live quote."""
    bull_df = _bullish_ohlcv_history()
    bull_df.index = pd.date_range('2025-01-01', periods=len(bull_df), freq='B')
    last_date = bull_df.index[-1].date()
    expected_spot = float(bull_df['close'].iloc[-1])

    chain = [
        _make_raw(option_type='put', strike=expected_spot * 0.98, mid=1.0, iv=0.22),
        _make_raw(option_type='call', strike=expected_spot * 1.02, mid=1.0, iv=0.22),
    ]
    # Construct a source whose fetch_spot returns a DIFFERENT value, so we can
    # prove the as_of path overrode it.
    src = _FakeSource(
        history=bull_df['close'], spot=999.99,  # bogus live quote
        chain=chain, ohlcv=bull_df,
    )

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=True, as_of=last_date),
            sources_override=[src],
        )

    # Every candidate's contract.spot_price should be the as_of close, not 999.99
    for c in result.candidates:
        assert abs(c.contract.spot_price - expected_spot) < 0.5, (
            f"Expected spot ~{expected_spot}, got {c.contract.spot_price}"
        )


def test_run_scan_populates_market_context_on_candidates():
    """Candidates produced under the technical filter must carry market context."""
    bull_df = _bullish_ohlcv_history()
    bull_df.index = pd.date_range('2025-01-01', periods=len(bull_df), freq='B')
    chain = [
        _make_raw(option_type='put', strike=98, mid=1.0, iv=0.22),
        _make_raw(option_type='call', strike=102, mid=1.0, iv=0.22),
    ]
    src = _FakeSource(history=bull_df['close'], spot=100.0, chain=chain, ohlcv=bull_df)
    src._history = bull_df['close']
    # Make the close-only history use a DatetimeIndex too (matches new contract)
    src._history.index = bull_df.index

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=True),
            sources_override=[src],
        )

    if result.candidates:
        c = result.candidates[0]
        assert c.weekly_trend in ('up', 'down', 'flat')
        assert isinstance(c.consecutive_close_streak, int)
        assert isinstance(c.pct_change_1w, float)
        assert isinstance(c.pct_change_2w, float)
        assert isinstance(c.pct_change_4w, float)
        assert isinstance(c.weekly_ribbon_agreement, bool)


def test_run_scan_legacy_regime_gate_when_filter_off():
    """When use_technical_filter=False, ScanConfig falls back to regime-based gating.
    With a regime favoring 'sell', only NakedPut/NakedCall candidates should appear."""
    # Tiny-noise random walk gives RV ~0.014% annualised (vs IV 30%),
    # so rv_iv_ratio << 0.8 → regime.favored == ['sell'].
    # Requires >=3 ATM contracts (within ±5% of spot=100) so _atm_iv returns a
    # valid non-zero value — otherwise compute_regime falls back to 'sell'+'buy'.
    np.random.seed(0)
    history = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.001, 365)))
    chain = [
        _make_raw(option_type='put',  strike=99,  mid=1.0, iv=0.30),
        _make_raw(option_type='put',  strike=100, mid=1.0, iv=0.30),
        _make_raw(option_type='call', strike=100, mid=1.0, iv=0.30),
        _make_raw(option_type='call', strike=101, mid=1.0, iv=0.30),
    ]
    src = _FakeSource(history=history, spot=100.0, chain=chain)

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=False),
            sources_override=[src],
        )

    # rv_iv_ratio ~0.0005 < 0.8 → regime favors 'sell' only.
    # Therefore LongPut/LongCall ('buy' direction) should NOT appear.
    strategy_names = {c.strategy.name for c in result.candidates}
    assert 'long_put' not in strategy_names
    assert 'long_call' not in strategy_names
