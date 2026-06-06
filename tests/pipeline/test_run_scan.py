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
    history = pd.Series(np.exp(np.cumsum(np.random.RandomState(0).normal(0, 0.01, 365)))) * 100
    chain = [_make_raw(option_type='put', strike=98, mid=1.0),
             _make_raw(option_type='call', strike=102, mid=1.0)]
    src = _FakeSource(history, spot=100.0, chain=chain)

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[src])

    assert isinstance(result, ScanResult)
    assert result.skipped == {} or 'X' not in result.skipped
    assert isinstance(result.candidates, list)


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
    _valid_bullish_setups = {'compression_breakout', 'pullback_in_trend', 'stage2_breakout', 'failed_breakdown_reversal'}
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
