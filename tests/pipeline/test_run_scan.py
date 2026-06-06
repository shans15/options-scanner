from datetime import date, datetime
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from data.sources.base import DataSource, RawContract
from pipeline.run_scan import run_scan, ScanConfig, ScanResult


class _FakeSource(DataSource):
    def __init__(self, history: pd.Series, spot: float, chain: list[RawContract]):
        self._history = history; self._spot = spot; self._chain = chain
    def fetch_spot(self, ticker): return self._spot
    def fetch_price_history(self, ticker, lookback_days): return self._history
    def fetch_option_chain(self, ticker): return self._chain
    def fetch_price_history_ohlcv(self, ticker, lookback_days): return pd.DataFrame()


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
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[src])

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
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[_Broken()])

    assert 'X' in result.skipped
    assert 'fetch_failed' in result.skipped['X']
