import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from data.fallback import DataFetchError
from data.sources.base import DataSource
from pipeline.technical_filter import filter_by_technicals


def _ohlcv_uptrend_with_squeeze(seed: int = 99) -> pd.DataFrame:
    np.random.seed(seed)
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


def _ohlcv_flat() -> pd.DataFrame:
    # Sine wave oscillates without directional trend — no EMA stacking, no setups.
    t = np.linspace(0, 8 * np.pi, 250)
    close = pd.Series(100.0 + 2.0 * np.sin(t))
    return pd.DataFrame({
        'open':   close.shift(1).fillna(close.iloc[0]),
        'high':   close * 1.003,
        'low':    close * 0.997,
        'close':  close,
        'volume': pd.Series(1_000_000, index=close.index),
    })


class _FakeSource(DataSource):
    def __init__(self, histories: dict[str, pd.DataFrame]):
        self._histories = histories
    def fetch_spot(self, ticker): return 100.0
    def fetch_price_history(self, ticker, lookback_days): return self._histories[ticker]['close']
    def fetch_price_history_ohlcv(self, ticker, lookback_days):
        if ticker not in self._histories:
            raise RuntimeError(f"unknown ticker {ticker}")
        return self._histories[ticker]
    def fetch_option_chain(self, ticker): return []


def test_filter_drops_tickers_without_setups():
    src = _FakeSource({
        'AAA': _ohlcv_uptrend_with_squeeze(),    # bullish compression
        'BBB': _ohlcv_flat(),                    # no setup
    })
    out = filter_by_technicals(['AAA', 'BBB'], [src])
    assert 'AAA' in out
    assert 'BBB' not in out
    assert len(out['AAA']) >= 1


def test_filter_handles_fetch_failure_gracefully():
    src = _FakeSource({'AAA': _ohlcv_uptrend_with_squeeze()})
    # CCC will raise because it's not in the dict
    out = filter_by_technicals(['AAA', 'CCC'], [src])
    assert 'AAA' in out
    assert 'CCC' not in out


def _ohlcv_downtrend_with_squeeze(seed: int = 99) -> pd.DataFrame:
    # Volatile phase → 30-bar downtrend dropping 30 pts → 80-bar flat squeeze.
    # At the end the EMA ribbon is stacked bearish (e8 < e13 < e21 < e48) and
    # PO bandwidth is in the bottom 20th percentile, reliably triggering at least
    # compression_breakout + pullback_in_trend with direction='bearish'.
    np.random.seed(seed)
    volatile = 100 + np.cumsum(np.random.normal(0, 2.0, 100))
    trend = np.linspace(volatile[-1], volatile[-1] - 30, 80)
    flat = np.full(80, trend[-1]) + np.random.normal(0, 0.05, 80)
    close = np.concatenate([volatile, trend, flat])
    s = pd.Series(close)
    return pd.DataFrame({
        'open':   s.shift(1).fillna(s.iloc[0]),
        'high':   s * 1.003,
        'low':    s * 0.997,
        'close':  s,
        'volume': pd.Series(1_000_000, index=s.index),
    })


def test_filter_returns_three_tickers_with_correct_directions():
    src = _FakeSource({
        'BULL': _ohlcv_uptrend_with_squeeze(),
        'BEAR': _ohlcv_downtrend_with_squeeze(),
        'FLAT': _ohlcv_flat(),
    })
    out = filter_by_technicals(['BULL', 'BEAR', 'FLAT'], [src])
    assert set(out.keys()) == {'BULL', 'BEAR'}
    assert any(s.direction == 'bullish' for s in out['BULL'])
    assert any(s.direction == 'bearish' for s in out['BEAR'])
