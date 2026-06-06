import pandas as pd
import pytest

from data.sources.base import DataSource


class _StubSource(DataSource):
    def fetch_spot(self, ticker): return 0.0
    def fetch_price_history(self, ticker, lookback_days): return pd.Series(dtype=float)
    def fetch_option_chain(self, ticker): return []
    # Does not override fetch_price_history_ohlcv → should raise NotImplementedError


def test_datasource_requires_fetch_price_history_ohlcv():
    # Sources missing OHLCV impl should fail to instantiate (abstract)
    with pytest.raises(TypeError):
        _StubSource()


from unittest.mock import patch, MagicMock
import numpy as np
from data.sources.yahooquery_source import YahooQuerySource


def test_yahooquery_ohlcv_returns_dataframe_with_required_columns():
    fake_df = pd.DataFrame({
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [99.0, 100.0],
        'close': [101.0, 102.0],
        'volume': [1_000_000, 1_100_000],
    }, index=pd.to_datetime(['2026-01-02', '2026-01-03']))

    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        instance = MagicMock()
        instance.history.return_value = fake_df
        MockTicker.return_value = instance
        out = YahooQuerySource().fetch_price_history_ohlcv('AAPL', 365)

    assert set(out.columns) >= {'open', 'high', 'low', 'close', 'volume'}
    assert len(out) == 2
    assert out['close'].iloc[-1] == 102.0
