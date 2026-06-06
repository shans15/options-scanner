import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from data.sources.base import DataSource
from data.sources.yahooquery_source import YahooQuerySource


class _StubSource(DataSource):
    def fetch_spot(self, ticker): return 0.0
    def fetch_price_history(self, ticker, lookback_days): return pd.Series(dtype=float)
    def fetch_option_chain(self, ticker): return []
    # Does not override fetch_price_history_ohlcv → should raise NotImplementedError


def test_datasource_requires_fetch_price_history_ohlcv():
    # Sources missing OHLCV impl should fail to instantiate (abstract)
    with pytest.raises(TypeError):
        _StubSource()


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


def test_yahooquery_ohlcv_respects_lookback_days():
    fake_df = pd.DataFrame({
        'open': list(range(10)),
        'high': list(range(10)),
        'low': list(range(10)),
        'close': list(range(10)),
        'volume': [1_000_000] * 10,
    }, index=pd.to_datetime([f'2026-01-{d:02d}' for d in range(2, 12)]))
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        instance = MagicMock()
        instance.history.return_value = fake_df
        MockTicker.return_value = instance
        out = YahooQuerySource().fetch_price_history_ohlcv('AAPL', 3)
    assert len(out) == 3
    assert out['close'].iloc[-1] == 9


def test_stooq_ohlcv_preserves_datetime_index():
    fake_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-01-02,100,102,99,101,1000000\n"
        "2026-01-03,101,103,100,102,1100000\n"
    )
    from io import StringIO
    from data.sources.stooq_source import StooqSource
    with patch('data.sources.stooq_source.pd.read_csv', return_value=pd.read_csv(StringIO(fake_csv))):
        out = StooqSource().fetch_price_history_ohlcv('AAPL', 365)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out['close'].iloc[-1] == 102
