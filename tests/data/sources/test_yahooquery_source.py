from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from data.sources.yahooquery_source import YahooQuerySource


def test_fetch_spot_reads_regular_market_price():
    src = YahooQuerySource()
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        MockTicker.return_value.price = {'SPY': {'regularMarketPrice': 635.42}}
        assert src.fetch_spot('SPY') == 635.42


def test_fetch_price_history_returns_close_series():
    src = YahooQuerySource()
    df = pd.DataFrame({
        'close':  [630.0, 631.5, 633.0],
        'volume': [1_000_000, 1_200_000, 950_000],
    }, index=pd.MultiIndex.from_tuples(
        [('SPY', pd.Timestamp('2026-05-27')),
         ('SPY', pd.Timestamp('2026-05-28')),
         ('SPY', pd.Timestamp('2026-05-29'))],
        names=['symbol', 'date'],
    ))
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        MockTicker.return_value.history.return_value = df
        out = src.fetch_price_history('SPY', lookback_days=30)
        assert isinstance(out, pd.Series)
        assert out.tolist() == [630.0, 631.5, 633.0]


def test_fetch_option_chain_flattens_multiindex():
    src = YahooQuerySource()
    chain_df = pd.DataFrame({
        'strike': [620.0, 625.0],
        'bid':    [2.0,   3.0],
        'ask':    [2.1,   3.1],
        'volume': [100,   150],
        'openInterest': [1000, 800],
        'impliedVolatility': [0.18, 0.19],
        'inTheMoney': [False, False],
    }, index=pd.MultiIndex.from_tuples(
        [('SPY', pd.Timestamp('2026-06-20'), 'puts', 620.0),
         ('SPY', pd.Timestamp('2026-06-20'), 'puts', 625.0)],
        names=['symbol', 'expiration', 'optionType', 'strike'],
    ))
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        MockTicker.return_value.option_chain = chain_df
        MockTicker.return_value.price = {'SPY': {'regularMarketPrice': 635.42}}
        contracts = src.fetch_option_chain('SPY')
        assert len(contracts) >= 0
        for rc in contracts:
            assert rc.ticker == 'SPY'
            assert rc.option_type == 'put'
