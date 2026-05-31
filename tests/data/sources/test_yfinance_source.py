from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from data.sources.yfinance_source import YfinanceSource


def test_fetch_spot_uses_fast_info_last_price():
    with patch('data.sources.yfinance_source.yf.Ticker') as MockTicker:
        instance = MockTicker.return_value
        instance.fast_info = MagicMock(last_price=635.0)
        assert YfinanceSource().fetch_spot('SPY') == 635.0


def test_fetch_price_history_returns_close_series():
    df = pd.DataFrame({'Close': [630.0, 631.0, 632.0]}, index=pd.date_range('2026-05-27', periods=3))
    with patch('data.sources.yfinance_source.yf.Ticker') as MockTicker:
        MockTicker.return_value.history.return_value = df
        out = YfinanceSource().fetch_price_history('SPY', 30)
        assert out.tolist() == [630.0, 631.0, 632.0]


def test_fetch_option_chain_iterates_expirations():
    today = date.today()
    exp1 = (pd.Timestamp(today) + pd.Timedelta(days=21)).date().isoformat()
    exp2 = (pd.Timestamp(today) + pd.Timedelta(days=2)).date().isoformat()
    puts_df = pd.DataFrame({
        'strike':[620.0], 'bid':[2.0], 'ask':[2.1], 'volume':[100],
        'openInterest':[1000], 'impliedVolatility':[0.20],
    })
    calls_df = pd.DataFrame({
        'strike':[630.0], 'bid':[1.5], 'ask':[1.6], 'volume':[200],
        'openInterest':[800], 'impliedVolatility':[0.18],
    })
    chain_tuple = MagicMock(); chain_tuple.puts = puts_df; chain_tuple.calls = calls_df

    with patch('data.sources.yfinance_source.yf.Ticker') as MockTicker:
        inst = MockTicker.return_value
        inst.options = [exp1, exp2]
        inst.option_chain.side_effect = lambda exp: chain_tuple if exp == exp1 else MagicMock(puts=pd.DataFrame(), calls=pd.DataFrame())
        inst.fast_info = MagicMock(last_price=635.0)

        contracts = YfinanceSource().fetch_option_chain('SPY')
        types = {rc.option_type for rc in contracts}
        assert types == {'put', 'call'}
        assert all(3 <= rc.dte <= 45 for rc in contracts)
