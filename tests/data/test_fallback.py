import pytest
import pandas as pd
from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.base import DataSource


class _Fail(DataSource):
    def __init__(self, msg='boom'): self.msg = msg
    def fetch_spot(self, ticker): raise RuntimeError(self.msg)
    def fetch_price_history(self, ticker, lookback_days): raise RuntimeError(self.msg)
    def fetch_option_chain(self, ticker): raise RuntimeError(self.msg)


class _OK(DataSource):
    def __init__(self, value=42.0): self.value = value
    def fetch_spot(self, ticker): return self.value
    def fetch_price_history(self, ticker, lookback_days): return pd.Series([1.0, 2.0])
    def fetch_option_chain(self, ticker): return ['raw']


def test_first_success_wins():
    out = fetch_with_fallback([_OK(value=99.0), _Fail()], 'fetch_spot', 'X')
    assert out == 99.0


def test_falls_back_to_next_on_failure():
    out = fetch_with_fallback([_Fail(), _OK(value=11.0)], 'fetch_spot', 'X')
    assert out == 11.0


def test_raises_when_all_sources_fail():
    with pytest.raises(DataFetchError):
        fetch_with_fallback([_Fail('a'), _Fail('b')], 'fetch_spot', 'X')


def test_passes_through_args_and_kwargs():
    captured = {}
    class _Capture(DataSource):
        def fetch_spot(self, ticker): captured['ticker'] = ticker; return 1.0
        def fetch_price_history(self, ticker, lookback_days):
            captured.update({'ticker': ticker, 'lb': lookback_days}); return pd.Series([1])
        def fetch_option_chain(self, ticker): return []
    fetch_with_fallback([_Capture()], 'fetch_price_history', 'SPY', 365)
    assert captured == {'ticker': 'SPY', 'lb': 365}
