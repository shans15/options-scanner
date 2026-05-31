from datetime import date
import pytest
import pandas as pd
from data.sources.base import DataSource, RawContract


def test_raw_contract_fields():
    rc = RawContract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.1, ask=2.15, volume=100, open_interest=500,
        implied_volatility=0.20, dte=21, spot_price=635.0,
    )
    assert rc.ticker == 'SPY'
    assert rc.option_type == 'put'


def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()


def test_concrete_source_must_implement_methods():
    class Incomplete(DataSource):
        pass
    with pytest.raises(TypeError):
        Incomplete()


def test_minimal_concrete_source_can_be_instantiated():
    class Minimal(DataSource):
        def fetch_spot(self, ticker): return 100.0
        def fetch_price_history(self, ticker, lookback_days): return pd.Series([1, 2, 3])
        def fetch_option_chain(self, ticker): return []
    src = Minimal()
    assert src.fetch_spot('X') == 100.0
