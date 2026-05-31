from datetime import date
import pytest
from domain.contract import Contract


def test_contract_is_frozen():
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.10, ask=2.15, mid=2.125,
        volume=1500, open_interest=8000, implied_volatility=0.18,
        delta=-0.20, gamma=0.012, theta=-0.05, vega=0.15,
        dte=21, spot_price=635.0,
    )
    with pytest.raises(Exception):
        c.strike = 999.0  # frozen dataclass forbids mutation


def test_contract_mid_independent_of_constructor():
    # mid is stored, not derived — callers compute it
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.0, ask=3.0, mid=2.5,
        volume=1, open_interest=1, implied_volatility=0.2,
        delta=-0.2, gamma=0.0, theta=0.0, vega=0.0,
        dte=10, spot_price=620.0,
    )
    assert c.mid == 2.5


def test_contract_option_type_constrained():
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='call', bid=1.0, ask=1.1, mid=1.05,
        volume=1, open_interest=1, implied_volatility=0.2,
        delta=0.3, gamma=0.0, theta=0.0, vega=0.0,
        dte=10, spot_price=620.0,
    )
    assert c.option_type == 'call'
