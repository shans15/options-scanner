from datetime import date
import pytest
from data.adapters import to_contract
from data.sources.base import RawContract


def _make_raw(option_type='put', strike=100.0, dte_days=21):
    today = date(2026, 5, 30)
    exp = date(2026, 6, 20)   # ~21 days later
    return RawContract(
        ticker='X', expiration=exp, strike=strike,
        option_type=option_type, bid=2.0, ask=2.10,
        volume=100, open_interest=500, implied_volatility=0.20,
        dte=(exp - today).days, spot_price=100.0,
    )


def test_to_contract_computes_mid_and_greeks():
    raw = _make_raw()
    c = to_contract(raw, spot=100.0, r=0.05, today=date(2026, 5, 30))
    assert c is not None
    assert c.mid == pytest.approx(2.05, abs=1e-6)
    assert c.delta != 0


def test_to_contract_drops_when_zero_bid_ask():
    raw = RawContract('X', date(2026, 6, 20), 100.0, 'put', 0.0, 0.0, 100, 500, 0.2, 21, 100.0)
    assert to_contract(raw, 100.0, 0.05, date(2026, 5, 30)) is None


def test_to_contract_drops_when_dte_below_3():
    today = date(2026, 6, 17)
    exp = date(2026, 6, 19)
    raw = RawContract('X', exp, 100.0, 'put', 2.0, 2.1, 100, 500, 0.2, 2, 100.0)
    assert to_contract(raw, 100.0, 0.05, today) is None


def test_to_contract_drops_when_dte_above_45():
    today = date(2026, 5, 30)
    exp = date(2026, 8, 1)
    raw = RawContract('X', exp, 100.0, 'put', 2.0, 2.1, 100, 500, 0.2, 63, 100.0)
    assert to_contract(raw, 100.0, 0.05, today) is None


def test_to_contract_put_delta_negative():
    raw = _make_raw(option_type='put', strike=100.0)
    c = to_contract(raw, spot=100.0, r=0.05, today=date(2026, 5, 30))
    assert c.delta < 0


def test_to_contract_call_delta_positive():
    raw = _make_raw(option_type='call', strike=100.0)
    c = to_contract(raw, spot=100.0, r=0.05, today=date(2026, 5, 30))
    assert c.delta > 0
