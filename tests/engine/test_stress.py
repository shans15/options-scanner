from datetime import date
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from engine.stress import compute_stress, StressResult


def _make_contract(option_type='put', strike=100.0, mid=2.0, spot=100.0, iv=0.25):
    delta = -0.20 if option_type == 'put' else 0.20
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=mid-0.05, ask=mid+0.05, mid=mid,
        volume=1000, open_interest=1000, implied_volatility=iv,
        delta=delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=spot,
    )


def test_naked_put_stress_2sd_negative_when_strike_breached():
    c = _make_contract(option_type='put', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c, NakedPut(), c.implied_volatility)
    assert s.stress_2sd < 0
    assert s.stress_2sd < s.stress_1sd


def test_naked_call_stress_uses_upside_move():
    c_call = _make_contract(option_type='call', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c_call, NakedCall(), c_call.implied_volatility)
    assert s.stress_2sd < 0


def test_long_put_stress_floored_at_negative_mid():
    c = _make_contract(option_type='put', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c, LongPut(), c.implied_volatility)
    assert s.stress_2sd >= -c.mid - 1e-6


def test_long_call_stress_floored_at_negative_mid():
    c = _make_contract(option_type='call', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c, LongCall(), c.implied_volatility)
    assert s.stress_2sd >= -c.mid - 1e-6


def test_stress_result_dataclass_fields():
    c = _make_contract()
    s = compute_stress(c, NakedPut(), c.implied_volatility)
    assert isinstance(s, StressResult)
    assert isinstance(s.stress_1sd, float)
    assert isinstance(s.stress_2sd, float)
    assert isinstance(s.stress_expiry, float)
