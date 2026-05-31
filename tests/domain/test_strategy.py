from datetime import date
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall


def _make_contract(option_type='put', delta=-0.20, strike=100.0, mid=2.0, spot=100.0):
    actual_delta = -abs(delta) if option_type == 'put' else abs(delta)
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=mid-0.05, ask=mid+0.05, mid=mid,
        volume=1000, open_interest=1000, implied_volatility=0.25,
        delta=actual_delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=spot,
    )


class TestNakedPut:
    def test_name_and_direction(self):
        s = NakedPut()
        assert s.name == 'naked_put'
        assert s.direction == 'sell'
        assert s.option_type == 'put'

    def test_applies_to_put_in_delta_range(self):
        c = _make_contract(option_type='put', delta=0.20)
        assert NakedPut().applies_to(c) is True

    def test_rejects_call_contract(self):
        c = _make_contract(option_type='call', delta=0.20)
        assert NakedPut().applies_to(c) is False

    def test_rejects_below_delta_floor(self):
        c = _make_contract(option_type='put', delta=0.10)
        assert NakedPut().applies_to(c) is False

    def test_rejects_above_delta_ceiling(self):
        c = _make_contract(option_type='put', delta=0.35)
        assert NakedPut().applies_to(c) is False

    def test_breakeven_is_strike_minus_mid(self):
        c = _make_contract(option_type='put', strike=100.0, mid=2.0)
        assert NakedPut().breakeven(c) == 98.0

    def test_profit_condition_above_strike(self):
        c = _make_contract(option_type='put', strike=100.0, mid=2.0)
        assert NakedPut().profit_condition(101.0, c) is True
        assert NakedPut().profit_condition(99.0, c) is False


class TestNakedCall:
    def test_breakeven_is_strike_plus_mid(self):
        c = _make_contract(option_type='call', strike=100.0, mid=2.0)
        assert NakedCall().breakeven(c) == 102.0

    def test_profit_condition_below_strike(self):
        c = _make_contract(option_type='call', strike=100.0, mid=2.0)
        assert NakedCall().profit_condition(99.0, c) is True
        assert NakedCall().profit_condition(101.0, c) is False


class TestLongPut:
    def test_delta_range_is_buyer_zone(self):
        c40 = _make_contract(option_type='put', delta=0.40)
        c60 = _make_contract(option_type='put', delta=0.60)
        c30 = _make_contract(option_type='put', delta=0.30)
        assert LongPut().applies_to(c40) is True
        assert LongPut().applies_to(c60) is True
        assert LongPut().applies_to(c30) is False

    def test_profit_condition_below_strike_minus_mid(self):
        c = _make_contract(option_type='put', strike=100.0, mid=2.0)
        assert LongPut().profit_condition(97.0, c) is True
        assert LongPut().profit_condition(99.0, c) is False


class TestLongCall:
    def test_profit_condition_above_strike_plus_mid(self):
        c = _make_contract(option_type='call', strike=100.0, mid=2.0)
        assert LongCall().profit_condition(103.0, c) is True
        assert LongCall().profit_condition(101.0, c) is False
