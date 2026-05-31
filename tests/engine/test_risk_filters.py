from datetime import date
import math
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from engine.stress import StressResult
from engine.risk_filters import FilterSet, FILTER_SETS, apply_filters, FilterResult


def _make(option_type='put', delta=-0.20, strike=100.0, mid=2.0, spot=100.0,
          bid=1.95, ask=2.05, volume=1000, oi=1000):
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=bid, ask=ask, mid=mid,
        volume=volume, open_interest=oi, implied_volatility=0.25,
        delta=delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=spot,
    )


def test_filter_set_for_naked_put_matches_spec():
    fs = FILTER_SETS['naked_put']
    assert fs.delta_range == (0.16, 0.30)
    assert fs.min_pop == 0.70
    assert fs.max_stress_loss_multiple == 3.0


def test_filter_set_for_long_call_matches_spec():
    fs = FILTER_SETS['long_call']
    assert fs.delta_range == (0.40, 0.60)
    assert fs.min_pop == 0.40
    assert fs.min_ev == 0.15
    assert math.isinf(fs.max_stress_loss_multiple)


def test_apply_filters_all_pass_for_clean_seller():
    c = _make(option_type='put', delta=-0.22, mid=2.0, bid=1.95, ask=2.05)
    stress = StressResult(stress_1sd=-1.0, stress_2sd=-3.0, stress_expiry=-2.0)
    result = apply_filters(c, NakedPut(), pop_blended=0.75, ev=0.5, stress=stress)
    assert result.passed is True
    assert result.failed_filters == []


def test_apply_filters_rejects_wide_spread():
    c = _make(bid=1.0, ask=3.0, mid=2.0)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert r.passed is False
    assert 'spread' in r.failed_filters


def test_apply_filters_rejects_low_volume():
    c = _make(volume=10)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'volume' in r.failed_filters


def test_apply_filters_rejects_low_oi():
    c = _make(oi=100)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'open_interest' in r.failed_filters


def test_apply_filters_rejects_delta_out_of_range():
    c = _make(delta=-0.40)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'delta' in r.failed_filters


def test_apply_filters_rejects_low_pop():
    c = _make(delta=-0.22)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.50, 0.5, stress)
    assert 'pop' in r.failed_filters


def test_seller_stress_loss_multiple_3x_rejects_high_stress():
    c = _make(delta=-0.22, mid=2.0)
    stress = StressResult(-2, -10, -5)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'stress' in r.failed_filters


def test_buyer_stress_filter_not_applied():
    c = _make(option_type='call', delta=0.50, mid=3.0)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, LongCall(), 0.45, 0.20, stress)
    assert r.passed is True
