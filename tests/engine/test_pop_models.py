from datetime import date
import numpy as np
import pandas as pd
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from engine.pop_models import pop_delta, pop_black_scholes, pop_historical, pop_garch_mc, blend_pop


def _make_contract(option_type='put', delta=-0.20, strike=100.0, mid=2.0, spot=100.0,
                   iv=0.25, dte=21):
    actual_delta = -abs(delta) if option_type == 'put' else abs(delta)
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=mid-0.05, ask=mid+0.05, mid=mid,
        volume=1000, open_interest=1000, implied_volatility=iv,
        delta=actual_delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=dte, spot_price=spot,
    )


def test_pop_delta_seller_is_one_minus_abs_delta():
    c = _make_contract(option_type='put', delta=0.20)
    assert pop_delta(c, NakedPut()) == pytest.approx(0.80, abs=0.01)


def test_pop_delta_buyer_is_abs_delta():
    c = _make_contract(option_type='put', delta=0.50)
    assert pop_delta(c, LongPut()) == pytest.approx(0.50, abs=0.01)


def test_pop_bs_naked_put_high_when_otm():
    c = _make_contract(option_type='put', strike=90.0, spot=100.0, mid=0.5, iv=0.20, dte=21)
    pop = pop_black_scholes(c, NakedPut(), r=0.05)
    assert pop > 0.80


def test_pop_bs_long_call_low_when_otm():
    c = _make_contract(option_type='call', strike=110.0, spot=100.0, mid=0.5, iv=0.20, dte=21)
    pop = pop_black_scholes(c, LongCall(), r=0.05)
    assert pop < 0.30


def test_pop_historical_returns_in_range():
    np.random.seed(0)
    log_returns = pd.Series(np.random.normal(0.0, 0.01, 500))
    c = _make_contract(option_type='put', strike=98.0, spot=100.0, mid=1.0)
    p = pop_historical(c, NakedPut(), log_returns)
    assert 0.0 <= p <= 1.0


def test_pop_historical_with_too_few_returns_returns_half():
    log_returns = pd.Series([0.01, -0.01, 0.005])
    c = _make_contract(option_type='put')
    assert pop_historical(c, NakedPut(), log_returns) == 0.5


def test_pop_garch_mc_returns_in_range():
    np.random.seed(0)
    log_returns = pd.Series(np.random.normal(0.0, 0.01, 300))
    c = _make_contract(option_type='put', strike=95.0, spot=100.0, mid=1.0, dte=21)
    p = pop_garch_mc(c, NakedPut(), log_returns, n_paths=1000)
    assert 0.0 <= p <= 1.0


def test_blend_pop_weights_match_spec():
    blended = blend_pop(p_delta=1.0, p_bs=0.0, p_hist=0.0, p_garch=0.0)
    assert blended == pytest.approx(0.20)
    blended = blend_pop(p_delta=0.0, p_bs=1.0, p_hist=0.0, p_garch=0.0)
    assert blended == pytest.approx(0.25)
    blended = blend_pop(p_delta=0.0, p_bs=0.0, p_hist=1.0, p_garch=0.0)
    assert blended == pytest.approx(0.20)
    blended = blend_pop(p_delta=0.0, p_bs=0.0, p_hist=0.0, p_garch=1.0)
    assert blended == pytest.approx(0.35)


def test_blend_pop_clipped_to_unit_interval():
    assert blend_pop(2.0, 2.0, 2.0, 2.0) == 1.0
    assert blend_pop(-1.0, -1.0, -1.0, -1.0) == 0.0
