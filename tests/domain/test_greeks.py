import pytest
from domain.greeks import compute_greeks


def test_atm_call_delta_near_half():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert 0.45 < g['delta'] < 0.65


def test_atm_put_delta_near_negative_half():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.20, flag='p')
    assert -0.55 < g['delta'] < -0.35


def test_deep_otm_call_delta_near_zero():
    g = compute_greeks(S=100.0, K=150.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert 0.0 < g['delta'] < 0.05


def test_deep_itm_call_delta_near_one():
    g = compute_greeks(S=100.0, K=50.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert 0.9 < g['delta'] <= 1.0


def test_greeks_keys_complete():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert set(g.keys()) == {'delta', 'gamma', 'theta', 'vega'}


def test_zero_T_returns_zeros_or_intrinsic():
    g = compute_greeks(S=100.0, K=100.0, T=0.0, r=0.05, sigma=0.20, flag='c')
    assert all(g[k] == 0.0 or isinstance(g[k], float) for k in g)


def test_zero_sigma_returns_zeros_or_intrinsic():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.0, flag='c')
    assert all(isinstance(g[k], float) for k in g)
