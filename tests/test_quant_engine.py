from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scanner.quant_engine import (
    pop_delta, pop_black_scholes, pop_historical,
    pop_garch_mc, compute_stress_scenarios, blend_pop,
    StressResult, PopResult, run_quant_engine,
)

# Realistic OTM put: SPY at 530, strike 510, 14 DTE, IV 18%
S, K, IV, DTE, R = 530.0, 510.0, 0.18, 14, 0.053


def _make_returns(n: int = 252, vol: float = 0.01) -> pd.Series:
    np.random.seed(42)
    return pd.Series(np.random.normal(0, vol, n))


def test_pop_delta_put():
    result = pop_delta(delta=-0.15, option_type='put')
    assert 0.7 < result < 0.95


def test_pop_delta_call():
    result = pop_delta(delta=0.15, option_type='call')
    assert 0.7 < result < 0.95


def test_pop_black_scholes_put():
    result = pop_black_scholes(S=S, K=K, r=R, sigma=IV, T=DTE/252, option_type='put')
    assert 0.5 < result < 1.0


def test_pop_black_scholes_call():
    result = pop_black_scholes(S=530, K=550, r=R, sigma=IV, T=DTE/252, option_type='call')
    assert 0.5 < result < 1.0


def test_pop_historical_returns_float():
    returns = _make_returns()
    result = pop_historical(S=S, K=K, dte=DTE, log_returns=returns)
    assert 0.0 <= result <= 1.0


def test_pop_garch_mc_returns_float():
    returns = _make_returns()
    result = pop_garch_mc(S=S, K=K, dte=DTE, log_returns=returns, n_paths=500)
    assert 0.0 <= result <= 1.0


def test_blend_pop_in_range():
    result = blend_pop(pop_d=0.85, pop_bs=0.83, pop_hist=0.80, pop_garch=0.82)
    assert 0.80 <= result <= 0.86


def test_stress_scenarios_put():
    result = compute_stress_scenarios(
        S=S, K=K, premium=1.50, sigma=IV, option_type='put'
    )
    assert isinstance(result, StressResult)
    assert result.stress_1sd <= 1.50   # at most premium received
    assert result.stress_2sd <= result.stress_1sd  # worse scenario is worse


def test_run_quant_engine_returns_pop_result():
    returns = _make_returns()
    contract = {
        'spot_price': S, 'strike': K, 'implied_volatility': IV,
        'dte': DTE, 'delta': -0.15, 'mid': 1.50,
        'strategy': 'naked_put',
    }
    result = run_quant_engine(contract, returns, risk_free_rate=R, n_paths=200)
    assert isinstance(result, PopResult)
    assert 0.0 <= result.pop_blended <= 1.0
    assert result.margin_estimate > 0
    assert result.breakeven < K  # put breakeven is below strike
