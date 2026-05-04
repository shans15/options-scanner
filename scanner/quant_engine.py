from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class StressResult:
    stress_1sd: float    # P&L if underlying moves 1 SD against position
    stress_2sd: float    # P&L if underlying moves 2 SD against position
    stress_expiry: float # P&L at expiry worst-case (5th percentile)


@dataclass
class PopResult:
    pop_delta: float
    pop_bs: float
    pop_historical: float
    pop_garch_mc: float
    pop_blended: float
    stress: StressResult
    expected_value: float
    breakeven: float
    margin_estimate: float
    iv_rank: float


def pop_delta(delta: float, option_type: str) -> float:
    """PoP from delta approximation: 1 - |delta|."""
    return float(np.clip(1.0 - abs(delta), 0.0, 1.0))


def pop_black_scholes(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str
) -> float:
    """Risk-neutral probability of expiring OTM (profitable for seller)."""
    if T <= 0 or sigma <= 0:
        return 0.5
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'put':
        return float(norm.cdf(d2))    # P(S_T > K)
    else:
        return float(norm.cdf(-d2))   # P(S_T < K)


def pop_historical(
    S: float, K: float, dte: int, log_returns: pd.Series
) -> float:
    """
    Historical PoP: fraction of rolling DTE-day log-return windows
    where stock stayed on profitable side of strike.
    """
    if len(log_returns) < dte + 10:
        return 0.5

    rolling = log_returns.rolling(dte).sum().dropna()
    required_return = np.log(K / S)

    if K < S:  # Put: profitable if cumulative return > required_return
        pop = (rolling > required_return).mean()
    else:       # Call: profitable if cumulative return < required_return
        pop = (rolling < required_return).mean()

    return float(np.clip(pop, 0.0, 1.0))


def pop_garch_mc(
    S: float,
    K: float,
    dte: int,
    log_returns: pd.Series,
    n_paths: int = 10000,
) -> float:
    """
    GARCH(1,1) + Monte Carlo PoP with Student-t innovations.
    Falls back to historical vol if GARCH fails.
    """
    try:
        from arch import arch_model
        pct_returns = log_returns * 100
        model = arch_model(pct_returns, vol='Garch', p=1, q=1, dist='t')
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=dte)
        avg_var = forecast.variance.values[-1].mean()
        daily_vol = np.sqrt(avg_var) / 100
        nu = float(result.params.get('nu', 8.0))
        nu = max(nu, 3.0)
    except Exception:
        daily_vol = float(log_returns.std())
        nu = 8.0

    np.random.seed(42)
    z = np.random.standard_t(df=nu, size=(n_paths, dte))
    z = z / np.sqrt(nu / (nu - 2))  # standardize to unit variance
    cumulative = (daily_vol * z).sum(axis=1)
    final_prices = S * np.exp(cumulative)

    if K < S:
        pop = (final_prices > K).mean()
    else:
        pop = (final_prices < K).mean()

    return float(np.clip(pop, 0.0, 1.0))


def blend_pop(
    pop_d: float, pop_bs: float, pop_hist: float, pop_garch: float
) -> float:
    """Weighted blend: delta 20%, BS 25%, historical 20%, GARCH-MC 35%."""
    blended = 0.20 * pop_d + 0.25 * pop_bs + 0.20 * pop_hist + 0.35 * pop_garch
    return float(np.clip(blended, 0.0, 1.0))


def compute_stress_scenarios(
    S: float, K: float, premium: float, sigma: float, option_type: str
) -> StressResult:
    """
    Unrealized P&L under 3 stress scenarios.
    Positive = still profitable, negative = losing.
    """
    daily_vol = sigma / np.sqrt(252)

    def intrinsic_loss(spot_at_scenario: float) -> float:
        if option_type == 'put':
            intrinsic = max(K - spot_at_scenario, 0)
        else:
            intrinsic = max(spot_at_scenario - K, 0)
        return round(premium - intrinsic, 4)

    if option_type == 'put':
        s1 = S * (1 - daily_vol)
        s2 = S * (1 - 2 * daily_vol)
    else:
        s1 = S * (1 + daily_vol)
        s2 = S * (1 + 2 * daily_vol)

    # 5th percentile expiry move
    worst_case_return = -1.645 * sigma * np.sqrt(30 / 252)
    if option_type == 'put':
        s_expiry = S * np.exp(worst_case_return)
    else:
        s_expiry = S * np.exp(-worst_case_return)

    return StressResult(
        stress_1sd=intrinsic_loss(s1),
        stress_2sd=intrinsic_loss(s2),
        stress_expiry=intrinsic_loss(s_expiry),
    )


def estimate_margin(S: float, K: float, premium: float, strategy: str) -> float:
    """Simplified Reg-T naked option margin estimate (per contract = x100)."""
    otm_amount = abs(S - K)
    if strategy == 'naked_put':
        margin = max(0.20 * S - otm_amount + premium, 0.10 * K + premium)
    else:
        margin = max(0.20 * S - otm_amount + premium, 0.10 * S + premium)
    return round(margin * 100, 2)


def compute_iv_rank_from_history(current_iv: float, log_returns: pd.Series) -> float:
    rv_30d = log_returns.rolling(30).std() * np.sqrt(252)
    low_rv = rv_30d.min()
    high_rv = rv_30d.max()
    if high_rv <= low_rv:
        return 50.0
    return float(np.clip((current_iv - low_rv) / (high_rv - low_rv) * 100, 0, 100))


def run_quant_engine(
    contract: dict,
    log_returns: pd.Series,
    risk_free_rate: float = 0.053,
    n_paths: int = 10000,
) -> PopResult:
    """Run all quantitative models for a single contract. Returns PopResult."""
    S = contract['spot_price']
    K = contract['strike']
    IV = contract['implied_volatility'] or float(log_returns.std() * np.sqrt(252))
    dte = contract['dte']
    delta = contract['delta']
    premium = contract['mid']
    strategy = contract['strategy']
    option_type = 'put' if strategy == 'naked_put' else 'call'
    T = dte / 252

    p_delta = pop_delta(delta, option_type)
    p_bs = pop_black_scholes(S, K, risk_free_rate, IV, T, option_type)
    p_hist = pop_historical(S, K, dte, log_returns)
    p_garch = pop_garch_mc(S, K, dte, log_returns, n_paths)
    p_blended = blend_pop(p_delta, p_bs, p_hist, p_garch)

    stress = compute_stress_scenarios(S, K, premium, IV, option_type)

    max_stress_loss = abs(min(stress.stress_2sd, 0))
    ev = (premium * p_blended) - (max_stress_loss * (1 - p_blended))

    if option_type == 'put':
        breakeven = K - premium
    else:
        breakeven = K + premium

    margin = estimate_margin(S, K, premium, strategy)
    iv_rank = compute_iv_rank_from_history(IV, log_returns)

    return PopResult(
        pop_delta=p_delta,
        pop_bs=p_bs,
        pop_historical=p_hist,
        pop_garch_mc=p_garch,
        pop_blended=p_blended,
        stress=stress,
        expected_value=round(ev, 4),
        breakeven=round(breakeven, 2),
        margin_estimate=margin,
        iv_rank=iv_rank,
    )
