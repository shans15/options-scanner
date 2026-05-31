from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from domain.contract import Contract
from domain.strategy import Strategy


def pop_delta(c: Contract, strategy: Strategy) -> float:
    if strategy.direction == 'sell':
        return float(np.clip(1.0 - abs(c.delta), 0.0, 1.0))
    return float(np.clip(abs(c.delta), 0.0, 1.0))


def pop_black_scholes(c: Contract, strategy: Strategy, r: float) -> float:
    S, sigma = c.spot_price, c.implied_volatility
    T = max(c.dte / 252, 1e-9)
    if sigma <= 0 or S <= 0:
        return 0.5
    if strategy.direction == 'sell':
        K = c.strike
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d2) if c.option_type == 'put' else norm.cdf(-d2))
    # Buyer needs to clear breakeven (K ± premium), not just strike.
    breakeven = strategy.breakeven(c)
    if breakeven <= 0:
        return 0.5
    d2 = (np.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(-d2) if c.option_type == 'put' else norm.cdf(d2))


def pop_historical(c: Contract, strategy: Strategy, log_returns: pd.Series) -> float:
    dte = c.dte
    series = log_returns.dropna()
    if len(series) < dte + 10:
        return 0.5
    rolling_sum = series.rolling(dte).sum().dropna()
    terminal_prices = c.spot_price * np.exp(rolling_sum)
    matches = terminal_prices.apply(lambda S_T: strategy.profit_condition(float(S_T), c))
    return float(np.clip(matches.mean(), 0.0, 1.0))


def pop_garch_mc(c: Contract, strategy: Strategy, log_returns: pd.Series, n_paths: int = 10000) -> float:
    series = log_returns.dropna()
    try:
        from arch import arch_model
        pct = series * 100
        model = arch_model(pct, vol='Garch', p=1, q=1, dist='t')
        res = model.fit(disp='off', show_warning=False)
        fc = res.forecast(horizon=c.dte)
        avg_var = fc.variance.values[-1].mean()
        daily_vol = float(np.sqrt(avg_var) / 100)
        nu = max(float(res.params.get('nu', 8.0)), 3.0)
    except Exception:
        daily_vol = float(series.std()) if len(series) > 1 else 0.01
        nu = 8.0
    np.random.seed(42)
    z = np.random.standard_t(df=nu, size=(n_paths, c.dte))
    z = z / np.sqrt(nu / (nu - 2))
    cumulative = (daily_vol * z).sum(axis=1)
    terminal_prices = c.spot_price * np.exp(cumulative)
    matches = np.array([strategy.profit_condition(float(p), c) for p in terminal_prices])
    return float(np.clip(matches.mean(), 0.0, 1.0))


def blend_pop(p_delta: float, p_bs: float, p_hist: float, p_garch: float) -> float:
    blended = 0.20 * p_delta + 0.25 * p_bs + 0.20 * p_hist + 0.35 * p_garch
    return float(np.clip(blended, 0.0, 1.0))
