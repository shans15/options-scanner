from __future__ import annotations
from typing import Literal
from py_vollib.black_scholes.greeks import analytical as bs_greeks


def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    flag: Literal['c', 'p'],
) -> dict[str, float]:
    """Black-Scholes Greeks via py_vollib. Returns zeros on degenerate inputs."""
    if T <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    try:
        return {
            'delta': float(bs_greeks.delta(flag, S, K, T, r, sigma)),
            'gamma': float(bs_greeks.gamma(flag, S, K, T, r, sigma)),
            'theta': float(bs_greeks.theta(flag, S, K, T, r, sigma)),
            'vega':  float(bs_greeks.vega(flag, S, K, T, r, sigma)),
        }
    except Exception:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
