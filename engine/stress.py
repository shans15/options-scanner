from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from domain.contract import Contract
from domain.strategy import Strategy


@dataclass
class StressResult:
    stress_1sd: float
    stress_2sd: float
    stress_expiry: float


def _adverse_spot(spot: float, daily_sigma: float, sigmas: float, strategy: Strategy) -> float:
    if strategy.name in ('naked_put', 'long_call'):
        return spot * (1 - daily_sigma * sigmas)
    return spot * (1 + daily_sigma * sigmas)


def _intrinsic_at(spot_at: float, c: Contract) -> float:
    if c.option_type == 'put':
        return max(c.strike - spot_at, 0.0)
    return max(spot_at - c.strike, 0.0)


def _pl_at(spot_at: float, c: Contract, strategy: Strategy) -> float:
    intrinsic = _intrinsic_at(spot_at, c)
    if strategy.direction == 'sell':
        return round(c.mid - intrinsic, 4)
    pl = intrinsic - c.mid
    return round(max(pl, -c.mid), 4)


def compute_stress(c: Contract, strategy: Strategy, sigma_annual: float) -> StressResult:
    daily_sigma = max(sigma_annual, 1e-9) / np.sqrt(252)
    s1 = _adverse_spot(c.spot_price, daily_sigma, 1.0, strategy)
    s2 = _adverse_spot(c.spot_price, daily_sigma, 2.0, strategy)

    horizon_sigma = sigma_annual * np.sqrt(30 / 252)
    if strategy.name in ('naked_put', 'long_call'):
        s_exp = c.spot_price * np.exp(-1.645 * horizon_sigma)
    else:
        s_exp = c.spot_price * np.exp(1.645 * horizon_sigma)

    return StressResult(
        stress_1sd=_pl_at(s1, c, strategy),
        stress_2sd=_pl_at(s2, c, strategy),
        stress_expiry=_pl_at(s_exp, c, strategy),
    )
