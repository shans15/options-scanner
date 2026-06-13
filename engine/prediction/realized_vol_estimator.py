"""Strategy B component: realized-vol scaling.

Computes RV from last 60 minutes of 1-min returns, scales forward by √T.

Theoretical grounding: Andersen et al. (2003) — RV is efficient estimator.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_fair_probability(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    minute_returns_last_60: pd.Series,
) -> float:
    """Estimate fair prob from realized vol scaling.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    hours_to_expiry : float
        Hours until expiry.
    minute_returns_last_60 : pd.Series
        Series of 1-minute log returns over the last 60 minutes.

    Returns
    -------
    float
        Estimated probability that BTC > strike at expiry.

    Notes
    -----
    1. RV = sum of squared 1-min log returns over last 60 min
    2. σ_per_hour = √RV  (already an hourly estimate)
    3. σ_T = σ_per_hour × √hours_to_expiry
    4. z = log(strike/spot) / σ_T
    5. P = 1 - student_t(df=4).cdf(z)
    """
    if hours_to_expiry <= 0:
        return 1.0 if spot > strike else 0.0

    returns_arr = np.asarray(minute_returns_last_60.dropna(), dtype=float)

    if len(returns_arr) == 0:
        return 0.5  # no data — no signal

    # Realized variance = sum of squared returns (no mean subtraction — short horizon)
    rv = float(np.sum(returns_arr ** 2))
    sigma_per_hour = math.sqrt(rv) if rv > 0 else 0.0

    if sigma_per_hour <= 0:
        # No movement at all — price will likely stay near spot
        return 1.0 if spot > strike else 0.0

    sigma_T = sigma_per_hour * math.sqrt(hours_to_expiry)

    if sigma_T <= 0:
        return 1.0 if spot > strike else 0.0

    z = math.log(strike / spot) / sigma_T

    # Student-t with df=4 for fat tails
    return float(1.0 - student_t.cdf(z, df=4))
