"""Strategy A: vol-model-based fair probability.

Uses the trained AUC-0.745 vol model artifact (cache/models/btc_vol_model.pkl)
to estimate forward σ, then computes P(BTC > X at T) via lognormal CDF.

Theoretical grounding: Black-Scholes (1973) — lognormal price distribution.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def estimate_fair_probability(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    vol_proba: float,
    historical_vol_quantiles: dict[float, float],
) -> float:
    """Estimate P(BTC > strike at expiry) using vol-model regime + lognormal CDF.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold the market resolves on.
    hours_to_expiry : float
        Hours until market expiration.
    vol_proba : float
        Output of vol model — P(vol expansion next 4 bars).
    historical_vol_quantiles : dict[float, float]
        Precomputed dict {0.25: σ_low, 0.5: σ_med, 0.75: σ_high}
        derived from last 365 days of hourly realized vol.

    Returns
    -------
    float
        Estimated probability that BTC > strike at expiry.

    Notes
    -----
    Regime selection:
      if vol_proba > 0.65: σ = historical_vol_quantiles[0.75]  # high-vol regime
      elif vol_proba < 0.35: σ = historical_vol_quantiles[0.25]  # low-vol
      else: σ = historical_vol_quantiles[0.5]  # neutral

    σ_T = σ × √hours_to_expiry  (scale to expiry)
    z = (log(strike/spot) - (-σ_T²/2)) / σ_T  (risk-neutral drift = 0 for short-horizon)
    P(BTC > strike) = 1 - Φ(z)
    """
    if hours_to_expiry <= 0:
        # Already expired — binary result based on current price
        return 1.0 if spot > strike else 0.0

    # Select vol regime
    if vol_proba > 0.65:
        sigma_hourly = historical_vol_quantiles[0.75]
    elif vol_proba < 0.35:
        sigma_hourly = historical_vol_quantiles[0.25]
    else:
        sigma_hourly = historical_vol_quantiles[0.5]

    # Scale σ to expiry horizon
    sigma_T = sigma_hourly * math.sqrt(hours_to_expiry)

    if sigma_T <= 0:
        return 1.0 if spot > strike else 0.0

    # Lognormal z-score with risk-neutral drift adjustment
    # Under risk-neutral measure: d = log(F/K) / σ_T where F = spot * exp(-σ²/2 * T)
    # Simplified for short-horizon (no interest rate): drift correction = -σ_T²/2
    log_moneyness = math.log(strike / spot)
    z = (log_moneyness - (-sigma_T ** 2 / 2)) / sigma_T

    # P(BTC > strike) = 1 - Φ(z)
    return float(1.0 - norm.cdf(z))
