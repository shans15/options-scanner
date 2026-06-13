"""Research-paper-derived adjustments applied to raw fair probabilities.

Each function returns (adjusted_prob, adjustment_label_str).

References:
  - Snowberg & Wolfers (2010): Explaining the Favorite-Long Shot Bias
  - Donaldson & Kim (1993): Support Levels and Resistance Levels
  - Lo & MacKinlay (1988): Stock Market Prices Do Not Follow Random Walks
  - Andersen et al. (2001): The Distribution of Realized Exchange Rate Volatility
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from scipy.stats import norm


def apply_favorite_longshot_bias(
    fair_prob: float,
) -> tuple[float, Optional[str]]:
    """Snowberg & Wolfers (2010): long-shots overpriced, favorites underpriced.

    For fair_prob < 0.15: shift toward 0.5 by 0.03 * (0.5 - fair_prob) * 2
    For fair_prob > 0.85: shift toward 0.5 similarly
    Otherwise: no adjustment

    Returns
    -------
    tuple[float, Optional[str]]
        (adjusted_prob, label string like 'favorite-longshot:-2%' or None)
    """
    if fair_prob < 0.15:
        # Long-shot: market overprices it, our fair_prob is actually lower — but
        # we adjust our estimate *toward* 0.5 to account for the systematic bias
        # (the bias means low-prob events are overpriced in markets, so our model
        # may underestimate because it hasn't captured this anchoring)
        shift = 0.03 * (0.5 - fair_prob) * 2
        adjusted = min(fair_prob + shift, 0.5)
        pct = round((adjusted - fair_prob) * 100)
        label = f"favorite-longshot:{pct:+d}%"
        return adjusted, label

    elif fair_prob > 0.85:
        # Favorite: market underprices it, shift toward 0.5
        shift = 0.03 * (fair_prob - 0.5) * 2
        adjusted = max(fair_prob - shift, 0.5)
        pct = round((adjusted - fair_prob) * 100)
        label = f"favorite-longshot:{pct:+d}%"
        return adjusted, label

    return fair_prob, None


def apply_round_number_magnetism(
    fair_prob: float,
    strike: float,
    spot: float,
    is_test_strike: bool = True,
) -> tuple[float, Optional[str]]:
    """Donaldson & Kim (1993): prices pin to round numbers.

    If strike % 1000 == 0 (e.g., $63,000, $64,000), and we are within 3% of strike:
      bump P(test strike) by 0.025
    Otherwise no adjustment.

    Parameters
    ----------
    fair_prob : float
        Raw fair probability.
    strike : float
        The price level being tested.
    spot : float
        Current BTC price.
    is_test_strike : bool
        True if market resolves on whether BTC *tests* the strike;
        False if pure above/below.

    Returns
    -------
    tuple[float, Optional[str]]
        (adjusted_prob, label or None)
    """
    # Check if strike is a round thousand
    if strike % 1000 != 0:
        return fair_prob, None

    # Check if we are within 3% of the round number
    distance_pct = abs(spot - strike) / strike
    if distance_pct > 0.03:
        return fair_prob, None

    if not is_test_strike:
        return fair_prob, None

    adjusted = min(fair_prob + 0.025, 1.0)
    return adjusted, "round-number-magnetism:+2.5%"


def apply_mean_reversion(
    fair_prob: float,
    last_hour_return: float,
    hourly_sigma: float,
) -> tuple[float, Optional[str]]:
    """Lo & MacKinlay (1988): negative serial correlation at sub-hourly horizons.

    If |last_hour_return| > hourly_sigma:
      dampen the "continuation" probability by 0.3 * (return / sigma)
      i.e., if last bar was up >1σ, P(BTC > strike) decreases (mean-reversion)

    Parameters
    ----------
    fair_prob : float
        Raw fair probability.
    last_hour_return : float
        Log return of the last hour.
    hourly_sigma : float
        Estimated hourly standard deviation.

    Returns
    -------
    tuple[float, Optional[str]]
        (adjusted_prob, label or None)
    """
    if hourly_sigma <= 0:
        return fair_prob, None

    z_last = last_hour_return / hourly_sigma

    if abs(z_last) <= 1.0:
        return fair_prob, None

    # Dampen continuation: large up move → reduce P(above strike)
    # large down move → increase P(above strike) (mean reversion)
    dampen = 0.3 * (last_hour_return / hourly_sigma)
    adjusted = fair_prob - dampen
    adjusted = max(0.0, min(1.0, adjusted))

    pct = round((adjusted - fair_prob) * 100)
    label = f"mean-reversion:{pct:+d}%"
    return adjusted, label


def apply_time_of_day_vol(
    fair_prob: float,
    expiration_utc: datetime,
    spot: float,
    strike: float,
    sigma_T: float,
) -> tuple[float, Optional[str]]:
    """Andersen et al. (2001): vol seasonality across sessions.

    Look at the UTC hour of expiration:
      22:00-02:00 UTC: Asia open, multiplier 1.15
      06:00-09:00 UTC: EU open, multiplier 1.10
      13:30-15:00 UTC: US open, multiplier 1.20
      19:30-20:00 UTC: US close, multiplier 1.15
      other: 1.0

    Apply multiplier to sigma_T, recompute fair_prob via lognormal/normal CDF
    using the inflated sigma.

    Parameters
    ----------
    fair_prob : float
        Raw fair probability.
    expiration_utc : datetime
        Expiration datetime (UTC).
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    sigma_T : float
        The base total vol (σ × √T) used in the raw fair_prob calculation.

    Returns
    -------
    tuple[float, Optional[str]]
        (adjusted_prob, label or None)
    """
    hour = expiration_utc.hour
    minute = expiration_utc.minute
    time_frac = hour + minute / 60.0  # fractional hour in [0, 24)

    # Determine session multiplier
    multiplier = 1.0
    session_label = None

    if time_frac >= 22.0 or time_frac < 2.0:
        multiplier = 1.15
        session_label = "asia-open"
    elif 6.0 <= time_frac < 9.0:
        multiplier = 1.10
        session_label = "eu-open"
    elif 13.5 <= time_frac < 15.0:
        multiplier = 1.20
        session_label = "us-open"
    elif 19.5 <= time_frac < 20.0:
        multiplier = 1.15
        session_label = "us-close"

    if multiplier == 1.0 or sigma_T <= 0:
        return fair_prob, None

    # Recompute probability with inflated sigma
    sigma_inflated = sigma_T * multiplier

    log_moneyness = math.log(strike / spot) if strike > 0 and spot > 0 else 0.0
    z_new = (log_moneyness - (-sigma_inflated ** 2 / 2)) / sigma_inflated
    adjusted = float(1.0 - norm.cdf(z_new))

    pct = round((adjusted - fair_prob) * 100)
    label = f"time-of-day-{session_label}:{pct:+d}%"
    return adjusted, label
