"""Weighted ensemble of all 4 estimators + 3-strategy harness for backtesting.

Strategy A:   vol-model only
Strategy B:   (GARCH + bootstrap + RV-scaling) / 3
Strategy A+B: (vol-model + GARCH + bootstrap + RV-scaling) / 4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from engine.prediction import (
    vol_model_estimator,
    garch_estimator,
    historical_bootstrap,
    realized_vol_estimator,
    adjustments,
)


@dataclass
class StrategyOutput:
    """Result of a single strategy computation."""

    fair_prob_raw: float
    fair_prob_adjusted: float
    adjustments_applied: list[str] = field(default_factory=list)
    constituent_estimates: dict[str, float] = field(default_factory=dict)


def _apply_all_adjustments(
    fair_prob: float,
    strike: float,
    spot: float,
    expiration_utc: datetime,
    last_hour_return: float,
    hourly_sigma: float,
    sigma_T: float,
) -> tuple[float, list[str]]:
    """Apply all 4 research-paper adjustments sequentially.

    Returns
    -------
    tuple[float, list[str]]
        (adjusted probability, list of non-None adjustment labels)
    """
    applied: list[str] = []

    # 1. Favorite-longshot bias
    fair_prob, label = adjustments.apply_favorite_longshot_bias(fair_prob)
    if label:
        applied.append(label)

    # 2. Round-number magnetism
    fair_prob, label = adjustments.apply_round_number_magnetism(
        fair_prob, strike=strike, spot=spot, is_test_strike=True
    )
    if label:
        applied.append(label)

    # 3. Mean reversion
    fair_prob, label = adjustments.apply_mean_reversion(
        fair_prob, last_hour_return=last_hour_return, hourly_sigma=hourly_sigma
    )
    if label:
        applied.append(label)

    # 4. Time-of-day vol
    fair_prob, label = adjustments.apply_time_of_day_vol(
        fair_prob,
        expiration_utc=expiration_utc,
        spot=spot,
        strike=strike,
        sigma_T=sigma_T,
    )
    if label:
        applied.append(label)

    return fair_prob, applied


def _derive_sigma_T(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    hourly_sigma: float,
) -> float:
    """Approximate σ_T = hourly_sigma × √hours_to_expiry for adjustment purposes."""
    if hours_to_expiry <= 0:
        return 0.0
    return hourly_sigma * math.sqrt(hours_to_expiry)


def compute_strategy_A(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    vol_proba: float,
    historical_vol_quantiles: dict[float, float],
    expiration_utc: datetime,
    last_hour_return: float,
    hourly_sigma: float,
) -> StrategyOutput:
    """Strategy A: vol-model estimator + 4 adjustments.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    hours_to_expiry : float
        Hours until expiry.
    vol_proba : float
        P(vol expansion) from the vol model.
    historical_vol_quantiles : dict[float, float]
        {0.25: σ_low, 0.5: σ_med, 0.75: σ_high}
    expiration_utc : datetime
        Expiry in UTC (for time-of-day adjustment).
    last_hour_return : float
        Log return of last hour (for mean-reversion adjustment).
    hourly_sigma : float
        Estimated hourly σ (for mean-reversion and σ_T derivation).
    """
    raw = vol_model_estimator.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        vol_proba=vol_proba,
        historical_vol_quantiles=historical_vol_quantiles,
    )

    sigma_T = _derive_sigma_T(spot, strike, hours_to_expiry, hourly_sigma)
    adjusted, applied = _apply_all_adjustments(
        fair_prob=raw,
        strike=strike,
        spot=spot,
        expiration_utc=expiration_utc,
        last_hour_return=last_hour_return,
        hourly_sigma=hourly_sigma,
        sigma_T=sigma_T,
    )

    return StrategyOutput(
        fair_prob_raw=raw,
        fair_prob_adjusted=adjusted,
        adjustments_applied=applied,
        constituent_estimates={"vol_model": raw},
    )


def compute_strategy_B(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    hourly_returns: pd.Series,
    minute_returns_last_60: pd.Series,
    expiration_utc: datetime,
    last_hour_return: float,
    hourly_sigma: float,
) -> StrategyOutput:
    """Strategy B: (GARCH + bootstrap + RV-scaling) average + 4 adjustments.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    hours_to_expiry : float
        Hours until expiry.
    hourly_returns : pd.Series
        Historical hourly log returns (last ~30+ days).
    minute_returns_last_60 : pd.Series
        Last 60 minutes of 1-min log returns.
    expiration_utc : datetime
        Expiry in UTC.
    last_hour_return : float
        Log return of last hour.
    hourly_sigma : float
        Estimated hourly σ.
    """
    garch_p = garch_estimator.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        hourly_returns=hourly_returns,
    )

    bootstrap_p = historical_bootstrap.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=int(math.ceil(hours_to_expiry)),
        hourly_returns=hourly_returns,
    )

    rv_p = realized_vol_estimator.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        minute_returns_last_60=minute_returns_last_60,
    )

    constituents = {
        "garch": garch_p,
        "bootstrap": bootstrap_p,
        "rv_scaling": rv_p,
    }

    # Average only the non-NaN estimates
    valid = [v for v in [garch_p, bootstrap_p, rv_p] if not (isinstance(v, float) and math.isnan(v))]
    raw = float(np.mean(valid)) if valid else 0.5

    sigma_T = _derive_sigma_T(spot, strike, hours_to_expiry, hourly_sigma)
    adjusted, applied = _apply_all_adjustments(
        fair_prob=raw,
        strike=strike,
        spot=spot,
        expiration_utc=expiration_utc,
        last_hour_return=last_hour_return,
        hourly_sigma=hourly_sigma,
        sigma_T=sigma_T,
    )

    return StrategyOutput(
        fair_prob_raw=raw,
        fair_prob_adjusted=adjusted,
        adjustments_applied=applied,
        constituent_estimates=constituents,
    )


def compute_strategy_AB(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    vol_proba: float,
    historical_vol_quantiles: dict[float, float],
    hourly_returns: pd.Series,
    minute_returns_last_60: pd.Series,
    expiration_utc: datetime,
    last_hour_return: float,
    hourly_sigma: float,
) -> StrategyOutput:
    """Strategy A+B: all 4 estimators averaged + 4 adjustments.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    hours_to_expiry : float
        Hours until expiry.
    vol_proba : float
        P(vol expansion) from the vol model.
    historical_vol_quantiles : dict[float, float]
        {0.25: σ_low, 0.5: σ_med, 0.75: σ_high}
    hourly_returns : pd.Series
        Historical hourly log returns.
    minute_returns_last_60 : pd.Series
        Last 60 minutes of 1-min log returns.
    expiration_utc : datetime
        Expiry in UTC.
    last_hour_return : float
        Log return of last hour.
    hourly_sigma : float
        Estimated hourly σ.
    """
    vol_model_p = vol_model_estimator.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        vol_proba=vol_proba,
        historical_vol_quantiles=historical_vol_quantiles,
    )

    garch_p = garch_estimator.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        hourly_returns=hourly_returns,
    )

    bootstrap_p = historical_bootstrap.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=int(math.ceil(hours_to_expiry)),
        hourly_returns=hourly_returns,
    )

    rv_p = realized_vol_estimator.estimate_fair_probability(
        spot=spot,
        strike=strike,
        hours_to_expiry=hours_to_expiry,
        minute_returns_last_60=minute_returns_last_60,
    )

    constituents = {
        "vol_model": vol_model_p,
        "garch": garch_p,
        "bootstrap": bootstrap_p,
        "rv_scaling": rv_p,
    }

    # Average all 4, skipping NaN
    valid = [v for v in [vol_model_p, garch_p, bootstrap_p, rv_p]
             if not (isinstance(v, float) and math.isnan(v))]
    raw = float(np.mean(valid)) if valid else 0.5

    sigma_T = _derive_sigma_T(spot, strike, hours_to_expiry, hourly_sigma)
    adjusted, applied = _apply_all_adjustments(
        fair_prob=raw,
        strike=strike,
        spot=spot,
        expiration_utc=expiration_utc,
        last_hour_return=last_hour_return,
        hourly_sigma=hourly_sigma,
        sigma_T=sigma_T,
    )

    return StrategyOutput(
        fair_prob_raw=raw,
        fair_prob_adjusted=adjusted,
        adjustments_applied=applied,
        constituent_estimates=constituents,
    )


def count_estimator_agreement(
    constituent_estimates: dict[str, float],
    market_prob: float,
) -> tuple[int, int]:
    """Count how many estimators agree on direction vs market_prob.

    An estimator "agrees" that there is edge if it is on the same side of
    market_prob (all above or all below).  Specifically, agreement is defined
    as: fair > market_prob → estimator says "BUY YES" (we think it's cheap).

    This function counts how many estimators sit on the *majority* side.

    Parameters
    ----------
    constituent_estimates : dict[str, float]
        Map of estimator name → fair probability estimate.
    market_prob : float
        The market-implied probability (mid of bid/ask).

    Returns
    -------
    tuple[int, int]
        (n_agree, n_total) — n_agree of n_total estimators agree on direction.
    """
    valid = {k: v for k, v in constituent_estimates.items()
             if not (isinstance(v, float) and math.isnan(v))}
    if not valid:
        return (0, 0)

    n_total = len(valid)

    # Determine the majority direction
    n_above = sum(1 for v in valid.values() if v > market_prob)
    n_below = n_total - n_above

    # n_agree = how many are on the majority side
    n_agree = max(n_above, n_below)
    return (n_agree, n_total)
