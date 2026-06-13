"""Strategy B component: empirical bootstrap from past returns.

Resamples N consecutive hourly returns from last 90 days, 5,000 trials.
Computes empirical P(BTC > strike at expiry).

Theoretical grounding: Efron (1979) bootstrap, Cont (2001) fat tails.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 90 days * 24 hours
_MAX_LOOKBACK_HOURS = 90 * 24


def estimate_fair_probability(
    spot: float,
    strike: float,
    hours_to_expiry: int,
    hourly_returns: pd.Series,
    n_trials: int = 5000,
    rng_seed: int = 42,
) -> float:
    """Bootstrap empirical probability that BTC ends above strike at expiry.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    hours_to_expiry : int
        Number of hours until expiry.
    hourly_returns : pd.Series
        Historical log returns at hourly frequency.
    n_trials : int
        Number of bootstrap trials (default 5,000).
    rng_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    float
        Fraction of trials where terminal price > strike.

    Notes
    -----
    1. Use last 90 days of hourly_returns (clip).
    2. For n_trials:
       - Sample n_hours consecutive returns with replacement
       - Compute cumulative return → terminal price = spot * exp(cum_return)
       - Count if terminal > strike
    3. Return fraction.
    """
    if hours_to_expiry <= 0:
        return 1.0 if spot > strike else 0.0

    # Clip to last 90 days
    returns_arr = np.asarray(hourly_returns.dropna(), dtype=float)
    returns_arr = returns_arr[-_MAX_LOOKBACK_HOURS:]

    n_hours = max(1, int(hours_to_expiry))

    if len(returns_arr) == 0:
        return 0.5  # no data — no signal

    rng = np.random.default_rng(rng_seed)

    # Draw n_trials samples, each of length n_hours, with replacement
    # Shape: (n_trials, n_hours)
    indices = rng.integers(0, len(returns_arr), size=(n_trials, n_hours))
    sampled = returns_arr[indices]

    # Cumulative log return for each trial
    cum_returns = sampled.sum(axis=1)  # shape (n_trials,)

    # Terminal prices
    terminal_prices = spot * np.exp(cum_returns)

    # Fraction above strike
    return float(np.mean(terminal_prices > strike))
