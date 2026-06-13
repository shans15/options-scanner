"""Strategy B component: GARCH(1,1) fair probability.

Fits GARCH(1,1) on last 30 days of hourly log returns, forecasts σ for N hours,
uses Student-t (df=4) for fat-tailed CDF per Christoffersen et al. (2016).

Theoretical grounding: Engle (1982) ARCH, Hansen & Lunde (2005) GARCH benchmarking.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# Minimum observations required for a stable GARCH fit
_MIN_OBS = 100


def estimate_fair_probability(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    hourly_returns: pd.Series,
) -> float:
    """Estimate P(BTC > strike at expiry) via GARCH(1,1) vol forecast.

    Parameters
    ----------
    spot : float
        Current BTC price.
    strike : float
        Price threshold.
    hours_to_expiry : float
        Hours until expiry.
    hourly_returns : pd.Series
        Log returns at hourly frequency over last ~30 days.

    Returns
    -------
    float
        Estimated probability, or NaN if fit fails (too little data,
        convergence issue).

    Notes
    -----
    1. Fit GARCH(1,1) using ``arch_model(rescaled returns, vol='Garch', p=1, q=1)``
    2. Forecast σ for next N hours; aggregate σ_T = sqrt(sum(forecast_variances))
    3. z = log(strike/spot) / σ_T
    4. P(BTC > strike) = 1 - student_t(df=4).cdf(z)
    """
    if len(hourly_returns) < _MIN_OBS:
        return float("nan")

    if hours_to_expiry <= 0:
        return 1.0 if spot > strike else 0.0

    try:
        from arch import arch_model  # deferred import — arch is heavy

        # Rescale returns to percentage (arch library convention)
        returns_pct = hourly_returns.dropna() * 100.0

        if len(returns_pct) < _MIN_OBS:
            return float("nan")

        n_hours = max(1, int(math.ceil(hours_to_expiry)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)

        # Forecast variance for next n_hours steps
        fc = res.forecast(horizon=n_hours, reindex=False)
        # fc.variance.values has shape (1, n_hours); values are in % units squared
        var_pct = fc.variance.values[-1, :]  # array of length n_hours

        # Aggregate variance and convert back from % to decimal
        total_var_pct = float(np.sum(var_pct))
        sigma_T = math.sqrt(total_var_pct) / 100.0  # back to decimal

        if sigma_T <= 0:
            return 1.0 if spot > strike else 0.0

        z = math.log(strike / spot) / sigma_T

        # Student-t with df=4 for fat tails
        return float(1.0 - student_t.cdf(z, df=4))

    except Exception:
        return float("nan")
