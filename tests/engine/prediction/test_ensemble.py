"""Tests for engine/prediction/ensemble.py."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.prediction.ensemble import (
    StrategyOutput,
    compute_strategy_A,
    compute_strategy_AB,
    compute_strategy_B,
    count_estimator_agreement,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_BTC_SPOT = 63_000.0
_BTC_STRIKE = 64_000.0
_HOURS = 4.0
_VOL_QUANTILES = {0.25: 0.01, 0.5: 0.015, 0.75: 0.025}
_EXPIRY_UTC = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
_LAST_HOUR_RETURN = 0.002
_HOURLY_SIGMA = 0.015


def _make_hourly_returns(n: int = 720) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 0.005, n))


def _make_minute_returns(n: int = 60) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 0.001, n))


def _base_A_kwargs() -> dict:
    return dict(
        spot=_BTC_SPOT,
        strike=_BTC_STRIKE,
        hours_to_expiry=_HOURS,
        vol_proba=0.5,
        historical_vol_quantiles=_VOL_QUANTILES,
        expiration_utc=_EXPIRY_UTC,
        last_hour_return=_LAST_HOUR_RETURN,
        hourly_sigma=_HOURLY_SIGMA,
    )


def _base_B_kwargs() -> dict:
    return dict(
        spot=_BTC_SPOT,
        strike=_BTC_STRIKE,
        hours_to_expiry=_HOURS,
        hourly_returns=_make_hourly_returns(),
        minute_returns_last_60=_make_minute_returns(),
        expiration_utc=_EXPIRY_UTC,
        last_hour_return=_LAST_HOUR_RETURN,
        hourly_sigma=_HOURLY_SIGMA,
    )


def _base_AB_kwargs() -> dict:
    return dict(
        spot=_BTC_SPOT,
        strike=_BTC_STRIKE,
        hours_to_expiry=_HOURS,
        vol_proba=0.5,
        historical_vol_quantiles=_VOL_QUANTILES,
        hourly_returns=_make_hourly_returns(),
        minute_returns_last_60=_make_minute_returns(),
        expiration_utc=_EXPIRY_UTC,
        last_hour_return=_LAST_HOUR_RETURN,
        hourly_sigma=_HOURLY_SIGMA,
    )


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------

class TestStrategyA:
    def test_strategy_A_uses_only_vol_model(self):
        """Strategy A must call vol_model_estimator and NOT call the B estimators."""
        with (
            patch("engine.prediction.ensemble.vol_model_estimator.estimate_fair_probability",
                  return_value=0.45) as mock_vm,
            patch("engine.prediction.ensemble.garch_estimator.estimate_fair_probability") as mock_garch,
            patch("engine.prediction.ensemble.historical_bootstrap.estimate_fair_probability") as mock_bs,
            patch("engine.prediction.ensemble.realized_vol_estimator.estimate_fair_probability") as mock_rv,
        ):
            result = compute_strategy_A(**_base_A_kwargs())

        mock_vm.assert_called_once()
        mock_garch.assert_not_called()
        mock_bs.assert_not_called()
        mock_rv.assert_not_called()

    def test_strategy_A_returns_strategy_output(self):
        result = compute_strategy_A(**_base_A_kwargs())
        assert isinstance(result, StrategyOutput)
        assert 0.0 <= result.fair_prob_raw <= 1.0
        assert 0.0 <= result.fair_prob_adjusted <= 1.0

    def test_strategy_A_stores_vol_model_in_constituents(self):
        result = compute_strategy_A(**_base_A_kwargs())
        assert "vol_model" in result.constituent_estimates

    def test_strategy_A_adjustments_list_is_list(self):
        result = compute_strategy_A(**_base_A_kwargs())
        assert isinstance(result.adjustments_applied, list)


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------

class TestStrategyB:
    def test_strategy_B_uses_three_estimators_only(self):
        """Strategy B must call GARCH + bootstrap + RV; NOT vol_model."""
        with (
            patch("engine.prediction.ensemble.vol_model_estimator.estimate_fair_probability") as mock_vm,
            patch("engine.prediction.ensemble.garch_estimator.estimate_fair_probability",
                  return_value=0.42) as mock_garch,
            patch("engine.prediction.ensemble.historical_bootstrap.estimate_fair_probability",
                  return_value=0.43) as mock_bs,
            patch("engine.prediction.ensemble.realized_vol_estimator.estimate_fair_probability",
                  return_value=0.44) as mock_rv,
        ):
            result = compute_strategy_B(**_base_B_kwargs())

        mock_vm.assert_not_called()
        mock_garch.assert_called_once()
        mock_bs.assert_called_once()
        mock_rv.assert_called_once()

    def test_strategy_B_averages_three_estimates(self):
        """With mocked estimates 0.42, 0.43, 0.44 → raw ≈ 0.43."""
        with (
            patch("engine.prediction.ensemble.garch_estimator.estimate_fair_probability",
                  return_value=0.42),
            patch("engine.prediction.ensemble.historical_bootstrap.estimate_fair_probability",
                  return_value=0.43),
            patch("engine.prediction.ensemble.realized_vol_estimator.estimate_fair_probability",
                  return_value=0.44),
        ):
            result = compute_strategy_B(**_base_B_kwargs())

        assert result.fair_prob_raw == pytest.approx(0.43, abs=0.001)

    def test_strategy_B_returns_strategy_output(self):
        result = compute_strategy_B(**_base_B_kwargs())
        assert isinstance(result, StrategyOutput)

    def test_strategy_B_stores_three_constituents(self):
        result = compute_strategy_B(**_base_B_kwargs())
        assert "garch" in result.constituent_estimates
        assert "bootstrap" in result.constituent_estimates
        assert "rv_scaling" in result.constituent_estimates
        assert "vol_model" not in result.constituent_estimates

    def test_strategy_B_skips_nan_in_average(self):
        """If GARCH returns NaN (short series), B should still average the other two."""
        hourly_returns_short = _make_hourly_returns(n=50)  # too short for GARCH
        kwargs = _base_B_kwargs()
        kwargs["hourly_returns"] = hourly_returns_short

        result = compute_strategy_B(**kwargs)
        # bootstrap and rv should still run
        assert not math.isnan(result.fair_prob_raw)
        assert 0.0 <= result.fair_prob_raw <= 1.0


# ---------------------------------------------------------------------------
# Strategy A+B
# ---------------------------------------------------------------------------

class TestStrategyAB:
    def test_strategy_AB_uses_all_four(self):
        """Strategy A+B must call all 4 estimators."""
        with (
            patch("engine.prediction.ensemble.vol_model_estimator.estimate_fair_probability",
                  return_value=0.45) as mock_vm,
            patch("engine.prediction.ensemble.garch_estimator.estimate_fair_probability",
                  return_value=0.42) as mock_garch,
            patch("engine.prediction.ensemble.historical_bootstrap.estimate_fair_probability",
                  return_value=0.43) as mock_bs,
            patch("engine.prediction.ensemble.realized_vol_estimator.estimate_fair_probability",
                  return_value=0.44) as mock_rv,
        ):
            result = compute_strategy_AB(**_base_AB_kwargs())

        mock_vm.assert_called_once()
        mock_garch.assert_called_once()
        mock_bs.assert_called_once()
        mock_rv.assert_called_once()

    def test_strategy_AB_stores_four_constituents(self):
        with (
            patch("engine.prediction.ensemble.vol_model_estimator.estimate_fair_probability",
                  return_value=0.45),
            patch("engine.prediction.ensemble.garch_estimator.estimate_fair_probability",
                  return_value=0.42),
            patch("engine.prediction.ensemble.historical_bootstrap.estimate_fair_probability",
                  return_value=0.43),
            patch("engine.prediction.ensemble.realized_vol_estimator.estimate_fair_probability",
                  return_value=0.44),
        ):
            result = compute_strategy_AB(**_base_AB_kwargs())

        assert "vol_model" in result.constituent_estimates
        assert "garch" in result.constituent_estimates
        assert "bootstrap" in result.constituent_estimates
        assert "rv_scaling" in result.constituent_estimates

    def test_constituent_estimates_stored_for_transparency(self):
        """All 4 constituent estimates must be stored in the output."""
        result = compute_strategy_AB(**_base_AB_kwargs())
        assert len(result.constituent_estimates) == 4

    def test_strategy_AB_raw_is_mean_of_four(self):
        """With mocked values 0.42, 0.43, 0.44, 0.45 → raw ≈ 0.435."""
        with (
            patch("engine.prediction.ensemble.vol_model_estimator.estimate_fair_probability",
                  return_value=0.45),
            patch("engine.prediction.ensemble.garch_estimator.estimate_fair_probability",
                  return_value=0.42),
            patch("engine.prediction.ensemble.historical_bootstrap.estimate_fair_probability",
                  return_value=0.43),
            patch("engine.prediction.ensemble.realized_vol_estimator.estimate_fair_probability",
                  return_value=0.44),
        ):
            result = compute_strategy_AB(**_base_AB_kwargs())

        assert result.fair_prob_raw == pytest.approx(0.435, abs=0.001)

    def test_strategy_AB_returns_strategy_output(self):
        result = compute_strategy_AB(**_base_AB_kwargs())
        assert isinstance(result, StrategyOutput)
        assert 0.0 <= result.fair_prob_adjusted <= 1.0


# ---------------------------------------------------------------------------
# count_estimator_agreement
# ---------------------------------------------------------------------------

class TestCountEstimatorAgreement:
    def test_all_agree_above_market(self):
        """4/4 estimators above market_prob → (4, 4)."""
        estimates = {"vm": 0.60, "garch": 0.62, "bs": 0.58, "rv": 0.61}
        n_agree, n_total = count_estimator_agreement(estimates, market_prob=0.50)
        assert n_total == 4
        assert n_agree == 4

    def test_all_agree_below_market(self):
        """4/4 estimators below market_prob → (4, 4)."""
        estimates = {"vm": 0.40, "garch": 0.38, "bs": 0.42, "rv": 0.39}
        n_agree, n_total = count_estimator_agreement(estimates, market_prob=0.50)
        assert n_total == 4
        assert n_agree == 4

    def test_split_agreement(self):
        """2 above, 2 below → majority is 2, returns (2, 4)."""
        estimates = {"vm": 0.60, "garch": 0.40, "bs": 0.60, "rv": 0.40}
        n_agree, n_total = count_estimator_agreement(estimates, market_prob=0.50)
        assert n_total == 4
        assert n_agree == 2

    def test_empty_estimates(self):
        n_agree, n_total = count_estimator_agreement({}, market_prob=0.50)
        assert n_agree == 0
        assert n_total == 0

    def test_nan_estimates_excluded(self):
        """NaN estimates should be excluded from the count."""
        estimates = {"vm": 0.60, "garch": float("nan"), "bs": 0.62, "rv": 0.58}
        n_agree, n_total = count_estimator_agreement(estimates, market_prob=0.50)
        assert n_total == 3  # NaN excluded
        assert n_agree == 3

    def test_three_agree_one_disagrees(self):
        estimates = {"vm": 0.60, "garch": 0.65, "bs": 0.55, "rv": 0.40}
        n_agree, n_total = count_estimator_agreement(estimates, market_prob=0.50)
        assert n_total == 4
        assert n_agree == 3  # majority = 3 above
