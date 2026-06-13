"""Tests for domain/prediction/fair_probability.py — the detect_edge() public API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from domain.prediction.fair_probability import detect_edge
from domain.prediction.market import KalshiMarket, PredictionMarketEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOL_QUANTILES = {0.25: 0.01, 0.5: 0.015, 0.75: 0.025}
_BTC_SPOT = 63_000.0


def _make_hourly_returns(n: int = 720) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 0.005, n))


def _make_minute_returns(n: int = 60) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 0.001, n))


def _make_market(
    yes_bid: float = 0.45,
    yes_ask: float = 0.47,
    strike: float = 64_000.0,
    side: str = "above",
    hours_until_expiry: float = 4.0,
) -> KalshiMarket:
    """Build a minimal KalshiMarket for testing."""
    expiry = datetime.now(tz=timezone.utc) + timedelta(hours=hours_until_expiry)
    return KalshiMarket(
        ticker="KXBTC-26JUN1316-T64000",
        event_ticker="KXBTC-26JUN1316",
        title="BTC > $64,000",
        strike=strike,
        side=side,
        expiration=expiry,
        close_time=expiry,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        volume_24h=1000,
        open_interest=500,
        status="open",
    )


def _base_kwargs(market: KalshiMarket) -> dict:
    return dict(
        market=market,
        btc_spot=_BTC_SPOT,
        btc_minute_returns_last_60=_make_minute_returns(),
        btc_hourly_returns_last_30d=_make_hourly_returns(),
        vol_proba=0.5,
        historical_vol_quantiles=_VOL_QUANTILES,
        last_hour_return=0.002,
        hourly_sigma=0.015,
        suggested_position_dollars=80.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectEdge:
    def test_returns_none_when_below_min_edge(self):
        """When fair_prob ≈ market_implied_prob, edge < 5% → None."""
        # Market at 0.46 mid; we mock canonical fair_prob to 0.47 → edge = 0.01 < 0.05
        market = _make_market(yes_bid=0.45, yes_ask=0.47)  # mid = 0.46

        with patch("domain.prediction.fair_probability.compute_strategy_AB") as mock_AB, \
             patch("domain.prediction.fair_probability.compute_strategy_A") as mock_A, \
             patch("domain.prediction.fair_probability.compute_strategy_B") as mock_B:

            # AB gives 0.47 → edge = 0.47 - 0.46 = 0.01 → below min_edge=0.05
            from engine.prediction.ensemble import StrategyOutput
            mock_AB.return_value = StrategyOutput(
                fair_prob_raw=0.47,
                fair_prob_adjusted=0.47,
                adjustments_applied=[],
                constituent_estimates={"vol_model": 0.47, "garch": 0.47,
                                       "bootstrap": 0.47, "rv_scaling": 0.47},
            )
            mock_A.return_value = StrategyOutput(fair_prob_raw=0.47, fair_prob_adjusted=0.47)
            mock_B.return_value = StrategyOutput(fair_prob_raw=0.47, fair_prob_adjusted=0.47)

            result = detect_edge(**_base_kwargs(market), min_edge=0.05)

        assert result is None

    def test_returns_object_when_above_min_edge(self):
        """When fair_prob >> market_implied_prob (edge >= 5%) → PredictionMarketEdge."""
        # Market at 0.46 mid; mock canonical fair_prob to 0.60 → edge = 0.14 > 0.05
        market = _make_market(yes_bid=0.45, yes_ask=0.47)

        with patch("domain.prediction.fair_probability.compute_strategy_AB") as mock_AB, \
             patch("domain.prediction.fair_probability.compute_strategy_A") as mock_A, \
             patch("domain.prediction.fair_probability.compute_strategy_B") as mock_B:

            from engine.prediction.ensemble import StrategyOutput
            mock_AB.return_value = StrategyOutput(
                fair_prob_raw=0.60,
                fair_prob_adjusted=0.60,
                adjustments_applied=[],
                constituent_estimates={"vol_model": 0.60, "garch": 0.62,
                                       "bootstrap": 0.58, "rv_scaling": 0.60},
            )
            mock_A.return_value = StrategyOutput(
                fair_prob_raw=0.60, fair_prob_adjusted=0.60,
                adjustments_applied=[], constituent_estimates={"vol_model": 0.60},
            )
            mock_B.return_value = StrategyOutput(
                fair_prob_raw=0.60, fair_prob_adjusted=0.60,
                adjustments_applied=[],
                constituent_estimates={"garch": 0.62, "bootstrap": 0.58, "rv_scaling": 0.60},
            )

            result = detect_edge(**_base_kwargs(market), min_edge=0.05)

        assert result is not None
        assert isinstance(result, PredictionMarketEdge)
        assert result.edge == pytest.approx(0.60 - 0.46, abs=0.01)

    def test_returns_none_when_market_expired(self):
        """Expired market → detect_edge returns None."""
        market = _make_market(hours_until_expiry=-1.0)  # already expired
        result = detect_edge(**_base_kwargs(market))
        assert result is None

    def test_edge_object_has_expected_fields(self):
        """The returned PredictionMarketEdge has all required fields populated."""
        market = _make_market(yes_bid=0.45, yes_ask=0.47)

        with patch("domain.prediction.fair_probability.compute_strategy_AB") as mock_AB, \
             patch("domain.prediction.fair_probability.compute_strategy_A") as mock_A, \
             patch("domain.prediction.fair_probability.compute_strategy_B") as mock_B:

            from engine.prediction.ensemble import StrategyOutput
            mock_AB.return_value = StrategyOutput(
                fair_prob_raw=0.65,
                fair_prob_adjusted=0.65,
                adjustments_applied=["mean-reversion:-2%"],
                constituent_estimates={"vol_model": 0.65, "garch": 0.66,
                                       "bootstrap": 0.64, "rv_scaling": 0.65},
            )
            mock_A.return_value = StrategyOutput(
                fair_prob_raw=0.65, fair_prob_adjusted=0.65,
                adjustments_applied=[], constituent_estimates={"vol_model": 0.65},
            )
            mock_B.return_value = StrategyOutput(
                fair_prob_raw=0.65, fair_prob_adjusted=0.65,
                adjustments_applied=[],
                constituent_estimates={"garch": 0.66, "bootstrap": 0.64, "rv_scaling": 0.65},
            )

            result = detect_edge(**_base_kwargs(market), min_edge=0.05)

        assert result is not None
        assert hasattr(result, "fair_prob_A")
        assert hasattr(result, "fair_prob_B")
        assert hasattr(result, "fair_prob_AB")
        assert hasattr(result, "edge")
        assert hasattr(result, "expected_value_per_dollar")
        assert hasattr(result, "estimator_agreement")
        assert hasattr(result, "adjustments_applied")
        assert isinstance(result.estimator_agreement, tuple)
        assert len(result.estimator_agreement) == 2

    def test_zero_bid_ask_returns_none(self):
        """Market with zero bid/ask (market_implied_prob=0) → guard clause → None."""
        market = _make_market(yes_bid=0.0, yes_ask=0.0)
        result = detect_edge(**_base_kwargs(market))
        assert result is None
