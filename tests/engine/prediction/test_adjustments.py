"""Tests for engine/prediction/adjustments.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engine.prediction.adjustments import (
    apply_favorite_longshot_bias,
    apply_mean_reversion,
    apply_round_number_magnetism,
    apply_time_of_day_vol,
)


class TestFavoriteLongshotBias:
    def test_long_shot_shifts_toward_half(self):
        """Input 0.05 → output > 0.05 (shifted toward 0.5)."""
        adjusted, label = apply_favorite_longshot_bias(0.05)
        assert adjusted > 0.05
        assert label is not None
        assert "favorite-longshot" in label

    def test_favorite_shifts_toward_half(self):
        """Input 0.95 → output < 0.95 (shifted toward 0.5)."""
        adjusted, label = apply_favorite_longshot_bias(0.95)
        assert adjusted < 0.95
        assert label is not None
        assert "favorite-longshot" in label

    def test_middle_no_change(self):
        """Input 0.5 → no adjustment."""
        adjusted, label = apply_favorite_longshot_bias(0.5)
        assert adjusted == pytest.approx(0.5, abs=0.001)
        assert label is None

    def test_borderline_no_change(self):
        """0.15 exactly is the boundary — no adjustment (not < 0.15)."""
        adjusted, label = apply_favorite_longshot_bias(0.15)
        assert adjusted == pytest.approx(0.15, abs=0.001)
        assert label is None

    def test_borderline_above_high(self):
        """0.85 is the high boundary — no adjustment (not > 0.85)."""
        adjusted, label = apply_favorite_longshot_bias(0.85)
        assert adjusted == pytest.approx(0.85, abs=0.001)
        assert label is None

    def test_adjusted_prob_stays_in_range(self):
        """Adjustment never pushes probability outside [0, 1]."""
        for p in [0.01, 0.05, 0.10, 0.14, 0.86, 0.90, 0.95, 0.99]:
            adjusted, _ = apply_favorite_longshot_bias(p)
            assert 0.0 <= adjusted <= 1.0


class TestRoundNumberMagnetism:
    def test_round_strike_within_range_bumps_prob(self):
        """strike=$63,000 + within 3% → P += 0.025."""
        spot = 62_500.0  # within 3% of 63k
        adjusted, label = apply_round_number_magnetism(
            fair_prob=0.40, strike=63_000.0, spot=spot, is_test_strike=True
        )
        assert adjusted == pytest.approx(0.425, abs=0.001)
        assert label == "round-number-magnetism:+2.5%"

    def test_non_round_strike_no_change(self):
        """Non-round strike → no adjustment."""
        adjusted, label = apply_round_number_magnetism(
            fair_prob=0.40, strike=63_250.0, spot=62_500.0, is_test_strike=True
        )
        assert adjusted == pytest.approx(0.40, abs=0.001)
        assert label is None

    def test_round_strike_but_too_far_no_change(self):
        """Round strike but spot is > 3% away → no adjustment."""
        spot = 50_000.0  # far from 63k
        adjusted, label = apply_round_number_magnetism(
            fair_prob=0.40, strike=63_000.0, spot=spot, is_test_strike=True
        )
        assert adjusted == pytest.approx(0.40, abs=0.001)
        assert label is None

    def test_non_test_strike_no_change(self):
        """is_test_strike=False → no adjustment even for round strike."""
        spot = 62_500.0
        adjusted, label = apply_round_number_magnetism(
            fair_prob=0.40, strike=63_000.0, spot=spot, is_test_strike=False
        )
        assert adjusted == pytest.approx(0.40, abs=0.001)
        assert label is None

    def test_adjusted_prob_capped_at_one(self):
        adjusted, _ = apply_round_number_magnetism(
            fair_prob=0.99, strike=63_000.0, spot=62_500.0, is_test_strike=True
        )
        assert adjusted <= 1.0


class TestMeanReversion:
    def test_large_up_move_dampens_continuation_prob(self):
        """After last bar was up > 1σ, P(BTC > strike) decreases."""
        hourly_sigma = 0.01
        last_hour_return = 0.03  # 3 * sigma = large up move
        original = 0.60
        adjusted, label = apply_mean_reversion(
            fair_prob=original,
            last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        assert adjusted < original, "Large up move should dampen continuation P"
        assert label is not None

    def test_small_move_no_adjustment(self):
        """Small move (|return| < 1σ) → no adjustment."""
        hourly_sigma = 0.01
        last_hour_return = 0.005  # 0.5 * sigma
        original = 0.60
        adjusted, label = apply_mean_reversion(
            fair_prob=original,
            last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        assert adjusted == pytest.approx(original, abs=0.001)
        assert label is None

    def test_large_down_move_increases_prob(self):
        """After large down move, P(above strike) increases (mean reversion)."""
        hourly_sigma = 0.01
        last_hour_return = -0.03  # large down move
        original = 0.40
        adjusted, label = apply_mean_reversion(
            fair_prob=original,
            last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        assert adjusted > original

    def test_zero_sigma_no_adjustment(self):
        """Zero sigma → guard clause, no adjustment."""
        adjusted, label = apply_mean_reversion(
            fair_prob=0.5,
            last_hour_return=0.10,
            hourly_sigma=0.0,
        )
        assert adjusted == pytest.approx(0.5, abs=0.001)
        assert label is None

    def test_adjusted_prob_clamped_to_range(self):
        """Extreme adjustment never produces out-of-range prob."""
        adjusted, _ = apply_mean_reversion(
            fair_prob=0.01,
            last_hour_return=-1.0,
            hourly_sigma=0.001,
        )
        assert 0.0 <= adjusted <= 1.0


class TestTimeOfDayVol:
    def _make_utc(self, hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 6, 13, hour, minute, tzinfo=timezone.utc)

    def test_us_open_multiplier(self):
        """14:00 UTC falls in US open (13:30-15:00) → sigma × 1.20."""
        expiry = self._make_utc(14, 0)
        spot = 63_000.0
        strike = 64_000.0
        sigma_T = 0.02

        adjusted, label = apply_time_of_day_vol(
            fair_prob=0.40,
            expiration_utc=expiry,
            spot=spot,
            strike=strike,
            sigma_T=sigma_T,
        )
        assert label is not None
        assert "us-open" in label
        # Inflated sigma should shift probability
        assert adjusted != pytest.approx(0.40, abs=0.001)

    def test_other_hours_no_change(self):
        """10:00 UTC (not a session open/close) → no adjustment."""
        expiry = self._make_utc(10, 0)
        original = 0.50
        adjusted, label = apply_time_of_day_vol(
            fair_prob=original,
            expiration_utc=expiry,
            spot=63_000.0,
            strike=63_000.0,
            sigma_T=0.02,
        )
        assert adjusted == pytest.approx(original, abs=0.001)
        assert label is None

    def test_asia_open_triggers(self):
        """23:00 UTC falls in Asia open (22:00-02:00)."""
        expiry = self._make_utc(23, 0)
        adjusted, label = apply_time_of_day_vol(
            fair_prob=0.40,
            expiration_utc=expiry,
            spot=63_000.0,
            strike=64_000.0,
            sigma_T=0.02,
        )
        assert label is not None
        assert "asia-open" in label

    def test_eu_open_triggers(self):
        """07:00 UTC falls in EU open (06:00-09:00)."""
        expiry = self._make_utc(7, 0)
        adjusted, label = apply_time_of_day_vol(
            fair_prob=0.40,
            expiration_utc=expiry,
            spot=63_000.0,
            strike=64_000.0,
            sigma_T=0.02,
        )
        assert label is not None
        assert "eu-open" in label

    def test_zero_sigma_no_change(self):
        """sigma_T=0 → guard clause, no adjustment."""
        expiry = self._make_utc(14, 0)
        adjusted, label = apply_time_of_day_vol(
            fair_prob=0.50,
            expiration_utc=expiry,
            spot=63_000.0,
            strike=64_000.0,
            sigma_T=0.0,
        )
        assert adjusted == pytest.approx(0.50, abs=0.001)
        assert label is None

    def test_adjusted_prob_in_range(self):
        """Result always in [0, 1]."""
        for hour in [1, 7, 14, 19]:
            expiry = self._make_utc(hour, 45)
            adjusted, _ = apply_time_of_day_vol(
                fair_prob=0.50,
                expiration_utc=expiry,
                spot=63_000.0,
                strike=64_000.0,
                sigma_T=0.02,
            )
            assert 0.0 <= adjusted <= 1.0
