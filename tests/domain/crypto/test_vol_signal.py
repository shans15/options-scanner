from __future__ import annotations

import pytest

from domain.crypto.vol_signal import VolSignal, classify_signal


# ---------------------------------------------------------------------------
# Boundary and classification correctness
# ---------------------------------------------------------------------------

class TestClassifySignalBoundaries:
    """Test every threshold boundary for classify_signal."""

    def test_strong_expansion_at_075(self):
        sig = classify_signal(0.75)
        assert sig.direction == "expansion"
        assert sig.strength == "strong"

    def test_strong_expansion_above_075(self):
        sig = classify_signal(0.90)
        assert sig.direction == "expansion"
        assert sig.strength == "strong"

    def test_strong_expansion_at_1(self):
        sig = classify_signal(1.0)
        assert sig.direction == "expansion"
        assert sig.strength == "strong"

    def test_mild_expansion_at_065(self):
        sig = classify_signal(0.65)
        assert sig.direction == "expansion"
        assert sig.strength == "mild"

    def test_mild_expansion_just_below_075(self):
        """0.7499 is below 0.75 threshold → mild, not strong."""
        sig = classify_signal(0.7499)
        assert sig.direction == "expansion"
        assert sig.strength == "mild"

    def test_mild_expansion_midrange(self):
        sig = classify_signal(0.70)
        assert sig.direction == "expansion"
        assert sig.strength == "mild"

    def test_neutral_at_035(self):
        sig = classify_signal(0.35)
        assert sig.direction == "neutral"
        assert sig.strength == "neutral"

    def test_neutral_at_050(self):
        sig = classify_signal(0.50)
        assert sig.direction == "neutral"
        assert sig.strength == "neutral"

    def test_neutral_just_below_065(self):
        """0.6499 is below 0.65 threshold → neutral."""
        sig = classify_signal(0.6499)
        assert sig.direction == "neutral"
        assert sig.strength == "neutral"

    def test_neutral_just_above_035(self):
        """0.3501 is above 0.35 → still neutral."""
        sig = classify_signal(0.3501)
        assert sig.direction == "neutral"
        assert sig.strength == "neutral"

    def test_mild_contraction_just_below_035(self):
        """0.3499 falls below 0.35 → mild contraction."""
        sig = classify_signal(0.3499)
        assert sig.direction == "contraction"
        assert sig.strength == "mild"

    def test_mild_contraction_at_025(self):
        """0.25 is the boundary between mild and strong contraction.
        >= 0.25 and < 0.35 → mild."""
        sig = classify_signal(0.25)
        assert sig.direction == "contraction"
        assert sig.strength == "mild"

    def test_strong_contraction_just_below_025(self):
        """0.2499 → strong contraction."""
        sig = classify_signal(0.2499)
        assert sig.direction == "contraction"
        assert sig.strength == "strong"

    def test_strong_contraction_at_zero(self):
        sig = classify_signal(0.0)
        assert sig.direction == "contraction"
        assert sig.strength == "strong"

    def test_strong_contraction_at_010(self):
        sig = classify_signal(0.10)
        assert sig.direction == "contraction"
        assert sig.strength == "strong"


class TestClassifySignalProbabilityPassthrough:
    """probability field must always equal the input value."""

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 1.0])
    def test_probability_field_matches_input(self, p: float):
        sig = classify_signal(p)
        assert sig.probability == p


class TestClassifySignalReturnType:
    """classify_signal must always return a frozen VolSignal dataclass."""

    @pytest.mark.parametrize("p", [0.0, 0.2, 0.35, 0.50, 0.65, 0.80, 1.0])
    def test_returns_vol_signal(self, p: float):
        sig = classify_signal(p)
        assert isinstance(sig, VolSignal)

    @pytest.mark.parametrize("p", [0.0, 0.2, 0.35, 0.50, 0.65, 0.80, 1.0])
    def test_is_frozen(self, p: float):
        sig = classify_signal(p)
        with pytest.raises((AttributeError, TypeError)):
            sig.direction = "changed"  # type: ignore[misc]


class TestClassifySignalInvalidInput:
    """Out-of-range inputs must raise ValueError."""

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="probability must be in"):
            classify_signal(-0.01)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="probability must be in"):
            classify_signal(1.01)

    def test_nan_raises(self):
        import math
        with pytest.raises((ValueError, Exception)):
            result = classify_signal(float("nan"))
            # If it doesn't raise at classify, it should at least return something
            # with direction=None or fail a subsequent assertion. Either is fine.


class TestClassifySignalSuggestedAction:
    """suggested_action must be non-empty for all inputs."""

    @pytest.mark.parametrize("p", [0.0, 0.2, 0.35, 0.50, 0.65, 0.80, 1.0])
    def test_suggested_action_non_empty(self, p: float):
        sig = classify_signal(p)
        assert isinstance(sig.suggested_action, str)
        assert len(sig.suggested_action) > 0

    @pytest.mark.parametrize("p", [0.0, 0.2, 0.35, 0.50, 0.65, 0.80, 1.0])
    def test_notes_non_empty(self, p: float):
        sig = classify_signal(p)
        assert isinstance(sig.notes, str)
        assert len(sig.notes) > 0


class TestClassifySignalStraddleStrangle:
    """Strong signals should mention straddle; mild expansion should mention strangle."""

    def test_strong_expansion_mentions_straddle(self):
        sig = classify_signal(0.80)
        assert "straddle" in sig.suggested_action.lower()

    def test_mild_expansion_mentions_strangle(self):
        sig = classify_signal(0.68)
        assert "strangle" in sig.suggested_action.lower()

    def test_neutral_mentions_no_trade(self):
        sig = classify_signal(0.50)
        assert "no trade" in sig.suggested_action.lower()
