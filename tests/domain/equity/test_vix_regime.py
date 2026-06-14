from __future__ import annotations
import pandas as pd
import pytest

from domain.equity.vix_regime import compute_vix_context, VixContext


def _make_series(n: int, base: float = 20.0) -> pd.Series:
    """Return a uniform VIX series of length n with constant value base."""
    return pd.Series([base] * n, dtype=float)


def _make_series_with_spike(n: int, base: float = 19.0, spike: float = 23.0) -> pd.Series:
    """Last value is spike; previous n-1 values are base (simulates expansion)."""
    vals = [base] * (n - 1) + [spike]
    return pd.Series(vals, dtype=float)


def _make_series_with_drop(n: int, base: float = 21.0, drop: float = 16.0) -> pd.Series:
    """Last value is drop; previous n-1 values are base (simulates contraction)."""
    vals = [base] * (n - 1) + [drop]
    return pd.Series(vals, dtype=float)


class TestComputeVixContextExpansion:
    def test_compute_vix_context_returns_expansion_when_vix_above_threshold(self):
        # vix_now = 23, 7d avg ≈ 19 → 23/19 = 1.21 > 1.15 threshold
        series = _make_series_with_spike(30, base=19.0, spike=23.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert ctx.regime == 'expansion'
        assert ctx.vix_now == pytest.approx(23.0)

    def test_expansion_pct_vs_7d_is_positive(self):
        series = _make_series_with_spike(30, base=19.0, spike=23.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert ctx.pct_vs_7d_avg > 0.0

    def test_expansion_notes_include_action(self):
        series = _make_series_with_spike(30, base=19.0, spike=23.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert 'expanding' in ctx.notes


class TestComputeVixContextContraction:
    def test_compute_vix_context_returns_contraction_when_below(self):
        # vix_now = 16, 7d avg ≈ 21 → 16/21 = 0.76 < 0.85 threshold
        series = _make_series_with_drop(30, base=21.0, drop=16.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert ctx.regime == 'contraction'
        assert ctx.vix_now == pytest.approx(16.0)

    def test_contraction_pct_vs_7d_is_negative(self):
        series = _make_series_with_drop(30, base=21.0, drop=16.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert ctx.pct_vs_7d_avg < 0.0

    def test_contraction_notes_include_action(self):
        series = _make_series_with_drop(30, base=21.0, drop=16.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert 'compressing' in ctx.notes


class TestComputeVixContextNeutral:
    def test_compute_vix_context_returns_neutral_in_band(self):
        # Uniform series → vix_now == 7d_avg → pct = 0 → neutral
        series = _make_series(30, base=18.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert ctx.regime == 'neutral'

    def test_neutral_pct_near_zero(self):
        series = _make_series(30, base=18.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        assert abs(ctx.pct_vs_7d_avg) < 0.01


class TestComputeVixContextEdgeCases:
    def test_compute_vix_context_returns_none_for_short_series(self):
        series = _make_series(9, base=20.0)
        assert compute_vix_context(series) is None

    def test_compute_vix_context_returns_none_for_empty_series(self):
        series = pd.Series([], dtype=float)
        assert compute_vix_context(series) is None

    def test_compute_vix_context_returns_none_for_all_nan(self):
        series = pd.Series([float('nan')] * 20)
        assert compute_vix_context(series) is None

    def test_exactly_10_values_is_accepted(self):
        series = _make_series(10, base=20.0)
        ctx = compute_vix_context(series)
        assert ctx is not None


class TestVixContextNotes:
    def test_vix_context_notes_includes_levels(self):
        series = _make_series(30, base=20.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        # Notes should contain the numeric VIX value and averages
        assert 'VIX' in ctx.notes
        assert '20.00' in ctx.notes  # vix_now formatted
        assert '30d avg' in ctx.notes

    def test_vix_context_notes_includes_pct(self):
        series = _make_series_with_spike(30, base=19.0, spike=23.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        # pct should appear as something like "+21.1%" or similar
        assert '%' in ctx.notes

    def test_vix_context_is_frozen(self):
        series = _make_series(30, base=20.0)
        ctx = compute_vix_context(series)
        assert ctx is not None
        with pytest.raises((AttributeError, TypeError)):
            ctx.vix_now = 99.0  # type: ignore[misc]
