import numpy as np
import pandas as pd
import pytest

from domain.market_context import MarketContext, compute_market_context


def _series_from_closes(closes: list[float]) -> pd.Series:
    idx = pd.date_range('2025-01-01', periods=len(closes), freq='B')
    return pd.Series(closes, index=idx)


def test_neutral_context_when_too_short():
    s = _series_from_closes([100.0] * 20)
    out = compute_market_context(s)
    assert out.weekly_trend == 'flat'
    assert out.consecutive_close_streak == 0
    assert out.weekly_ribbon_agreement is False


def test_up_streak_detected():
    # 30 flat-ish bars then 5 strictly increasing closes
    closes = [100.0] * 30 + [101.0, 102.0, 103.0, 104.0, 105.0]
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.consecutive_close_streak == 5


def test_down_streak_detected():
    closes = [100.0] * 30 + [99.0, 98.0, 97.0]
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.consecutive_close_streak == -3


def test_pct_changes_match_expected():
    # Build a smooth uptrend so 1w/2w/4w pct changes are positive and ordered
    closes = list(np.linspace(80.0, 120.0, 60))
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.pct_change_4w > out.pct_change_2w > out.pct_change_1w > 0


def test_weekly_trend_up_on_uptrend():
    closes = list(np.linspace(80.0, 120.0, 250))
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.weekly_trend == 'up'


def test_weekly_trend_down_on_downtrend():
    closes = list(np.linspace(120.0, 80.0, 250))
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.weekly_trend == 'down'


def test_weekly_ribbon_agreement_true_on_clean_uptrend():
    closes = list(np.linspace(80.0, 120.0, 250))
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.weekly_ribbon_agreement is True


def test_weekly_ribbon_agreement_false_on_disagreement():
    # Long downtrend (makes weekly EMAs bearish) followed by a short-term spike
    # (makes daily EMAs bullish). Daily and weekly ribbons disagree => no agreement.
    closes_down = list(np.linspace(120.0, 90.0, 200))
    closes_up = list(np.linspace(90.0, 95.0, 50))
    closes = closes_down + closes_up
    s = _series_from_closes(closes)
    out = compute_market_context(s)
    assert out.weekly_ribbon_agreement is False


def test_compute_does_not_raise_on_nan_or_short_close():
    s = pd.Series(dtype=float)
    out = compute_market_context(s)
    assert out.weekly_trend == 'flat'
