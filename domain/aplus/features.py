"""Per-candidate feature extractors for the A+ confluence scorer.

Each function returns a float in [0, 10]. Missing inputs return 5.0 (neutral)
unless the spec specifies a different default (e.g., VIX expansion forces 0).
"""
from __future__ import annotations
from typing import Optional
from datetime import date


_SETUP_TYPE_SCORES = {
    'compression_breakout': 10.0,
    'stage_2_breakout': 8.0,
    'failed_breakdown_reversal': 5.0,
    'pullback_in_trend': 3.0,
}


def score_setup_type(setup_name: Optional[str]) -> float:
    if setup_name is None:
        return 5.0
    return _SETUP_TYPE_SCORES.get(setup_name, 5.0)


def score_setup_strength(strength: Optional[float]) -> float:
    if strength is None:
        return 5.0
    return max(0.0, min(10.0, strength * 10.0))


def score_weekly_ribbon_agreement(agreement: Optional[bool]) -> float:
    if agreement is None:
        return 5.0
    return 10.0 if agreement else 5.0


def score_atr_pivot_position(spot: float, prev_close: float, atr: float) -> float:
    """10 if spot is at prior daily close; linear decline to 0 at 2 ATR away."""
    if atr <= 0:
        return 5.0
    distance = abs(spot - prev_close) / atr
    return max(0.0, 10.0 - distance * 5.0)


def score_volume_zscore(z: float) -> float:
    """Higher relative volume scores higher. 10 at z=2, 0 at z=-1."""
    return max(0.0, min(10.0, (z + 1.0) * (10.0 / 3.0)))


def score_vix_regime(regime: Optional[str]) -> float:
    if regime == 'expansion':
        return 0.0
    if regime == 'contraction':
        return 10.0
    if regime == 'neutral':
        return 7.0
    return 5.0


def score_vix_zscore_30d(z: float) -> float:
    """Lower VIX z-score scores higher; higher z penalises."""
    return max(0.0, min(10.0, 10.0 - z * 6.0))


def score_vvix_level(category_score: float) -> float:
    """Pass-through — MarketContext already converted VVIX level to a score."""
    return max(0.0, min(10.0, category_score))


def score_iv_percentile(iv: float) -> float:
    """For LONG premium scoring: low IV is better. 10 at IV<=20%, 0 at IV>=60%."""
    if iv <= 0.20:
        return 10.0
    if iv >= 0.60:
        return 0.0
    return 10.0 - (iv - 0.20) / 0.40 * 10.0
