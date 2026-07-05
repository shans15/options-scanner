"""Short interest overlay — direction-aware scoring.

For LONG candidates (base_rank >= 5):
  - High SI (> 15%) → squeeze validation, score 10
  - Medium (5-15%) → normal, score 6
  - Low (< 5%)     → clean value, score 5 (neutral)

For SHORT candidates (base_rank < 5):
  - High SI (> 15%) → crowded short, score 0 (danger)
  - Medium (5-15%) → some crowd, score 4
  - Low (< 5%)     → fresh short, score 10
"""
from __future__ import annotations
from typing import Optional

from domain.value.types import ValuationInputs, ShortOverlayResult


def compute_short_overlay(
    v: ValuationInputs,
    base_rank: float,
    prior_short_interest_pct: Optional[float],
) -> ShortOverlayResult:
    if (v.shares_short is None or v.float_shares is None
            or v.float_shares <= 0):
        return ShortOverlayResult(score=5.0)

    si_pct = v.shares_short / v.float_shares * 100.0

    days_to_cover: Optional[float] = None
    if v.avg_daily_volume_30d and v.avg_daily_volume_30d > 0:
        days_to_cover = v.shares_short / v.avg_daily_volume_30d

    delta_pp: Optional[float] = None
    if prior_short_interest_pct is not None:
        delta_pp = si_pct - prior_short_interest_pct

    # Direction-aware scoring
    if base_rank >= 5.0:
        if si_pct > 15.0:
            score = 10.0
        elif si_pct >= 5.0:
            score = 6.0
        else:
            score = 5.0
    else:
        if si_pct > 15.0:
            score = 0.0
        elif si_pct >= 5.0:
            score = 4.0
        else:
            score = 10.0

    return ShortOverlayResult(
        score=score,
        short_interest_pct=si_pct,
        days_to_cover=days_to_cover,
        short_interest_delta_pp=delta_pp,
    )
