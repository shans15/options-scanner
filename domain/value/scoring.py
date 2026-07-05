"""Role 1 (priority tag) + Role 2 (composite) scoring.

Both are computed in parallel on every run so the archived output
supports A/B comparison over multiple monthly cycles.
"""
from __future__ import annotations
from typing import Optional

from domain.value.types import (
    SectorRelativeResult,
    FundamentalDivergenceResult,
    VolumeOverlayResult,
    ShortOverlayResult,
    Role1Score,
    Role2Score,
)


_ROLE2_WEIGHTS: dict[str, float] = {
    'sector_relative': 0.35,
    'fund_divergence': 0.35,
    'volume': 0.15,
    'short_signal': 0.15,
}


def _combined_rank(
    sr: Optional[SectorRelativeResult],
    fd: Optional[FundamentalDivergenceResult],
) -> float:
    """Mean of the two mispricing scores where available."""
    scores: list[float] = []
    if sr is not None:
        scores.append(sr.score)
    if fd is not None:
        scores.append(fd.score)
    if not scores:
        return 5.0
    return sum(scores) / len(scores)


def _priority(rank: float, short: ShortOverlayResult, volume: VolumeOverlayResult) -> str:
    si = short.short_interest_pct
    dtc = short.days_to_cover
    delta = short.short_interest_delta_pp

    if rank >= 7.0:
        # Long lean
        if si is not None and si > 15.0 and dtc is not None and dtc > 5.0:
            return 'HIGH_PRIORITY_SQUEEZE'
        if delta is not None and delta > 1.0:
            return 'SHORTS_BUILDING'
        return 'HIGH_PRIORITY'
    if rank <= 3.0:
        # Short lean
        if si is not None and si > 15.0:
            return 'CROWDED_SHORT_AVOID'
        if si is not None and si >= 5.0:
            return 'PATIENT_SHORT'
        return 'CLEAN_SHORT'
    return 'NONE'


def assign_role1(
    sector_relative: Optional[SectorRelativeResult],
    fundamental_divergence: Optional[FundamentalDivergenceResult],
    volume: VolumeOverlayResult,
    short_overlay: ShortOverlayResult,
) -> Role1Score:
    rank = _combined_rank(sector_relative, fundamental_divergence)
    priority = _priority(rank, short_overlay, volume)
    return Role1Score(combined_rank_score=rank, priority=priority)  # type: ignore[arg-type]


def assign_role2(
    sector_relative: Optional[SectorRelativeResult],
    fundamental_divergence: Optional[FundamentalDivergenceResult],
    volume: VolumeOverlayResult,
    short_overlay: ShortOverlayResult,
) -> Role2Score:
    available: dict[str, float] = {}
    if sector_relative is not None:
        available['sector_relative'] = sector_relative.score
    if fundamental_divergence is not None:
        available['fund_divergence'] = fundamental_divergence.score
    available['volume'] = volume.score
    available['short_signal'] = short_overlay.score

    total_weight = sum(_ROLE2_WEIGHTS[k] for k in available)
    weighted_sum = 0.0
    for k, score in available.items():
        weighted_sum += (_ROLE2_WEIGHTS[k] / total_weight) * score

    return Role2Score(
        composite_score=weighted_sum * 10.0,
        component_breakdown=available,
    )
