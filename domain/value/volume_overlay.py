"""Volume overlay — 5-day average / 20-day average ratio + tag.

Applies to both Role 1 (tag) and Role 2 (score component).  Missing
volume data → FLAT, score 5 (neutral).
"""
from __future__ import annotations

from domain.value.types import ValuationInputs, VolumeOverlayResult


def compute_volume_overlay(v: ValuationInputs) -> VolumeOverlayResult:
    if v.volume_5d_avg is None or v.volume_20d_avg is None or v.volume_20d_avg <= 0:
        return VolumeOverlayResult(score=5.0, tag='FLAT', ratio=None,
                                    vol_5d_avg=v.volume_5d_avg,
                                    vol_20d_avg=v.volume_20d_avg)
    ratio = v.volume_5d_avg / v.volume_20d_avg
    if ratio >= 1.15:
        return VolumeOverlayResult(score=10.0, tag='RISING', ratio=ratio,
                                    vol_5d_avg=v.volume_5d_avg,
                                    vol_20d_avg=v.volume_20d_avg)
    if ratio <= 0.85:
        return VolumeOverlayResult(score=0.0, tag='DECLINING', ratio=ratio,
                                    vol_5d_avg=v.volume_5d_avg,
                                    vol_20d_avg=v.volume_20d_avg)
    return VolumeOverlayResult(score=5.0, tag='FLAT', ratio=ratio,
                                vol_5d_avg=v.volume_5d_avg,
                                vol_20d_avg=v.volume_20d_avg)
