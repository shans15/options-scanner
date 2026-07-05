"""Dataclasses used across the Value scanner.

Each unit is small and has one responsibility.  Importing modules
should depend only on this file for type information.

Per spec docs/superpowers/specs/2026-07-02-fundamental-value-scanner-design.md
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


VolumeTag = Literal['RISING', 'FLAT', 'DECLINING']
Direction = Literal['long', 'short']
Role1Priority = Literal[
    'HIGH_PRIORITY_SQUEEZE',
    'HIGH_PRIORITY',
    'PATIENT',
    'STALE_REASSESS',
    'SHORTS_BUILDING',
    'CLEAN_SHORT',
    'PATIENT_SHORT',
    'CROWDED_SHORT_AVOID',
    'NONE',
]


@dataclass(frozen=True)
class ValuationInputs:
    """Structured, per-ticker fundamentals + price + volume + short data.

    Any field may be None if the underlying data source did not
    supply it — downstream methodologies decide how to handle missing
    fields (skip vs neutral score)."""
    ticker: str
    sector: Optional[str]
    price: Optional[float]
    forward_eps: Optional[float]
    revenue_ttm: Optional[float]
    revenue_ttm_1y_ago: Optional[float]
    eps_ttm: Optional[float]
    eps_ttm_1y_ago: Optional[float]
    book_value_per_share: Optional[float]
    enterprise_value: Optional[float]
    ebitda_ttm: Optional[float]
    shares_short: Optional[float]
    float_shares: Optional[float]
    avg_daily_volume_30d: Optional[float]
    volume_5d_avg: Optional[float]
    volume_20d_avg: Optional[float]
    price_1y_ago: Optional[float]


@dataclass(frozen=True)
class SectorRelativeResult:
    score: float                       # 0-10
    pe_fwd: Optional[float] = None
    pe_fwd_peer_median: Optional[float] = None
    pe_discount_pct: Optional[float] = None
    ps_ttm: Optional[float] = None
    ps_ttm_peer_median: Optional[float] = None
    ps_discount_pct: Optional[float] = None
    pb: Optional[float] = None
    pb_peer_median: Optional[float] = None
    pb_discount_pct: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_ebitda_peer_median: Optional[float] = None
    ratios_used: int = 0


@dataclass(frozen=True)
class FundamentalDivergenceResult:
    score: float                       # 0-10
    revenue_growth_yoy_pct: Optional[float] = None
    eps_growth_yoy_pct: Optional[float] = None
    price_growth_yoy_pct: Optional[float] = None
    divergence_pp: Optional[float] = None


@dataclass(frozen=True)
class VolumeOverlayResult:
    score: float                       # 0, 5, or 10
    tag: VolumeTag
    ratio: Optional[float] = None
    vol_5d_avg: Optional[float] = None
    vol_20d_avg: Optional[float] = None


@dataclass(frozen=True)
class ShortOverlayResult:
    score: float                       # 0-10 (direction-aware)
    short_interest_pct: Optional[float] = None
    days_to_cover: Optional[float] = None
    short_interest_delta_pp: Optional[float] = None


@dataclass(frozen=True)
class Role1Score:
    combined_rank_score: float         # 0-10 (mean of sector_rel + fund_div where available)
    priority: Role1Priority


@dataclass(frozen=True)
class Role2Score:
    composite_score: float             # 0-100
    component_breakdown: dict          # {'sector_relative': 8.5, ...}


@dataclass(frozen=True)
class ValuationSnapshot:
    """All computed results for one ticker in one monthly run."""
    ticker: str
    inputs: ValuationInputs
    sector_relative: Optional[SectorRelativeResult]
    fundamental_divergence: Optional[FundamentalDivergenceResult]
    volume: Optional[VolumeOverlayResult]
    short_overlay: Optional[ShortOverlayResult]
    role1: Optional[Role1Score]
    role2: Optional[Role2Score]
    skip_reason: Optional[str] = None      # populated iff ticker was skipped entirely


@dataclass(frozen=True)
class MispricingReport:
    """Final output for a single monthly run."""
    run_timestamp_utc: str
    universe: str
    universe_size: int
    graded_size: int
    skipped: int
    skipped_reasons: dict[str, int]
    role1_ranked_longs: list[dict]
    role1_ranked_shorts: list[dict]
    role2_ranked_longs: list[dict]
    role2_ranked_shorts: list[dict]
