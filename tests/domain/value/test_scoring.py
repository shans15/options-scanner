"""Role 1 tags + Role 2 composite scoring."""
from domain.value.types import (
    SectorRelativeResult, FundamentalDivergenceResult,
    VolumeOverlayResult, ShortOverlayResult,
)
from domain.value.scoring import assign_role1, assign_role2


# ---- Role 1 ----

def test_role1_high_priority_squeeze_when_undervalued_high_si_high_dtc():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=8.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0, short_interest_pct=18.0, days_to_cover=6.0),
    )
    assert r.priority == 'HIGH_PRIORITY_SQUEEZE'
    assert r.combined_rank_score == 8.0


def test_role1_high_priority_when_undervalued_clean():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=7.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=6.0, short_interest_pct=8.0),
    )
    assert r.priority == 'HIGH_PRIORITY'


def test_role1_crowded_short_when_overvalued_high_si():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=2.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=2.0),
        volume=VolumeOverlayResult(score=0.0, tag='DECLINING'),
        short_overlay=ShortOverlayResult(score=0.0, short_interest_pct=20.0),
    )
    assert r.priority == 'CROWDED_SHORT_AVOID'


def test_role1_clean_short_when_overvalued_low_si():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=2.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=2.0),
        volume=VolumeOverlayResult(score=0.0, tag='DECLINING'),
        short_overlay=ShortOverlayResult(score=10.0, short_interest_pct=3.0),
    )
    assert r.priority == 'CLEAN_SHORT'


def test_role1_none_priority_for_middle_ranks():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=5.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=5.0),
        volume=VolumeOverlayResult(score=5.0, tag='FLAT'),
        short_overlay=ShortOverlayResult(score=5.0),
    )
    assert r.priority == 'NONE'


def test_role1_combined_rank_uses_only_available_scores():
    r = assign_role1(
        sector_relative=None,
        fundamental_divergence=FundamentalDivergenceResult(score=8.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=6.0),
    )
    assert r.combined_rank_score == 8.0


def test_role1_shorts_building_tag():
    """Undervalued + SI delta > +1 pp → SHORTS_BUILDING (when NOT satisfying SQUEEZE)."""
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=7.5),
        volume=VolumeOverlayResult(score=5.0, tag='FLAT'),
        short_overlay=ShortOverlayResult(score=10.0, short_interest_pct=17.0,
                                          days_to_cover=4.0, short_interest_delta_pp=1.5),
    )
    assert r.priority == 'SHORTS_BUILDING'


# ---- Role 2 ----

def test_role2_composite_full_weights():
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=8.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0),
    )
    # 0.35*8 + 0.35*8 + 0.15*10 + 0.15*10 = 8.6 → *10 = 86.0
    assert abs(r.composite_score - 86.0) < 1e-9


def test_role2_composite_all_zero():
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=0.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=0.0),
        volume=VolumeOverlayResult(score=0.0, tag='DECLINING'),
        short_overlay=ShortOverlayResult(score=0.0),
    )
    assert r.composite_score == 0.0


def test_role2_weights_redistribute_when_component_missing():
    """If fundamental_divergence is None, remaining components get proportional weight redistribution."""
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=10.0, ratios_used=3),
        fundamental_divergence=None,
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0),
    )
    # All scores 10 → composite = 10 * 10 = 100
    assert abs(r.composite_score - 100.0) < 1e-9


def test_role2_component_breakdown_present():
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=7.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=6.0),
    )
    assert r.component_breakdown['sector_relative'] == 8.0
    assert r.component_breakdown['fund_divergence'] == 7.0
    assert r.component_breakdown['volume'] == 10.0
    assert r.component_breakdown['short_signal'] == 6.0
