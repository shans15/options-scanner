"""Verify types.py exposes the dataclasses used across the value module."""
from domain.value.types import (
    ValuationInputs,
    SectorRelativeResult,
    FundamentalDivergenceResult,
    VolumeOverlayResult,
    ShortOverlayResult,
    Role1Score,
    Role2Score,
    ValuationSnapshot,
    MispricingReport,
)


def test_valuation_inputs_holds_all_fields():
    v = ValuationInputs(
        ticker='INTC', sector='Technology', price=24.11,
        forward_eps=2.87, revenue_ttm=54_000_000_000,
        revenue_ttm_1y_ago=48_000_000_000,
        eps_ttm=1.4, eps_ttm_1y_ago=1.29,
        book_value_per_share=21.9, enterprise_value=100_000_000_000,
        ebitda_ttm=15_400_000_000,
        shares_short=180_000_000, float_shares=1_040_000_000,
        avg_daily_volume_30d=29_000_000,
        volume_5d_avg=45_000_000, volume_20d_avg=38_000_000,
        price_1y_ago=29.55,
    )
    assert v.ticker == 'INTC'
    assert v.price == 24.11


def test_valuation_snapshot_combines_all_partial_results():
    """A ValuationSnapshot is the per-ticker container of all sub-results."""
    v = ValuationInputs(
        ticker='X', sector='Technology', price=10.0,
        forward_eps=None, revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None, book_value_per_share=None,
        enterprise_value=None, ebitda_ttm=None,
        shares_short=None, float_shares=None, avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None, price_1y_ago=None,
    )
    snap = ValuationSnapshot(
        ticker='X', inputs=v,
        sector_relative=None, fundamental_divergence=None,
        volume=None, short_overlay=None,
        role1=None, role2=None,
        skip_reason='no_fundamentals',
    )
    assert snap.ticker == 'X'
    assert snap.skip_reason == 'no_fundamentals'


def test_mispricing_report_has_four_lists():
    r = MispricingReport(
        run_timestamp_utc='2026-07-02T18:22:00Z',
        universe='russell2000',
        universe_size=2000,
        graded_size=1847,
        skipped=153,
        skipped_reasons={'missing_fundamentals': 98},
        role1_ranked_longs=[],
        role1_ranked_shorts=[],
        role2_ranked_longs=[],
        role2_ranked_shorts=[],
    )
    assert r.universe == 'russell2000'
    assert isinstance(r.role1_ranked_longs, list)
