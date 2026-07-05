"""MispricingReport assembly + serialisation."""
from domain.value.types import (
    ValuationInputs, ValuationSnapshot,
    SectorRelativeResult, FundamentalDivergenceResult,
    VolumeOverlayResult, ShortOverlayResult,
    Role1Score, Role2Score,
)
from domain.value.report import build_report, snapshot_to_dict


def _snap(ticker: str, rank: float, composite: float, skip: str = None):
    v = ValuationInputs(
        ticker=ticker, sector='Technology', price=100.0,
        forward_eps=5.0, revenue_ttm=1e9, revenue_ttm_1y_ago=8e8,
        eps_ttm=5.0, eps_ttm_1y_ago=4.0, book_value_per_share=20.0,
        enterprise_value=1e10, ebitda_ttm=1e9,
        shares_short=100_000_000, float_shares=1_000_000_000,
        avg_daily_volume_30d=1e7, volume_5d_avg=1.2e7, volume_20d_avg=1e7,
        price_1y_ago=80.0,
    )
    if skip:
        return ValuationSnapshot(
            ticker=ticker, inputs=v,
            sector_relative=None, fundamental_divergence=None,
            volume=None, short_overlay=None, role1=None, role2=None,
            skip_reason=skip,
        )
    return ValuationSnapshot(
        ticker=ticker, inputs=v,
        sector_relative=SectorRelativeResult(score=rank, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=rank),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0, short_interest_pct=15.0),
        role1=Role1Score(combined_rank_score=rank, priority='HIGH_PRIORITY'),
        role2=Role2Score(composite_score=composite,
                          component_breakdown={'sector_relative': rank,
                                                'fund_divergence': rank,
                                                'volume': 10.0,
                                                'short_signal': 10.0}),
        skip_reason=None,
    )


def test_report_ranks_top_longs_by_role1():
    snaps = [
        _snap('A', rank=9.0, composite=85.0),
        _snap('B', rank=7.5, composite=80.0),
        _snap('C', rank=5.0, composite=60.0),
    ]
    r = build_report(
        snapshots=snaps, universe_name='russell2000',
        universe_size=3, run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=2,
    )
    assert [x['ticker'] for x in r.role1_ranked_longs] == ['A', 'B']


def test_report_ranks_top_shorts_by_role1_ascending():
    snaps = [
        _snap('A', rank=9.0, composite=85.0),
        _snap('B', rank=1.5, composite=15.0),
        _snap('C', rank=2.5, composite=25.0),
    ]
    r = build_report(
        snapshots=snaps, universe_name='russell2000',
        universe_size=3, run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=2,
    )
    assert [x['ticker'] for x in r.role1_ranked_shorts] == ['B', 'C']


def test_report_skipped_counts():
    snaps = [
        _snap('A', rank=9.0, composite=85.0),
        _snap('X', rank=0, composite=0, skip='missing_fundamentals'),
        _snap('Y', rank=0, composite=0, skip='revenue_too_small'),
        _snap('Z', rank=0, composite=0, skip='missing_fundamentals'),
    ]
    r = build_report(
        snapshots=snaps, universe_name='russell2000',
        universe_size=4, run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=5,
    )
    assert r.graded_size == 1
    assert r.skipped == 3
    assert r.skipped_reasons == {'missing_fundamentals': 2, 'revenue_too_small': 1}


def test_snapshot_to_dict_has_all_keys():
    snap = _snap('A', rank=9.0, composite=85.0)
    d = snapshot_to_dict(snap)
    assert d['ticker'] == 'A'
    assert d['sector_relative_score'] == 9.0
    assert d['composite_score'] == 85.0
    assert 'priority' in d
    assert 'fundamental_trend' in d
    assert 'sector_multiples' in d


def test_snapshot_with_role1_but_no_role2_excluded_from_graded():
    """Defensive: if a snapshot has role1 but not role2 (unlikely but possible),
    it must not enter the graded set, or role2 sort would crash."""
    v = ValuationInputs(
        ticker='PARTIAL', sector='Technology', price=100.0,
        forward_eps=5.0, revenue_ttm=1e9, revenue_ttm_1y_ago=8e8,
        eps_ttm=5.0, eps_ttm_1y_ago=4.0, book_value_per_share=20.0,
        enterprise_value=1e10, ebitda_ttm=1e9,
        shares_short=100_000_000, float_shares=1_000_000_000,
        avg_daily_volume_30d=1e7, volume_5d_avg=1.2e7, volume_20d_avg=1e7,
        price_1y_ago=80.0,
    )
    partial = ValuationSnapshot(
        ticker='PARTIAL', inputs=v,
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=None,
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0),
        role1=Role1Score(combined_rank_score=8.0, priority='HIGH_PRIORITY'),
        role2=None,          # <-- the guard case
        skip_reason=None,
    )
    normal = _snap('OK', rank=7.0, composite=70.0)

    r = build_report(
        snapshots=[partial, normal],
        universe_name='russell2000',
        universe_size=2,
        run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=5,
    )
    # PARTIAL should NOT appear in role2_ranked_longs
    tickers = [x['ticker'] for x in r.role2_ranked_longs]
    assert 'PARTIAL' not in tickers
    assert 'OK' in tickers
