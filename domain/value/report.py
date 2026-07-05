"""Assemble ValuationSnapshots into a MispricingReport with ranked lists."""
from __future__ import annotations
from collections import Counter
from typing import Iterable

from domain.value.types import ValuationSnapshot, MispricingReport


def snapshot_to_dict(snap: ValuationSnapshot) -> dict:
    """Serialize a ValuationSnapshot to the flat dict used in JSON output."""
    v = snap.inputs
    sr = snap.sector_relative
    fd = snap.fundamental_divergence
    vol = snap.volume
    short = snap.short_overlay
    r1 = snap.role1
    r2 = snap.role2

    d: dict = {
        'ticker': snap.ticker,
        'sector': v.sector,
        'spot': v.price,
    }
    if sr is not None:
        d['sector_relative_score'] = sr.score
        d['sector_multiples'] = {
            'pe_fwd': sr.pe_fwd, 'pe_fwd_peer_median': sr.pe_fwd_peer_median,
            'pe_discount_pct': sr.pe_discount_pct,
            'ps_ttm': sr.ps_ttm, 'ps_ttm_peer_median': sr.ps_ttm_peer_median,
            'ps_discount_pct': sr.ps_discount_pct,
            'pb': sr.pb, 'pb_peer_median': sr.pb_peer_median,
            'pb_discount_pct': sr.pb_discount_pct,
            'ev_ebitda': sr.ev_ebitda, 'ev_ebitda_peer_median': sr.ev_ebitda_peer_median,
        }
    if fd is not None:
        d['fundamental_divergence_score'] = fd.score
        d['fundamental_trend'] = {
            'revenue_growth_yoy_pct': fd.revenue_growth_yoy_pct,
            'eps_growth_yoy_pct': fd.eps_growth_yoy_pct,
            'price_growth_yoy_pct': fd.price_growth_yoy_pct,
            'divergence_pp': fd.divergence_pp,
        }
    if vol is not None:
        d['volume_tag'] = vol.tag
        d['volume_score'] = vol.score
        d['volume'] = {'vol_5d_avg': vol.vol_5d_avg,
                       'vol_20d_avg': vol.vol_20d_avg,
                       'ratio': vol.ratio}
    if short is not None:
        d['short_signal_score'] = short.score
        d['short_interest_pct'] = short.short_interest_pct
        d['days_to_cover'] = short.days_to_cover
        d['short_interest_delta_pp'] = short.short_interest_delta_pp
    if r1 is not None:
        d['combined_rank_score'] = r1.combined_rank_score
        d['priority'] = r1.priority
    if r2 is not None:
        d['composite_score'] = r2.composite_score
        d['component_breakdown'] = r2.component_breakdown
    return d


def build_report(
    snapshots: Iterable[ValuationSnapshot],
    universe_name: str,
    universe_size: int,
    run_timestamp_utc: str,
    top_n: int = 20,
) -> MispricingReport:
    all_snaps = list(snapshots)
    graded = [s for s in all_snaps if s.skip_reason is None and s.role1 is not None]
    skipped = [s for s in all_snaps if s.skip_reason is not None]

    skipped_reasons = Counter(s.skip_reason for s in skipped)

    # Role 1 rankings
    role1_longs = sorted(graded, key=lambda s: -s.role1.combined_rank_score)[:top_n]
    role1_shorts = sorted(graded, key=lambda s: s.role1.combined_rank_score)[:top_n]

    # Role 2 rankings
    role2_longs = sorted(graded, key=lambda s: -s.role2.composite_score)[:top_n]
    role2_shorts = sorted(graded, key=lambda s: s.role2.composite_score)[:top_n]

    return MispricingReport(
        run_timestamp_utc=run_timestamp_utc,
        universe=universe_name,
        universe_size=universe_size,
        graded_size=len(graded),
        skipped=len(skipped),
        skipped_reasons=dict(skipped_reasons),
        role1_ranked_longs=[snapshot_to_dict(s) for s in role1_longs],
        role1_ranked_shorts=[snapshot_to_dict(s) for s in role1_shorts],
        role2_ranked_longs=[snapshot_to_dict(s) for s in role2_longs],
        role2_ranked_shorts=[snapshot_to_dict(s) for s in role2_shorts],
    )
