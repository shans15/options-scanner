"""Methodology A — sector-relative valuation multiples.

Compare a ticker's core valuation ratios to the median of its GICS
sector peers.  Score in [0, 10], higher = more undervalued.

P/S uses `float_shares` as a proxy for shares outstanding.  This is a
small approximation for names where float ≠ shares outstanding, but
close enough for a screener.
"""
from __future__ import annotations
import statistics
from typing import Optional

from domain.value.types import ValuationInputs, SectorRelativeResult


_RATIO_KEYS = ('pe_fwd', 'ps_ttm', 'pb', 'ev_ebitda')

# Maps ratio key → discount_pct field name on SectorRelativeResult (None = no discount field)
_DISCOUNT_FIELD: dict[str, Optional[str]] = {
    'pe_fwd': 'pe_discount_pct',
    'ps_ttm': 'ps_discount_pct',
    'pb': 'pb_discount_pct',
    'ev_ebitda': None,
}


def _ratios(v: ValuationInputs) -> dict[str, Optional[float]]:
    """Compute the four ratios; None when input is missing or math undefined."""
    pe = ps = pb = ev_ebitda = None

    if v.price is not None and v.forward_eps and v.forward_eps > 0:
        pe = v.price / v.forward_eps

    if v.price is not None and v.book_value_per_share and v.book_value_per_share > 0:
        pb = v.price / v.book_value_per_share

    if v.enterprise_value and v.ebitda_ttm and v.ebitda_ttm > 0:
        ev_ebitda = v.enterprise_value / v.ebitda_ttm

    # P/S = price / (revenue_ttm / shares).  Use float_shares as a proxy for
    # shares outstanding — good enough for a value screener.
    if (v.price is not None and v.revenue_ttm and v.revenue_ttm > 0
            and v.float_shares and v.float_shares > 0):
        revenue_per_share = v.revenue_ttm / v.float_shares
        if revenue_per_share > 0:
            ps = v.price / revenue_per_share

    return {'pe_fwd': pe, 'ps_ttm': ps, 'pb': pb, 'ev_ebitda': ev_ebitda}


def compute_sector_medians(
    inputs: list[ValuationInputs],
    min_peers: int = 10,
) -> dict[str, dict[str, float]]:
    """Group inputs by sector, compute median of each ratio.

    Sectors qualify iff at least one ratio has >= min_peers values.
    Only ratios meeting the min_peers floor produce medians for that sector.
    """
    by_sector: dict[str, dict[str, list[float]]] = {}
    for v in inputs:
        if not v.sector:
            continue
        r = _ratios(v)
        bucket = by_sector.setdefault(v.sector, {k: [] for k in _RATIO_KEYS})
        for key in _RATIO_KEYS:
            val = r.get(key)
            if val is not None and val > 0:
                bucket[key].append(val)

    medians: dict[str, dict[str, float]] = {}
    for sector, ratios in by_sector.items():
        counts = {k: len(vs) for k, vs in ratios.items()}
        if not counts:
            continue
        max_n = max(counts.values())
        if max_n < min_peers:
            continue
        medians[sector] = {
            k: statistics.median(vs) for k, vs in ratios.items() if len(vs) >= min_peers
        }
    return medians


def _discount_pct(ticker_ratio: float, peer_median: float) -> float:
    """Return (peer_median - ticker_ratio) / peer_median as a percentage."""
    return (peer_median - ticker_ratio) / peer_median * 100.0


def _score_from_discount(discount_pct: float) -> float:
    """Map a single-ratio discount percentage to a 0-10 score."""
    if discount_pct >= 30.0:
        return 10.0
    if discount_pct >= 15.0:
        return 7.0 + (discount_pct - 15.0) / 15.0 * 3.0
    if discount_pct >= -15.0:
        return 5.0 + discount_pct / 15.0 * 2.0
    if discount_pct >= -30.0:
        return 2.0 + (discount_pct + 30.0) / 15.0 * 2.0
    return max(0.0, 1.0 + (discount_pct + 45.0) / 15.0)


def compute_sector_relative(
    v: ValuationInputs,
    sector_medians: dict[str, dict[str, float]],
) -> Optional[SectorRelativeResult]:
    """Score a single ticker against its sector's median ratios.

    Returns None if the ticker's sector is not in `sector_medians`
    OR the ticker has fewer than 2 computable ratios with peer medians.
    """
    if not v.sector or v.sector not in sector_medians:
        return None
    peer_medians = sector_medians[v.sector]
    ticker_ratios = _ratios(v)

    per_ratio_scores: list[float] = []
    result_fields: dict = {}

    for key in _RATIO_KEYS:
        r = ticker_ratios.get(key)
        pm = peer_medians.get(key)
        if r is None or pm is None or pm <= 0:
            continue
        d = _discount_pct(r, pm)
        s = _score_from_discount(d)
        per_ratio_scores.append(s)
        result_fields[key] = r
        result_fields[f'{key}_peer_median'] = pm
        discount_field = _DISCOUNT_FIELD.get(key)
        if discount_field is not None:
            result_fields[discount_field] = d

    if len(per_ratio_scores) < 2:
        return None

    return SectorRelativeResult(
        score=sum(per_ratio_scores) / len(per_ratio_scores),
        ratios_used=len(per_ratio_scores),
        **result_fields,
    )
