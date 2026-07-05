"""Methodology B — fundamental momentum divergence.

Compare 12-month fundamental trends (revenue + EPS) to 12-month price
trends.  Divergence in favour of fundamentals = undervalued.
"""
from __future__ import annotations
from typing import Optional

from domain.value.types import ValuationInputs, FundamentalDivergenceResult


_MIN_REVENUE_USD = 50_000_000


def _score_from_divergence(divergence_pp: float) -> float:
    """Map divergence in percentage points to a 0-10 score.

    Bands:
      > +25 pp   fundamentals sprinting ahead of price → undervalued (8-10)
      +10 to +25 mild undervaluation                     (6-8)
      -10 to +10 aligned                                 (4-6)
      -25 to -10 mild overvaluation                      (2-4)
      < -25 pp   price sprinted ahead of fundamentals → overvalued (0-2)
    """
    if divergence_pp > 25:
        return 8.0 + min(2.0, (divergence_pp - 25.0) / 25.0 * 2.0)
    if divergence_pp > 10:
        return 6.0 + (divergence_pp - 10.0) / 15.0 * 2.0
    if divergence_pp >= -10:
        return 4.0 + (divergence_pp + 10.0) / 20.0 * 2.0
    if divergence_pp >= -25:
        return 2.0 + (divergence_pp + 25.0) / 15.0 * 2.0
    return max(0.0, 2.0 + (divergence_pp + 25.0) / 25.0 * 2.0)


def compute_fundamental_divergence(
    v: ValuationInputs,
) -> Optional[FundamentalDivergenceResult]:
    """Return a score + component fields, or None if skipped."""
    # Guardrail: revenue must exist and exceed floor
    if not v.revenue_ttm or v.revenue_ttm < _MIN_REVENUE_USD:
        return None
    if not v.revenue_ttm_1y_ago or v.revenue_ttm_1y_ago <= 0:
        return None
    if v.price is None or v.price_1y_ago is None or v.price_1y_ago <= 0:
        return None

    revenue_growth = (v.revenue_ttm - v.revenue_ttm_1y_ago) / v.revenue_ttm_1y_ago * 100.0

    eps_growth: Optional[float] = None
    if v.eps_ttm is not None and v.eps_ttm_1y_ago is not None:
        # Both negative → revenue-only weighting
        if v.eps_ttm < 0 and v.eps_ttm_1y_ago < 0:
            eps_growth = None
        elif v.eps_ttm * v.eps_ttm_1y_ago < 0:
            # Sign flip → growth rate undefined; skip ticker
            return None
        elif v.eps_ttm_1y_ago == 0:
            return None
        else:
            eps_growth = (v.eps_ttm - v.eps_ttm_1y_ago) / abs(v.eps_ttm_1y_ago) * 100.0

    if eps_growth is None:
        fundamental_growth = revenue_growth
    else:
        fundamental_growth = 0.6 * revenue_growth + 0.4 * eps_growth

    price_growth = (v.price - v.price_1y_ago) / v.price_1y_ago * 100.0
    divergence = fundamental_growth - price_growth

    return FundamentalDivergenceResult(
        score=_score_from_divergence(divergence),
        revenue_growth_yoy_pct=revenue_growth,
        eps_growth_yoy_pct=eps_growth,
        price_growth_yoy_pct=price_growth,
        divergence_pp=divergence,
    )
