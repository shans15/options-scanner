"""Map composite + category floors + vix regime → A+/A/B+/B/F.

Per spec docs/superpowers/specs/2026-06-29-aplus-rebalance-design.md.

Thresholds (percentile-derived from the 90-day backtest):
    A+ : composite >= 75 AND every category >= 6 AND vix_regime != 'expansion'
    A  : composite >= 74 AND every category >= 6 AND vix_regime != 'expansion'
    B+ : composite >= 67
    B  : composite >= 60
    F  : composite <  60  OR  vix_regime == 'expansion'

The expansion-regime hard kill applies independently of composite —
overriding even a perfect 100 score — because the backtest shows
expansion regime has a 31.7% 1-day win rate (n=41) vs. 48.9% in
neutral. This is the strongest single signal in the data set.
"""
from __future__ import annotations
from typing import Optional
from domain.aplus.types import CategoryScores, Grade


def _all_categories_at_least(cs: CategoryScores, floor: float) -> bool:
    return (
        cs.technical >= floor and cs.vol_vix >= floor
        and cs.catalyst >= floor and cs.macro_breadth >= floor
        and cs.liquidity >= floor
    )


def assign_grade(cs: CategoryScores, vix_regime: Optional[str] = None) -> Grade:
    """Return the grade for a CategoryScores under the current spec.

    Args:
        cs: Per-category scores in [0, 10].
        vix_regime: 'expansion' forces F regardless of composite.
                    Any other value (including None) passes through.
    """
    if vix_regime == 'expansion':
        return 'F'

    composite = cs.composite()

    if composite >= 75.0 and _all_categories_at_least(cs, 6.0):
        return 'A+'
    if composite >= 74.0 and _all_categories_at_least(cs, 6.0):
        return 'A'
    if composite >= 67.0:
        return 'B+'
    if composite >= 60.0:
        return 'B'
    return 'F'
