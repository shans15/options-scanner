"""Map composite + category floors → A+/A/B+/B/F."""
from __future__ import annotations
from domain.aplus.types import CategoryScores, Grade


def _all_categories_at_least(cs: CategoryScores, floor: float) -> bool:
    return (
        cs.technical >= floor and cs.vol_vix >= floor
        and cs.catalyst >= floor and cs.macro_breadth >= floor
        and cs.liquidity >= floor
    )


def assign_grade(cs: CategoryScores) -> Grade:
    """Apply the grading thresholds from the spec:

        A+ : composite >= 90 AND every category >= 8
        A  : composite >= 80 AND every category >= 7
        B+ : composite >= 70 AND every category >= 6
        B  : composite >= 60
        F  : composite <  60
    """
    composite = cs.composite()

    if composite >= 90.0 and _all_categories_at_least(cs, 8.0):
        return 'A+'
    if composite >= 80.0 and _all_categories_at_least(cs, 7.0):
        return 'A'
    if composite >= 70.0 and _all_categories_at_least(cs, 6.0):
        return 'B+'
    if composite >= 60.0:
        return 'B'
    return 'F'
