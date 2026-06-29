"""Dataclasses used across the A+ confluence scorer.

Each unit is small and has one responsibility. Importing modules
should depend only on this file for type information.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


Grade = Literal['A+', 'A', 'B+', 'B', 'F']


class TradeStructure(str, Enum):
    """Recommended trade structure for a graded candidate."""
    LONG_PREMIUM = 'long_premium'
    DEBIT_SPREAD = 'debit_spread'


@dataclass(frozen=True)
class FeatureScores:
    """Per-feature scores in [0, 10]. Keys match feature names in
    domain/aplus/features.py."""
    values: dict[str, float]


@dataclass(frozen=True)
class CategoryScores:
    """Five category scores in [0, 10]. Average of the constituent
    feature scores per category."""
    technical: float
    vol_vix: float
    catalyst: float
    macro_breadth: float
    liquidity: float

    def composite(self) -> float:
        """Weighted composite, scaled to [0, 100].

        Weights are lift-derived from the 90-day backtest (top-vs-bottom
        tertile 1-day win-rate lift). See:
          docs/superpowers/specs/2026-06-29-aplus-rebalance-design.md
        """
        return (
            0.35 * self.technical
            + 0.10 * self.vol_vix
            + 0.25 * self.catalyst
            + 0.10 * self.macro_breadth
            + 0.20 * self.liquidity
        ) * 10.0


@dataclass(frozen=True)
class MarketContext:
    """Per-scan market data shared across all candidates. Computed once
    when the watchlist runs; reused for every candidate."""
    spx_trend_score: float
    sector_rotation_rank: dict[str, int]   # sector_etf → rank 1..11 by 1w return
    dxy_trend_score: float
    yield_10y_score: float
    vvix_score: float
    days_to_macro_event: int | None


@dataclass(frozen=True)
class GradedCandidate:
    """Final output unit. One per candidate that survived grading."""
    ticker: str
    strategy: str
    composite_score: float
    grade: Grade
    category_scores: CategoryScores
    feature_scores: FeatureScores
    structure: TradeStructure
    structure_rationale: str
    sizing_pct: float          # fraction of account to risk
    max_risk_dollars: float
    raw_candidate: dict        # the original scan-output dict
