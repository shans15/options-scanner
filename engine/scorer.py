from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

from domain.contract import Contract
from domain.strategy import Strategy
from domain.signals import Regime
from engine.stress import StressResult
from engine.risk_filters import FilterResult


Label = Literal['TRADE', 'WATCHLIST', 'NO_TRADE']


@dataclass
class ScoredCandidate:
    contract: Contract
    strategy: Strategy
    pop_blended: float
    pop_delta: float
    pop_bs: float
    pop_historical: float
    pop_garch_mc: float
    stress: StressResult
    ev: float
    max_adverse_loss: float
    margin_estimate: float
    filter_result: FilterResult
    composite_score: float
    label: Label
    reason_for: str
    reason_against: str
    setup_name: Optional[str] = None
    setup_direction: Optional[str] = None
    setup_strength: Optional[float] = None
    # NEW — display-only context (not used in composite_score)
    weekly_trend: Optional[str] = None
    consecutive_close_streak: Optional[int] = None
    pct_change_1w: Optional[float] = None
    pct_change_2w: Optional[float] = None
    pct_change_4w: Optional[float] = None
    weekly_ribbon_agreement: Optional[bool] = None
    # VIX-regime context tag (equity-only, display-only, not used in composite_score)
    vix_now: Optional[float] = None
    vix_regime: Optional[str] = None
    vix_pct_vs_7d: Optional[float] = None


def _regime_alignment_score(strategy: Strategy, regime: Regime) -> float:
    if strategy.direction in regime.favored and len(regime.favored) == 1:
        return 1.0
    if strategy.direction in regime.favored:
        return 0.6
    return 0.3


def composite_score(*, pop_blended: float, ev: float, max_adverse_loss: float,
                    strategy: Strategy, regime: Regime, mid: float) -> float:
    pop_pts = 50.0 * np.clip(pop_blended, 0, 1)
    if strategy.direction == 'sell':
        denom = max_adverse_loss if max_adverse_loss > 0 else 1.0
        ev_score = float(np.clip(ev / denom, 0.0, 1.0))
    else:
        ev_score = float(np.clip(ev / max(mid, 1e-9), 0.0, 1.0))
    ev_pts = 30.0 * ev_score
    regime_pts = 20.0 * _regime_alignment_score(strategy, regime)
    return round(pop_pts + ev_pts + regime_pts, 2)


def label_from(score: float, all_filters_pass: bool) -> Label:
    if not all_filters_pass:
        return 'NO_TRADE'
    if score >= 65.0:
        return 'TRADE'
    if score >= 50.0:
        return 'WATCHLIST'
    return 'NO_TRADE'
