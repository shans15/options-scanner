from __future__ import annotations
from dataclasses import dataclass, field
import math
from domain.contract import Contract
from domain.strategy import Strategy
from engine.stress import StressResult


@dataclass(frozen=True)
class FilterSet:
    max_spread_pct: float
    min_volume: int
    min_oi: int
    delta_range: tuple[float, float]
    min_pop: float
    min_ev: float
    max_stress_loss_multiple: float


FILTER_SETS: dict[str, FilterSet] = {
    'naked_put':  FilterSet(0.20, 100, 500, (0.16, 0.30), 0.70, 0.0,  3.0),
    'naked_call': FilterSet(0.20, 100, 500, (0.16, 0.30), 0.70, 0.0,  3.0),
    'long_put':   FilterSet(0.20, 100, 500, (0.40, 0.60), 0.40, 0.15, math.inf),
    'long_call':  FilterSet(0.20, 100, 500, (0.40, 0.60), 0.40, 0.15, math.inf),
}


@dataclass
class FilterResult:
    passed: bool
    failed_filters: list[str] = field(default_factory=list)


def apply_filters(c: Contract, strategy: Strategy, pop_blended: float, ev: float, stress: StressResult) -> FilterResult:
    fs = FILTER_SETS[strategy.name]
    failed: list[str] = []

    spread = c.ask - c.bid
    if c.mid <= 0 or spread / c.mid > fs.max_spread_pct:
        failed.append('spread')
    if c.volume < fs.min_volume:
        failed.append('volume')
    if c.open_interest < fs.min_oi:
        failed.append('open_interest')
    abs_d = abs(c.delta)
    if abs_d < fs.delta_range[0] or abs_d > fs.delta_range[1]:
        failed.append('delta')
    if pop_blended < fs.min_pop:
        failed.append('pop')
    if ev < fs.min_ev:
        failed.append('ev')
    if strategy.direction == 'sell' and math.isfinite(fs.max_stress_loss_multiple):
        worst = abs(min(stress.stress_2sd, 0.0))
        if c.mid > 0 and worst > fs.max_stress_loss_multiple * c.mid:
            failed.append('stress')

    return FilterResult(passed=(len(failed) == 0), failed_filters=failed)
