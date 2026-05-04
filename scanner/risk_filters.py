from __future__ import annotations
from dataclasses import dataclass
import os

MAX_SPREAD_PCT = float(os.getenv('MAX_SPREAD_PCT', '0.20'))
MIN_VOLUME = int(os.getenv('MIN_VOLUME', '100'))
MIN_OI = int(os.getenv('MIN_OI', '500'))
MIN_POP = float(os.getenv('MIN_POP_THRESHOLD', '0.70'))
MAX_STRESS_MULTIPLIER = 3.0
DELTA_MIN = 0.05
DELTA_MAX = 0.35


@dataclass
class FilterResult:
    passed: bool
    reason: str  # empty string if passed


def apply_hard_filters(contract: dict, pop_result: object) -> FilterResult:
    """
    Apply all hard filters. Any failure returns passed=False with reason.
    pop_result can be a PopResult dataclass or a dict with same keys.
    """
    def _get(obj: object, key: str, default: object = None) -> object:
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    bid = contract.get('bid', 0)
    ask = contract.get('ask', 0)
    mid = contract.get('mid', 0)
    volume = contract.get('volume', 0)
    oi = contract.get('open_interest', 0)
    delta = contract.get('delta', 0)
    premium = mid

    pop_blended = _get(pop_result, 'pop_blended', 0)
    ev = _get(pop_result, 'expected_value', 0)
    stress_obj = _get(pop_result, 'stress', None)
    stress_2sd = getattr(stress_obj, 'stress_2sd', 0) if stress_obj else 0

    # 1. Bid/ask spread
    if mid > 0:
        spread_pct = (ask - bid) / mid
        if spread_pct > MAX_SPREAD_PCT:
            return FilterResult(False, f"Bid/ask spread {spread_pct:.1%} exceeds {MAX_SPREAD_PCT:.0%} of mid")

    # 2. Volume
    if volume < MIN_VOLUME:
        return FilterResult(False, f"Volume {volume} below minimum {MIN_VOLUME}")

    # 3. Open interest
    if oi < MIN_OI:
        return FilterResult(False, f"Open interest {oi} below minimum {MIN_OI}")

    # 4. Delta range
    abs_delta = abs(delta)
    if not (DELTA_MIN <= abs_delta <= DELTA_MAX):
        return FilterResult(False, f"Delta {delta:.2f} outside target range [{DELTA_MIN},{DELTA_MAX}]")

    # 5. Blended PoP
    if pop_blended < MIN_POP:
        return FilterResult(False, f"Blended PoP {pop_blended:.1%} below minimum {MIN_POP:.0%}")

    # 6. Expected value
    if ev <= 0:
        return FilterResult(False, f"Expected value {ev:.4f} is not positive")

    # 7. Stress loss (2SD must be <= 3x premium)
    stress_loss = abs(min(stress_2sd, 0))
    if stress_loss > MAX_STRESS_MULTIPLIER * premium:
        return FilterResult(
            False,
            f"2SD stress loss ${stress_loss:.2f} exceeds {MAX_STRESS_MULTIPLIER}x premium ${premium:.2f}"
        )

    return FilterResult(True, '')
