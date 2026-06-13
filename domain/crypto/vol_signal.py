from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VolSignal:
    """Structured vol trading signal derived from model probability output."""

    direction: Literal["expansion", "contraction", "neutral"]
    probability: float           # raw model output (P(vol expansion))
    strength: Literal["strong", "mild", "neutral"]
    suggested_action: str        # human-readable trade recommendation
    notes: str                   # context / caveats


def classify_signal(probability: float) -> VolSignal:
    """Map raw model probability to a tradeable signal.

    Thresholds
    ----------
    >= 0.75 : STRONG expansion  → long straddle on BTC/ETH proxies
    >= 0.65 : MILD expansion    → long strangle (cheaper, less directional)
    >= 0.35 and < 0.65 : NEUTRAL → no trade
    < 0.35  : MILD contraction  → sell premium (v2 — too risky for retail v1)
    < 0.25  : STRONG contraction → iron condor (v2)

    Parameters
    ----------
    probability : float
        Model output: P(vol expansion), in [0, 1].

    Returns
    -------
    VolSignal
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be in [0, 1], got {probability:.4f}"
        )

    if probability >= 0.75:
        return VolSignal(
            direction="expansion",
            probability=probability,
            strength="strong",
            suggested_action=(
                "Long straddle on BTC/ETH proxies "
                "(buy ATM call + ATM put, e.g. MSTR, IBIT, ETHA)"
            ),
            notes=(
                "Strong expansion signal. Model confidence is high. "
                "Straddle profits if spot moves > break-even in either direction. "
                "Max loss = total premium paid."
            ),
        )

    if probability >= 0.65:
        return VolSignal(
            direction="expansion",
            probability=probability,
            strength="mild",
            suggested_action=(
                "Long strangle on BTC/ETH proxies "
                "(buy OTM call + OTM put — lower cost, needs larger move to profit)"
            ),
            notes=(
                "Mild expansion signal. Use a strangle (1-2 strikes OTM) rather "
                "than a straddle to reduce premium outlay. "
                "Max loss = total premium paid."
            ),
        )

    if probability >= 0.35:
        return VolSignal(
            direction="neutral",
            probability=probability,
            strength="neutral",
            suggested_action="No trade — model is not decisive",
            notes=(
                f"Probability {probability:.2f} falls in the neutral band [0.35, 0.65). "
                "Wait for a clearer signal before entering a vol position."
            ),
        )

    if probability >= 0.25:
        return VolSignal(
            direction="contraction",
            probability=probability,
            strength="mild",
            suggested_action=(
                "Sell premium — v2 placeholder "
                "(short straddle / covered strangle requires margin; not recommended for v1)"
            ),
            notes=(
                "Mild contraction signal. Premium selling strategies (short straddle, "
                "short strangle) can profit but carry unlimited risk without a hedge. "
                "Deferred to v2."
            ),
        )

    # probability < 0.25 — strong contraction
    return VolSignal(
        direction="contraction",
        probability=probability,
        strength="strong",
        suggested_action=(
            "Iron condor — v2 placeholder "
            "(defined-risk short-vol; requires level-3 options approval)"
        ),
        notes=(
            "Strong contraction signal. An iron condor profits in a low-vol, "
            "range-bound environment. Requires understanding of spread mechanics "
            "and margin. Deferred to v2."
        ),
    )
