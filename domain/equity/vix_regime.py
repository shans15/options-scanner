from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd


VixRegime = Literal['expansion', 'contraction', 'neutral']


@dataclass(frozen=True)
class VixContext:
    vix_now: float                       # current VIX level
    vix_7d_avg: float                    # 7-day average VIX
    vix_30d_avg: float                   # 30-day average (regime baseline)
    regime: VixRegime
    pct_vs_7d_avg: float                 # (vix_now − vix_7d_avg) / vix_7d_avg
    notes: str                           # human-readable summary


_EXPANSION_THRESHOLD = 1.15              # vix_now > 7d_avg × 1.15 → expansion
_CONTRACTION_THRESHOLD = 0.85            # vix_now < 7d_avg × 0.85 → contraction


def compute_vix_context(vix_history: pd.Series) -> Optional[VixContext]:
    """Given a Series of VIX daily closes (last 30+ days), compute the current
    vol regime context.

    Returns None if the series is too short or all-NaN.
    """
    series = vix_history.dropna()
    if len(series) < 10:
        return None
    vix_now = float(series.iloc[-1])
    vix_7d_avg = float(series.tail(7).mean())
    vix_30d_avg = float(series.tail(30).mean())
    pct_vs_7d = (vix_now - vix_7d_avg) / vix_7d_avg if vix_7d_avg > 0 else 0.0

    if vix_now > vix_7d_avg * _EXPANSION_THRESHOLD:
        regime: VixRegime = 'expansion'
        action = 'equity vol expanding → directional setups likely to follow through; long premium favored'
    elif vix_now < vix_7d_avg * _CONTRACTION_THRESHOLD:
        regime = 'contraction'
        action = 'equity vol compressing → directional setups bleed theta; favor shorter DTE or skip directional long premium'
    else:
        regime = 'neutral'
        action = 'no strong vol regime signal'

    notes = (
        f"VIX {vix_now:.2f}, 7d avg {vix_7d_avg:.2f} "
        f"({pct_vs_7d*100:+.1f}%), 30d avg {vix_30d_avg:.2f}. {action}"
    )

    return VixContext(
        vix_now=vix_now,
        vix_7d_avg=vix_7d_avg,
        vix_30d_avg=vix_30d_avg,
        regime=regime,
        pct_vs_7d_avg=pct_vs_7d,
        notes=notes,
    )
