from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd


SetupName = Literal[
    'compression_breakout',
    'pullback_in_trend',
    'stage_2_breakout',
    'failed_breakdown_reversal',
]
Direction = Literal['bullish', 'bearish']


@dataclass(frozen=True)
class TechnicalSetup:
    setup_name: SetupName
    direction: Direction
    strength: float
    notes: str


_EMA_LENGTHS = (8, 13, 21, 48, 200)
_ATR_PERIOD = 14
_PO_BBW_INNER = 20         # rolling std window on PO
_PO_BBW_OUTER = 126        # percentile rank window (~6 months)


def _emas(close: pd.Series) -> dict[int, pd.Series]:
    return {n: close.ewm(span=n, adjust=False).mean() for n in _EMA_LENGTHS}


def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing = EMA with alpha = 1/period
    return tr.ewm(alpha=1 / _ATR_PERIOD, adjust=False).mean()


def _phase_oscillator(close: pd.Series, ema21: pd.Series, atr14: pd.Series) -> pd.Series:
    # Guard against zero ATR (would produce inf). Replace with NaN.
    safe_atr = atr14.where(atr14 > 0)
    return (close - ema21) / safe_atr


def _po_bandwidth_percentile(po: pd.Series) -> pd.Series:
    """Compression metric: rolling stdev of PO, then percentile-rank over 6 months.
    Lower percentile = tighter PO = stronger squeeze."""
    po_std = po.rolling(_PO_BBW_INNER).std()
    return po_std.rolling(_PO_BBW_OUTER).rank(pct=True)
