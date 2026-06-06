from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
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


_COMPRESSION_PCT_CUTOFF = 0.20
_NOT_EXTENDED_ATR_MULT = 0.5


def _detect_compression_breakout(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    emas = _emas(close)
    atr = _atr14(df['high'], df['low'], close)
    po = _phase_oscillator(close, emas[21], atr)
    bbw_pct = _po_bandwidth_percentile(po)

    last_bbw = bbw_pct.iloc[-1]
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    last_atr = atr.iloc[-1]

    if pd.isna(last_bbw) or pd.isna(last_atr) or last_atr <= 0:
        return None
    if last_bbw >= _COMPRESSION_PCT_CUTOFF:
        return None
    if abs(last_close - prev_close) >= _NOT_EXTENDED_ATR_MULT * last_atr:
        return None

    e8, e13, e21, e48 = emas[8].iloc[-1], emas[13].iloc[-1], emas[21].iloc[-1], emas[48].iloc[-1]
    strength = float(1.0 - last_bbw)

    if e8 > e13 > e21 and last_close > e48:
        return TechnicalSetup(
            setup_name='compression_breakout',
            direction='bullish',
            strength=strength,
            notes=f'PO squeeze pct={last_bbw:.2f}, ribbon stacked bullish',
        )
    if e8 < e13 < e21 and last_close < e48:
        return TechnicalSetup(
            setup_name='compression_breakout',
            direction='bearish',
            strength=strength,
            notes=f'PO squeeze pct={last_bbw:.2f}, ribbon stacked bearish',
        )
    return None
