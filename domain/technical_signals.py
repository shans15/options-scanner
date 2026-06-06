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


_PO_LAUNCH_BAND = 23.6
_PULLBACK_VOL_MULT = 1.1


def _detect_pullback_in_trend(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    low = df['low']
    volume = df['volume']
    emas = _emas(close)
    atr = _atr14(df['high'], low, close)
    po = _phase_oscillator(close, emas[21], atr)

    e8, e13, e21, e48 = emas[8].iloc[-1], emas[13].iloc[-1], emas[21].iloc[-1], emas[48].iloc[-1]
    e200 = emas[200].iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    last_po = po.iloc[-1]
    last_vol = volume.iloc[-1]
    avg_vol_prev = volume.iloc[-21:-1].mean()

    if pd.isna(last_po) or pd.isna(avg_vol_prev) or avg_vol_prev <= 0:
        return None
    if not (-_PO_LAUNCH_BAND <= last_po <= _PO_LAUNCH_BAND):
        return None
    if last_vol >= _PULLBACK_VOL_MULT * avg_vol_prev:
        return None

    strength = float(1.0 - abs(last_po) / _PO_LAUNCH_BAND)
    strength = max(0.0, min(1.0, strength))

    bullish_stack = e8 > e13 > e21 > e48 and last_close > e200
    bearish_stack = e8 < e13 < e21 < e48 and last_close < e200
    bullish_touch = e21 * 0.99 <= last_low <= e13 * 1.01
    bearish_touch = e13 * 0.99 <= df['high'].iloc[-1] <= e21 * 1.01

    if bullish_stack and bullish_touch:
        return TechnicalSetup(
            setup_name='pullback_in_trend',
            direction='bullish',
            strength=strength,
            notes=f'Ribbon stacked bullish, low tagged 13-21 EMA, PO={last_po:.1f}',
        )
    if bearish_stack and bearish_touch:
        return TechnicalSetup(
            setup_name='pullback_in_trend',
            direction='bearish',
            strength=strength,
            notes=f'Ribbon stacked bearish, high tagged 21-13 EMA, PO={last_po:.1f}',
        )
    return None


_FRESH_CROSS_BARS = 5
_NEAR_52W_PCT = 0.10
_PRIOR_COMPRESSION_LOOKBACK = 20
_PRIOR_COMPRESSION_PCT = 0.25
_EMA200_SLOPE_LOOKBACK = 30


def _crossed_above_within(series: pd.Series, reference: pd.Series, bars: int) -> bool:
    window_s = series.iloc[-(bars + 1):]
    window_r = reference.iloc[-(bars + 1):]
    diff = window_s.values - window_r.values
    return bool((diff[:-1] <= 0).any() and diff[-1] > 0)


def _crossed_below_within(series: pd.Series, reference: pd.Series, bars: int) -> bool:
    window_s = series.iloc[-(bars + 1):]
    window_r = reference.iloc[-(bars + 1):]
    diff = window_s.values - window_r.values
    return bool((diff[:-1] >= 0).any() and diff[-1] < 0)


def _crossed_zero_above(series: pd.Series, bars: int) -> bool:
    window = series.iloc[-(bars + 1):]
    return bool((window.iloc[:-1] <= 0).any() and window.iloc[-1] > 0)


def _crossed_zero_below(series: pd.Series, bars: int) -> bool:
    window = series.iloc[-(bars + 1):]
    return bool((window.iloc[:-1] >= 0).any() and window.iloc[-1] < 0)


def _detect_stage_2_breakout(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    high = df['high']
    low = df['low']
    emas = _emas(close)
    atr = _atr14(high, low, close)
    po = _phase_oscillator(close, emas[21], atr)
    bbw_pct = _po_bandwidth_percentile(po)

    e48 = emas[48]
    e200 = emas[200]
    last_close = close.iloc[-1]
    last_atr = atr.iloc[-1]

    if pd.isna(last_atr) or last_atr <= 0:
        return None
    if pd.isna(e200.iloc[-1]) or pd.isna(e200.iloc[-_EMA200_SLOPE_LOOKBACK - 1]):
        return None

    ema200_rising = e200.iloc[-1] > e200.iloc[-_EMA200_SLOPE_LOOKBACK - 1]
    ema200_falling = e200.iloc[-1] < e200.iloc[-_EMA200_SLOPE_LOOKBACK - 1]

    recent_compression = bbw_pct.iloc[-_PRIOR_COMPRESSION_LOOKBACK:].min()
    if pd.isna(recent_compression) or recent_compression >= _PRIOR_COMPRESSION_PCT:
        return None

    # Bullish branch
    high_52w = high.iloc[-252:].max() if len(high) >= 252 else high.max()
    near_high = last_close >= (1 - _NEAR_52W_PCT) * high_52w

    if (ema200_rising
            and _crossed_above_within(close, e48, _FRESH_CROSS_BARS)
            and _crossed_zero_above(po, _FRESH_CROSS_BARS)
            and near_high):
        strength = float(np.clip((last_close - e48.iloc[-1]) / last_atr, 0.0, 2.0) / 2.0)
        return TechnicalSetup(
            setup_name='stage_2_breakout',
            direction='bullish',
            strength=strength,
            notes='EMA200 rising, fresh EMA48 + PO zero cross, near 52w high',
        )

    # Bearish branch (Stage 4 breakdown)
    low_52w = low.iloc[-252:].min() if len(low) >= 252 else low.min()
    near_low = last_close <= (1 + _NEAR_52W_PCT) * low_52w

    if (ema200_falling
            and _crossed_below_within(close, e48, _FRESH_CROSS_BARS)
            and _crossed_zero_below(po, _FRESH_CROSS_BARS)
            and near_low):
        strength = float(np.clip((e48.iloc[-1] - last_close) / last_atr, 0.0, 2.0) / 2.0)
        return TechnicalSetup(
            setup_name='stage_2_breakout',
            direction='bearish',
            strength=strength,
            notes='EMA200 falling, fresh EMA48 + PO zero cross, near 52w low',
        )

    return None
