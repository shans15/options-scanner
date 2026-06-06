from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd


WeeklyTrend = Literal['up', 'down', 'flat']


@dataclass(frozen=True)
class MarketContext:
    weekly_trend: WeeklyTrend
    consecutive_close_streak: int             # signed: +5 = up 5 sessions, -3 = down 3
    pct_change_1w: float                      # decimal: 0.025 = +2.5%
    pct_change_2w: float
    pct_change_4w: float
    weekly_ribbon_agreement: bool             # daily ribbon stack matches weekly ribbon stack


_NEUTRAL = MarketContext(
    weekly_trend='flat',
    consecutive_close_streak=0,
    pct_change_1w=0.0,
    pct_change_2w=0.0,
    pct_change_4w=0.0,
    weekly_ribbon_agreement=False,
)


def compute_market_context(close: pd.Series) -> MarketContext:
    """Compute multi-timeframe context from a daily close-only Series.
    Returns a neutral MarketContext when there isn't enough history."""
    if close is None or len(close) < 30:
        return _NEUTRAL

    last = float(close.iloc[-1])

    # Consecutive close streak — count back from the last bar while sign stays the same
    diffs = close.diff().dropna()
    if len(diffs) == 0:
        streak = 0
    else:
        last_d = diffs.iloc[-1]
        if last_d > 0:
            sign = 1
        elif last_d < 0:
            sign = -1
        else:
            sign = 0
        streak = 0
        for d in reversed(diffs.values):
            if sign == 0:
                break
            if (d > 0 and sign == 1) or (d < 0 and sign == -1):
                streak += sign
            else:
                break

    # Percentage changes (~5 trading days per week)
    def _pct(n_bars: int) -> float:
        if len(close) <= n_bars:
            return 0.0
        prev = float(close.iloc[-n_bars - 1])
        return (last / prev) - 1.0 if prev > 0 else 0.0

    pct_1w = _pct(5)
    pct_2w = _pct(10)
    pct_4w = _pct(20)

    # Weekly trend — resample to weekly closes, slope of last 4 weekly bars
    try:
        weekly_close = close.resample('W').last().dropna()
    except (TypeError, ValueError):
        weekly_close = close.iloc[::5].dropna()

    if len(weekly_close) >= 4:
        recent = weekly_close.iloc[-4:]
        slope = float(recent.iloc[-1] - recent.iloc[0])
        if slope > recent.iloc[0] * 0.005:        # > 0.5% over 4 weeks → up
            weekly_trend: WeeklyTrend = 'up'
        elif slope < -recent.iloc[0] * 0.005:
            weekly_trend = 'down'
        else:
            weekly_trend = 'flat'
    else:
        weekly_trend = 'flat'

    # Daily ribbon direction (8/13/21 EMA stack)
    e8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    e13 = close.ewm(span=13, adjust=False).mean().iloc[-1]
    e21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    daily_bullish = e8 > e13 > e21
    daily_bearish = e8 < e13 < e21

    # Weekly ribbon agreement — same EMAs computed on weekly closes
    if len(weekly_close) >= 22:
        w8 = weekly_close.ewm(span=8, adjust=False).mean().iloc[-1]
        w13 = weekly_close.ewm(span=13, adjust=False).mean().iloc[-1]
        w21 = weekly_close.ewm(span=21, adjust=False).mean().iloc[-1]
        weekly_bullish = w8 > w13 > w21
        weekly_bearish = w8 < w13 < w21
        agreement = bool((daily_bullish and weekly_bullish) or (daily_bearish and weekly_bearish))
    else:
        agreement = False

    return MarketContext(
        weekly_trend=weekly_trend,
        consecutive_close_streak=streak,
        pct_change_1w=pct_1w,
        pct_change_2w=pct_2w,
        pct_change_4w=pct_4w,
        weekly_ribbon_agreement=agreement,
    )
