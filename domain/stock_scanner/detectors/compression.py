from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from domain.stock_scanner.types import StockSetup


_BB_PERIOD = 20
_BB_STD = 2.0
_LOOKBACK = 126  # 6 months trading days


def detect(ticker: str, history: pd.DataFrame) -> Optional[StockSetup]:
    """Detect Bollinger Band width compression vs 6-month history.

    Compression = current BBW <= 30th percentile of last 126 days.
    Direction inferred from EMA50 trend (up = bullish, down = bearish).

    Strength = 0.6 * tightness + 0.4 * duration
      tightness = 1 - (current_bbw / max_bbw_126)
      duration  = consecutive bars where BBW <= 30th percentile, capped at 20.
    """
    if len(history) < _LOOKBACK + _BB_PERIOD:
        return None

    closes = history["close"]
    mid = closes.rolling(_BB_PERIOD).mean()
    std = closes.rolling(_BB_PERIOD).std()
    upper = mid + _BB_STD * std
    lower = mid - _BB_STD * std
    bbw = (upper - lower) / mid

    bbw_recent = bbw.dropna().iloc[-_LOOKBACK:]
    if len(bbw_recent) < 30:
        return None
    current_bbw = float(bbw.iloc[-1])
    threshold = float(np.percentile(bbw_recent, 30))

    if current_bbw > threshold:
        return None  # Not compressed

    # Direction
    ema50 = closes.ewm(span=50, adjust=False).mean()
    e50_now = float(ema50.iloc[-1])
    e50_20 = float(ema50.iloc[-20])
    direction = "bullish" if e50_now > e50_20 else "bearish"

    # Strength
    max_bbw = float(bbw_recent.max())
    tightness = 1.0 - (current_bbw / max_bbw)

    consecutive = 0
    for i in range(len(bbw_recent) - 1, -1, -1):
        if float(bbw_recent.iloc[i]) <= threshold:
            consecutive += 1
        else:
            break
    duration_score = min(consecutive / 20.0, 1.0)

    strength = 0.6 * tightness + 0.4 * duration_score
    if strength < 0.5:
        return None

    return StockSetup(
        ticker=ticker,
        setup_type="compression",
        direction=direction,
        strength=round(strength, 3),
        spot=float(closes.iloc[-1]),
        notes={
            "current_bbw_pct": round(current_bbw * 100, 2),
            "threshold_bbw_pct": round(threshold * 100, 2),
            "max_bbw_pct_6mo": round(max_bbw * 100, 2),
            "compression_days": consecutive,
            "ema50_slope_20d_pct": round((e50_now / e50_20 - 1) * 100, 2),
        },
    )
