from __future__ import annotations
import pandas as pd
from typing import Optional
from domain.stock_scanner.types import StockSetup


_LOOKBACK = 252


def detect(ticker: str, history: pd.DataFrame) -> Optional[StockSetup]:
    """Detect proximity to 52-week high (momentum) or 52-week low (reversal).

    Within 5% of 52w high -> bullish momentum
    Within 5% of 52w low  -> bearish (continuation) OR pending reversal

    Strength = 1 - (distance / 0.05)   capped at 1.0
    """
    if len(history) < 60:
        return None
    recent = history.iloc[-_LOOKBACK:] if len(history) >= _LOOKBACK else history
    high_52w = float(recent["high"].max())
    low_52w = float(recent["low"].min())
    close = float(history["close"].iloc[-1])
    if high_52w <= 0 or close <= 0:
        return None

    dist_to_high = (high_52w - close) / close
    dist_to_low = (close - low_52w) / close

    if dist_to_high <= 0.05:
        direction = "bullish"
        distance_used = dist_to_high
    elif dist_to_low <= 0.05:
        direction = "bearish"
        distance_used = dist_to_low
    else:
        return None

    strength = max(0.0, 1.0 - distance_used / 0.05)
    if strength < 0.3:
        return None

    return StockSetup(
        ticker=ticker,
        setup_type="proximity_52w",
        direction=direction,
        strength=round(strength, 3),
        spot=close,
        notes={
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "dist_to_high_pct": round(dist_to_high * 100, 2),
            "dist_to_low_pct": round(dist_to_low * 100, 2),
        },
    )
