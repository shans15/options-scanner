from __future__ import annotations
import pandas as pd
from typing import Optional
from domain.stock_scanner.types import StockSetup


def detect(ticker: str, history: pd.DataFrame) -> Optional[StockSetup]:
    """Detect volume surge in last 5 days vs 20-day average.

    Detected when: avg(volume, 5) / avg(volume, 20) >= 1.5

    Direction from net price action over surge period:
      close[-1] > close[-6] -> bullish accumulation
      close[-1] < close[-6] -> bearish distribution

    Strength = clamp((surge_ratio - 1.0) / 1.5, 0, 1)
    """
    if len(history) < 25:
        return None
    volume = history["volume"]
    close = history["close"]
    avg5 = float(volume.iloc[-5:].mean())
    avg20 = float(volume.iloc[-20:].mean())
    if avg20 <= 0:
        return None
    surge_ratio = avg5 / avg20
    if surge_ratio < 1.5:
        return None

    last_close = float(close.iloc[-1])
    close_5d_ago = float(close.iloc[-6])
    net_change = (last_close - close_5d_ago) / close_5d_ago
    direction = "bullish" if net_change > 0 else "bearish"

    strength = min((surge_ratio - 1.0) / 1.5, 1.0)
    if strength < 0.3:
        return None

    return StockSetup(
        ticker=ticker,
        setup_type="volume_surge",
        direction=direction,
        strength=round(strength, 3),
        spot=last_close,
        notes={
            "surge_ratio": round(surge_ratio, 2),
            "avg_5d_volume": int(avg5),
            "avg_20d_volume": int(avg20),
            "5d_price_change_pct": round(net_change * 100, 2),
        },
    )
