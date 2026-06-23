from __future__ import annotations
import pandas as pd
from typing import Optional
from domain.stock_scanner.types import StockSetup


_EMA_PERIODS = [10, 20, 50, 200]


def detect(ticker: str, history: pd.DataFrame) -> Optional[StockSetup]:
    """Detect bullish/bearish EMA stack on daily OHLCV.

    Bullish stack: price > EMA10 > EMA20 > EMA50 > EMA200 (stacked ascending)
    Bearish stack: price < EMA10 < EMA20 < EMA50 < EMA200 (stacked descending)

    Strength = 0.5 * spread_score + 0.5 * persistence_score
      spread_score    = min(|EMA10/EMA200 - 1| / 0.10, 1.0)    (10% spread = full score)
      persistence_score = (bars in last 20 with intact stack) / 20
    """
    if len(history) < 200:
        return None

    closes = history["close"]
    emas = {p: closes.ewm(span=p, adjust=False).mean() for p in _EMA_PERIODS}
    last_close = float(closes.iloc[-1])
    last_emas = {p: float(emas[p].iloc[-1]) for p in _EMA_PERIODS}

    bullish_stacked = (last_close > last_emas[10] > last_emas[20] > last_emas[50] > last_emas[200])
    bearish_stacked = (last_close < last_emas[10] < last_emas[20] < last_emas[50] < last_emas[200])

    if not (bullish_stacked or bearish_stacked):
        return None
    direction = "bullish" if bullish_stacked else "bearish"

    # Spread score
    spread_pct = abs(last_emas[10] / last_emas[200] - 1)
    spread_score = min(spread_pct / 0.10, 1.0)

    # Persistence score: how many of last 20 bars had the stack intact?
    persistence_count = 0
    for i in range(-20, 0):
        c = float(closes.iloc[i])
        e10, e20, e50, e200 = (float(emas[p].iloc[i]) for p in _EMA_PERIODS)
        if direction == "bullish" and c > e10 > e20 > e50 > e200:
            persistence_count += 1
        elif direction == "bearish" and c < e10 < e20 < e50 < e200:
            persistence_count += 1
    persistence_score = persistence_count / 20.0

    strength = 0.5 * spread_score + 0.5 * persistence_score
    if strength < 0.5:
        return None

    return StockSetup(
        ticker=ticker,
        setup_type="sma_stack",
        direction=direction,
        strength=round(strength, 3),
        spot=last_close,
        notes={
            "ema10": round(last_emas[10], 2),
            "ema20": round(last_emas[20], 2),
            "ema50": round(last_emas[50], 2),
            "ema200": round(last_emas[200], 2),
            "spread_pct": round(spread_pct * 100, 2),
            "persistence_days_of_20": persistence_count,
        },
    )
