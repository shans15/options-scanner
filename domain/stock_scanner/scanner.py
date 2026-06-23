"""Top-level orchestrator — fetches OHLCV, runs all detectors, scores candidates."""
from __future__ import annotations
import logging
from typing import Optional
import pandas as pd

from data.sources.base import DataSource
from data.fallback import fetch_with_fallback, DataFetchError
from domain.stock_scanner.detectors import sma_stack, compression, proximity_52w, volume_surge
from domain.stock_scanner.scoring import score_candidate
from domain.stock_scanner.types import StockCandidate

log = logging.getLogger(__name__)
_DETECTORS = [sma_stack, compression, proximity_52w, volume_surge]


def scan(universe: list[str], sources: list[DataSource], lookback_days: int = 365) -> list[StockCandidate]:
    """Run all detectors against the universe. Return ranked candidates."""
    candidates: list[StockCandidate] = []
    for ticker in universe:
        try:
            history = fetch_with_fallback(sources, "fetch_price_history_ohlcv", ticker, lookback_days)
        except DataFetchError as e:
            log.warning("Skip %s: %s", ticker, e)
            continue
        if history is None or history.empty or len(history) < 60:
            continue

        setups = []
        for det in _DETECTORS:
            try:
                s = det.detect(ticker, history)
            except Exception as e:
                log.warning("Detector %s failed on %s: %s", det.__name__, ticker, e)
                continue
            if s is not None:
                setups.append(s)

        if not setups:
            continue

        spot = float(history["close"].iloc[-1])
        snapshot = {
            "close": round(spot, 2),
            "volume_5d_avg": int(history["volume"].iloc[-5:].mean()),
            "high_52w": round(float(history["high"].iloc[-252:].max() if len(history) >= 252 else history["high"].max()), 2),
            "low_52w": round(float(history["low"].iloc[-252:].min() if len(history) >= 252 else history["low"].min()), 2),
        }
        candidates.append(score_candidate(ticker, spot, setups, snapshot))

    candidates.sort(key=lambda c: -abs(c.net_score))
    return candidates
