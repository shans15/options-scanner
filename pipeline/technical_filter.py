from __future__ import annotations
import logging

from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.base import DataSource
from domain.technical_signals import TechnicalSetup, detect_setups


log = logging.getLogger(__name__)

_OHLCV_LOOKBACK_DAYS = 365


def filter_by_technicals(
    tickers: list[str],
    sources: list[DataSource],
) -> dict[str, list[TechnicalSetup]]:
    """For each ticker, fetch OHLCV history and run detect_setups.
    Returns dict containing only tickers with at least one setup matched.
    Tickers whose fetch raises are logged and dropped."""
    results: dict[str, list[TechnicalSetup]] = {}
    for ticker in tickers:
        try:
            history = fetch_with_fallback(
                sources, 'fetch_price_history_ohlcv', ticker, _OHLCV_LOOKBACK_DAYS,
            )
        except DataFetchError as e:
            log.info("technical_filter: dropping %s — fetch failed (%s)", ticker, e)
            continue
        setups = detect_setups(history)
        if setups:
            results[ticker] = setups
    return results
