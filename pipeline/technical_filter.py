from __future__ import annotations
import logging
from datetime import date
from typing import Optional

from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.base import DataSource
from domain.technical_signals import TechnicalSetup, detect_setups


log = logging.getLogger(__name__)

_OHLCV_LOOKBACK_DAYS = 365


def filter_by_technicals(
    tickers: list[str],
    sources: list[DataSource],
    as_of: Optional[date] = None,
) -> dict[str, list[TechnicalSetup]]:
    """For each ticker, fetch OHLCV history and run detect_setups.
    If as_of is provided, slice history to bars on or before that date so the
    detectors see only the EOD snapshot that would have been visible then.
    Returns dict containing only tickers with at least one setup matched."""
    results: dict[str, list[TechnicalSetup]] = {}
    for ticker in tickers:
        try:
            history = fetch_with_fallback(
                sources, 'fetch_price_history_ohlcv', ticker, _OHLCV_LOOKBACK_DAYS,
            )
        except DataFetchError as e:
            log.info("technical_filter: dropping %s — fetch failed (%s)", ticker, e)
            continue
        if as_of is not None and not history.empty:
            if hasattr(history.index, 'date'):
                history = history[history.index.date <= as_of]
            else:
                log.warning(
                    "technical_filter: %s history has non-DatetimeIndex; "
                    "skipping as_of slice", ticker,
                )
        setups = detect_setups(history)
        if setups:
            results[ticker] = setups
    return results
