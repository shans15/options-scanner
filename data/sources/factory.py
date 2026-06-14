"""Factory for constructing the standard DataSource fallback chain.

Centralises the logic so every call site (run_scan, main, equity proxies)
uses the same priority order.

Order:
    1. SchwabSource  — primary; live-ish data with penny-tight spreads.
       Skipped if Schwab auth is not configured (no token file or env vars).
    2. YahooQuerySource — fallback #1; 15-min delayed but no auth needed.
    3. YfinanceSource  — fallback #2; different scrape path.
    4. StooqSource     — fallback²; OHLCV only, no options.
"""

from __future__ import annotations

import logging
from typing import Optional

from data.sources.base import DataSource
from data.sources.yahooquery_source import YahooQuerySource
from data.sources.yfinance_source import YfinanceSource
from data.sources.stooq_source import StooqSource

log = logging.getLogger(__name__)


def default_sources(prefer_schwab: bool = True) -> list[DataSource]:
    """Return the standard fallback chain.

    If `prefer_schwab=True` (the default), attempts to construct a SchwabSource
    using credentials from env. If auth is not configured or the token is
    missing/expired, silently falls back to the no-auth chain.
    """
    sources: list[DataSource] = []
    if prefer_schwab:
        try:
            from data.sources.schwab_source import SchwabSource

            sources.append(SchwabSource())
            log.info("SchwabSource enabled as primary data source.")
        except Exception as exc:
            log.warning(
                "Schwab auth not available (%s); falling back to yahooquery/yfinance/stooq.",
                exc,
            )
    sources.extend([YahooQuerySource(), YfinanceSource(), StooqSource()])
    return sources
