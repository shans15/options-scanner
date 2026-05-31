from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import logging
import pandas as pd

from data.sources.base import DataSource
from data.fallback import fetch_with_fallback, DataFetchError
from pipeline.universe import KNOWN_GOOD_FALLBACK


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseFilters:
    min_avg_volume: int = 1_000_000
    require_options_chain: bool = True
    min_iv_rank: float = 0.0
    top_n: int = 50


def fetch_sp500_constituents() -> list[str]:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    symbols = tables[0]['Symbol'].astype(str).tolist()
    return [s.replace('.', '-') for s in symbols]


def _avg_volume_30d(history: pd.Series) -> float:
    tail = history.dropna().tail(30)
    return float(tail.mean()) if len(tail) else 0.0


def build_universe(filters: UniverseFilters, sources: list[DataSource]) -> list[str]:
    try:
        candidates = fetch_sp500_constituents()
    except Exception as e:
        # Wikipedia scrape failed; live filtering is unreliable in this state.
        # Return the curated KNOWN_GOOD_FALLBACK directly — it is pre-vetted and
        # exists precisely so we can serve a sensible universe without live data.
        log.warning("S&P 500 fetch failed (%s); using KNOWN_GOOD_FALLBACK (no live filtering)", e)
        return list(KNOWN_GOOD_FALLBACK[: filters.top_n])

    qualified: list[tuple[str, float]] = []
    for t in candidates:
        try:
            hist = fetch_with_fallback(sources, 'fetch_price_history', t, 30)
        except DataFetchError:
            continue
        avg_vol = _avg_volume_30d(hist)
        if avg_vol < filters.min_avg_volume:
            continue
        if filters.require_options_chain:
            try:
                chain = fetch_with_fallback(sources[:2], 'fetch_option_chain', t)
            except DataFetchError:
                continue
            if not chain:
                continue
        qualified.append((t, avg_vol))

    qualified.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in qualified[: filters.top_n]]


def build_universe_cached(
    filters: UniverseFilters,
    sources: list[DataSource],
    cache_dir: Path = Path('cache/universe'),
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'universe_{date.today().isoformat()}.json'
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    universe = build_universe(filters, sources)
    cache_file.write_text(json.dumps(universe))
    return universe
