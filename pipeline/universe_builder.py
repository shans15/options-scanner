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


SP500_FILE = Path('data/universe/sp500_constituents.json')


@dataclass(frozen=True)
class UniverseFilters:
    min_avg_volume: int = 1_000_000
    require_options_chain: bool = True
    min_iv_rank: float = 0.0
    top_n: int = 50


def fetch_sp500_constituents() -> list[str]:
    """Read the committed constituent snapshot.  Refresh via
    ``python -m scripts.refresh_sp500`` (intended cadence: monthly)."""
    if not SP500_FILE.exists():
        raise FileNotFoundError(
            f"{SP500_FILE} missing — run `python -m scripts.refresh_sp500`."
        )
    return json.loads(SP500_FILE.read_text())


def _avg_volume_30d(history: pd.Series) -> float:
    tail = history.dropna().tail(30)
    return float(tail.mean()) if len(tail) else 0.0


def build_universe(filters: UniverseFilters, sources: list[DataSource]) -> list[str]:
    """Return the committed S&P 500 list capped to ``top_n``.

    The committed snapshot is treated as pre-vetted — no live volume or chain
    pre-filter.  The per-ticker scan loop already skips names with missing
    data, and free chain sources (yahooquery/yfinance) return spuriously empty
    chains often enough that pre-filtering hides real candidates.
    """
    try:
        candidates = fetch_sp500_constituents()
    except Exception as e:
        log.warning("Local constituent file unavailable (%s); using KNOWN_GOOD_FALLBACK", e)
        return list(KNOWN_GOOD_FALLBACK[: filters.top_n])
    return candidates[: filters.top_n]


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
