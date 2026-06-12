from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

_DERIBIT_API = "https://www.deribit.com/api/v2/public"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "crypto"

logger = logging.getLogger(__name__)


class DeribitSource:
    """Deribit public API for BTC/ETH implied volatility indices.
    No authentication required for historical public data."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """GET with one retry on transient error."""
        url = f"{_DERIBIT_API}/{endpoint}"
        for attempt in range(2):
            try:
                resp = requests.get(url, params=params or {}, timeout=20)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    logger.warning("Deribit request failed (%s), retrying once…", exc)
                    time.sleep(2)
                else:
                    raise
        return {}

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def fetch_historical_volatility(self, currency: str = "BTC") -> pd.DataFrame:
        """Fetch realized volatility history from Deribit.

        Uses /api/v2/public/get_historical_volatility.
        Returns DataFrame indexed by UTC datetime with column ['hv_30d'].
        Resolution: ~1-day, going back ~12 months.
        Cached to Parquet.

        Note: Deribit returns *realized* (historical) volatility here, not implied.
        The endpoint name is slightly misleading — it is 30-day realized vol.
        Implied volatility (DVOL index) requires WebSocket subscription; we'll add
        that in Phase 2.

        Response format:
            {"jsonrpc": "2.0", "result": [[timestamp_ms, hv_value], ...]}
        """
        cache_file = self._cache_path(f"deribit_hv_{currency.upper()}")
        if cache_file.exists():
            logger.info("Loading Deribit HV from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        data = self._get(
            "get_historical_volatility",
            params={"currency": currency.upper()},
        )
        result = data.get("result", [])
        if not result:
            raise ValueError(
                f"Deribit get_historical_volatility returned no data for {currency}. "
                "Check the currency argument (use 'BTC' or 'ETH')."
            )

        # result is a list of [timestamp_ms, hv_value]
        df = pd.DataFrame(result, columns=["ts_ms", "hv_30d"])
        df["datetime"] = pd.to_datetime(df["ts_ms"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("datetime")[["hv_30d"]].astype(float)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache_file)
        logger.info(
            "Cached Deribit HV (%s) to %s (%d rows)", currency, cache_file, len(df)
        )
        return df
