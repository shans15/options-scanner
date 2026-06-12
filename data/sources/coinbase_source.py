from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

_CB_API = "https://api.exchange.coinbase.com"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "crypto"

logger = logging.getLogger(__name__)


class CoinbaseSource:
    """Coinbase Exchange public API client for spot OHLCV data.
    No authentication required for historical public market data."""

    # granularity in seconds
    _INTERVALS = {
        "1m": 60, "5m": 300, "15m": 900,
        "1h": 3600, "6h": 21600, "1d": 86400,
    }
    # Max rows per single API call
    _PAGE_LIMIT = 300

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, params: dict) -> list:
        for attempt in range(2):
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    logger.warning("Coinbase request failed (%s), retrying once…", exc)
                    time.sleep(1)
                else:
                    raise
        return []

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def fetch_ohlcv(self, product_id: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch spot OHLCV from Coinbase Exchange.

        product_id: 'BTC-USD', 'ETH-USD', etc.
        interval: '1m', '5m', '15m', '1h', '6h', '1d'
        Returns DataFrame indexed by UTC datetime with columns
        ['open', 'high', 'low', 'close', 'volume'].

        Coinbase returns at most 300 rows per call. Paginate forward in time.
        Each row from API: [time_seconds, low, high, open, close, volume]
        """
        if interval not in self._INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}. Use one of {list(self._INTERVALS)}")

        cache_key = f"cb_ohlcv_{product_id}_{interval}_{start_ms}_{end_ms}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading OHLCV from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        gran_s = self._INTERVALS[interval]
        page_window_s = self._PAGE_LIMIT * gran_s  # max seconds per call
        url = f"{_CB_API}/products/{product_id}/candles"

        all_rows: list[list] = []
        current_start_s = start_ms // 1000
        end_s = end_ms // 1000

        while current_start_s < end_s:
            page_end_s = min(current_start_s + page_window_s, end_s)
            start_iso = pd.Timestamp(current_start_s, unit="s", tz="UTC").isoformat()
            end_iso = pd.Timestamp(page_end_s, unit="s", tz="UTC").isoformat()
            params = {"granularity": gran_s, "start": start_iso, "end": end_iso}
            batch = self._get(url, params)
            if not batch:
                # Coinbase sometimes returns empty for short ranges; advance to avoid infinite loop
                current_start_s = page_end_s + gran_s
                continue
            all_rows.extend(batch)
            # Coinbase returns DESCENDING by time. Last row = earliest in batch.
            earliest_time_s = int(batch[-1][0])
            latest_time_s = int(batch[0][0])
            current_start_s = latest_time_s + gran_s
            time.sleep(0.2)

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Coinbase: [time_s, low, high, open, close, volume]
        df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
        df = df.set_index("time")
        df.index.name = "datetime"
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache_file)
        logger.info("Cached Coinbase OHLCV to %s (%d rows)", cache_file, len(df))
        return df
