from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

_BINANCE_FAPI = "https://fapi.binance.com"
_BINANCE_API = "https://api.binance.com"
_CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "crypto"

logger = logging.getLogger(__name__)


class BinanceSource:
    """Public Binance API client for crypto OHLCV and perpetual-futures funding rates.
    No authentication required for historical public data."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict) -> list:
        """GET with one retry on transient error."""
        for attempt in range(2):
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    logger.warning("Binance request failed (%s), retrying once…", exc)
                    time.sleep(1)
                else:
                    raise
        return []  # unreachable but satisfies type checkers

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        """Fetch spot OHLCV from Binance.

        Returns a DataFrame indexed by UTC datetime with columns
        ['open', 'high', 'low', 'close', 'volume'].

        Paginates automatically (max 1000 klines per call). Results are cached
        to Parquet under cache/crypto/ and returned from cache if available.
        """
        cache_key = f"ohlcv_{symbol}_{interval}_{start_ms}_{end_ms}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading OHLCV from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        url = f"{_BINANCE_API}/api/v3/klines"
        rows: list[list] = []
        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            batch = self._get(url, params)
            if not batch:
                break
            rows.extend(batch)
            # Each kline: [open_time, open, high, low, close, volume, close_time, ...]
            last_close_time = int(batch[-1][6])
            current_start = last_close_time + 1
            if len(batch) < 1000:
                break
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("open_time")
        df.index.name = "datetime"
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df = df[numeric_cols].astype(float)
        df = df.sort_index()

        df.to_parquet(cache_file)
        logger.info("Cached OHLCV to %s (%d rows)", cache_file, len(df))
        return df

    def fetch_perp_funding(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        """Fetch perpetual-futures funding rate history from Binance.

        Returns a DataFrame indexed by UTC datetime (funding settlement time)
        with columns ['funding_rate', 'mark_price'].

        Paginates automatically (max 1000 entries per call). Results are cached.
        """
        cache_key = f"funding_{symbol}_{start_ms}_{end_ms}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading funding from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        url = f"{_BINANCE_FAPI}/fapi/v1/fundingRate"
        rows: list[dict] = []
        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }
            batch = self._get(url, params)
            if not batch:
                break
            rows.extend(batch)
            # Each entry: {"symbol":…, "fundingTime": ms, "fundingRate": str, "markPrice": str}
            last_time = int(batch[-1]["fundingTime"])
            current_start = last_time + 1
            if len(batch) < 1000:
                break
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame(columns=["funding_rate", "mark_price"])

        df = pd.DataFrame(rows)
        df["fundingTime"] = pd.to_datetime(
            df["fundingTime"].astype("int64"), unit="ms", utc=True
        )
        df = df.set_index("fundingTime")
        df.index.name = "datetime"
        df = df.rename(columns={"fundingRate": "funding_rate", "markPrice": "mark_price"})
        df["funding_rate"] = df["funding_rate"].astype(float)
        df["mark_price"] = df["mark_price"].astype(float)
        df = df[["funding_rate", "mark_price"]].sort_index()

        df.to_parquet(cache_file)
        logger.info("Cached funding to %s (%d rows)", cache_file, len(df))
        return df
