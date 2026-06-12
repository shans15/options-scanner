from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

_HL_API = "https://api.hyperliquid.xyz/info"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "crypto"

logger = logging.getLogger(__name__)


class HyperliquidSource:
    """Hyperliquid public API client for crypto OHLCV and perpetual-futures funding rates.
    No authentication required for historical public data."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _post(self, body: dict) -> list | dict:
        """POST with one retry on transient error."""
        for attempt in range(2):
            try:
                resp = requests.post(_HL_API, json=body, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    logger.warning("Hyperliquid request failed (%s), retrying once…", exc)
                    time.sleep(1)
                else:
                    raise
        return []

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def fetch_ohlcv(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch perp OHLCV from Hyperliquid.

        symbol: ticker like 'BTC' (Hyperliquid uses plain ticker, not 'BTCUSDT')
        interval: '1m', '5m', '15m', '1h', '4h', '1d'
        Returns DataFrame indexed by UTC datetime with columns ['open','high','low','close','volume'].

        Hyperliquid's /info candleSnapshot returns up to 5000 candles per call. Paginate.
        Each candle: {"t": open_time_ms, "T": close_time_ms, "s": "BTC", "i": "15m",
                      "o": "open_str", "c": "close_str", "h": "high_str", "l": "low_str",
                      "v": "volume_str", "n": num_trades}
        """
        cache_key = f"hl_ohlcv_{symbol}_{interval}_{start_ms}_{end_ms}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading OHLCV from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        rows: list[dict] = []
        current_start = start_ms
        page_limit_ms = self._interval_ms(interval) * 5000  # Hyperliquid max 5000 candles

        while current_start < end_ms:
            page_end = min(current_start + page_limit_ms, end_ms)
            body = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": page_end,
                },
            }
            batch = self._post(body)
            if not batch:
                break
            rows.extend(batch)
            last_close_time = int(batch[-1]["T"])
            current_start = last_close_time + 1
            if len(batch) < 5000 and current_start >= end_ms - self._interval_ms(interval):
                break
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("t")
        df.index.name = "datetime"
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        df = df[numeric_cols].sort_index()
        # Hyperliquid sometimes returns duplicates at page boundaries
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache_file)
        logger.info("Cached OHLCV to %s (%d rows)", cache_file, len(df))
        return df

    def fetch_perp_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch perpetual-futures funding rate history from Hyperliquid.

        symbol: ticker like 'BTC'
        Returns DataFrame indexed by UTC datetime (funding settlement time)
        with columns ['funding_rate', 'mark_price'].

        Hyperliquid funding settles HOURLY (more granular than Binance's 8h).
        Each entry: {"coin": "BTC", "fundingRate": "0.0000125", "premium": "...",
                     "time": ms}  (premium is the basis above oracle price)
        Note: Hyperliquid doesn't return mark_price in this endpoint, so we'll use
        the close price of the corresponding hour as a proxy and join it post-fetch.
        For Phase 1 we just return funding_rate; mark_price column is filled with NaN
        and the validation script doesn't currently rely on it.
        """
        cache_key = f"hl_funding_{symbol}_{start_ms}_{end_ms}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading funding from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        rows: list[dict] = []
        current_start = start_ms
        # Hyperliquid funding endpoint returns up to 500 entries per call (hourly = ~21 days)
        page_limit_ms = 500 * 60 * 60 * 1000

        while current_start < end_ms:
            page_end = min(current_start + page_limit_ms, end_ms)
            body = {
                "type": "fundingHistory",
                "coin": symbol,
                "startTime": current_start,
                "endTime": page_end,
            }
            batch = self._post(body)
            if not batch:
                break
            rows.extend(batch)
            last_time = int(batch[-1]["time"])
            current_start = last_time + 1
            if len(batch) < 500 and current_start >= end_ms - 60 * 60 * 1000:
                break
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame(columns=["funding_rate", "mark_price"])

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("time")
        df.index.name = "datetime"
        df = df.rename(columns={"fundingRate": "funding_rate"})
        df["funding_rate"] = df["funding_rate"].astype(float)
        df["mark_price"] = pd.NA  # not returned by this endpoint
        df = df[["funding_rate", "mark_price"]].sort_index()
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache_file)
        logger.info("Cached funding to %s (%d rows)", cache_file, len(df))
        return df

    @staticmethod
    def _interval_ms(interval: str) -> int:
        units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        n = int(interval[:-1])
        return n * units[interval[-1]]
