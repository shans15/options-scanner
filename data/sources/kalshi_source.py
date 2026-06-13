from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

# Kalshi unified everything to api.elections.kalshi.com in 2026; the old
# trading-api.kalshi.com now returns a redirect notice. The same host serves
# political markets unauthenticated AND financial markets (BTC, etc.) when
# requests are signed.
_KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
# Alias kept for backwards compatibility with tests.
_KALSHI_PUBLIC_API = _KALSHI_API

_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "kalshi"

logger = logging.getLogger(__name__)


class KalshiSource:
    """Kalshi REST API client for prediction-market data.

    Uses the authenticated trading-api endpoint (trading-api.kalshi.com) which
    supports BTC hourly markets.  When KALSHI_ACCESS_KEY_ID and
    KALSHI_PRIVATE_KEY_PATH are present in the environment a KalshiSigner is
    auto-constructed; otherwise only public (political) markets will work.
    """

    _PAGE_LIMIT = 200  # Kalshi max per page

    def __init__(self, cache_dir: Path | None = None, signer=None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if signer is None:
            try:
                from data.sources.kalshi_auth import KalshiAuthError, KalshiSigner

                signer = KalshiSigner()
            except Exception:
                # KalshiAuthError or ImportError — run unauthenticated
                signer = None
                logger.warning(
                    "Kalshi auth not configured; only public endpoints will work."
                )
        self._signer = signer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Signed (if signer available) HTTP request with one retry.

        path: bare path segment such as 'markets' or 'markets/{ticker}/orderbook'.
              It is joined to the base URL with a '/'.
        """
        url = f"{_KALSHI_API}/{path}"
        headers: dict[str, str] = {"Accept": "application/json"}

        if self._signer:
            # Build the full path (with leading /) including query string for signing
            api_path = f"/trade-api/v2/{path}"
            if params:
                api_path = f"{api_path}?{urlencode(params)}"
            headers.update(self._signer.sign_headers(method, api_path))

        for attempt in range(2):
            try:
                resp = requests.request(
                    method, url, params=params, headers=headers, timeout=15
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    logger.warning("Kalshi request failed (%s), retrying once…", exc)
                    time.sleep(1)
                else:
                    raise
        return {}

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Convenience wrapper around _request for GET calls."""
        return self._request("GET", endpoint, params)

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def _cache_json_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_btc_markets(self, status: str = "open") -> list[dict]:
        """List BTC-related markets. status in {'open', 'closed', 'settled'}.

        Returns the raw market dicts from /markets endpoint, filtered to ones
        with 'BTC' in event_ticker or title.

        Paginates via 'cursor' field; max ~200 per page.
        Results are cached to Parquet under cache/kalshi/.
        """
        cache_key = f"kalshi_btc_markets_{status}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading BTC markets from cache: %s", cache_file)
            df = pd.read_parquet(cache_file)
            return df.to_dict(orient="records")

        markets: list[dict] = []
        cursor: str | None = None

        while True:
            params: dict = {
                "series_ticker": "KXBTC",
                "status": status,
                "limit": self._PAGE_LIMIT,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._get("markets", params)
            page_markets = data.get("markets", [])

            # Filter to BTC-related markets
            for m in page_markets:
                event_ticker = m.get("event_ticker", "")
                title = m.get("title", "")
                if "BTC" in event_ticker.upper() or "BTC" in title.upper():
                    markets.append(m)

            cursor = data.get("cursor")
            if not cursor or len(page_markets) < self._PAGE_LIMIT:
                break

            time.sleep(0.1)

        if markets:
            df = pd.DataFrame(markets)
            df.to_parquet(cache_file)
            logger.info(
                "Cached %d BTC markets (status=%s) to %s", len(markets), status, cache_file
            )

        return markets

    def get_market_orderbook(self, ticker: str) -> dict:
        """Get current orderbook for a market.

        Returns raw dict with 'yes' and 'no' sides.
        Each side is a list of [price_cents, quantity] pairs.
        """
        data = self._get(f"markets/{ticker}/orderbook")
        return data.get("orderbook", data)

    def get_market_history(self, ticker: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Get price history for a market.

        Returns DataFrame indexed by UTC datetime with columns
        ['yes_bid', 'yes_ask', 'no_bid', 'no_ask', 'volume'].

        Cached to Parquet under cache/kalshi/.
        """
        cache_key = f"kalshi_hist_{ticker}_{start_ms}_{end_ms}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading market history from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        start_s = start_ms // 1000
        end_s = end_ms // 1000
        params = {"start_ts": start_s, "end_ts": end_s}
        data = self._get(f"markets/{ticker}/history", params)

        history = data.get("history", [])
        if not history:
            return pd.DataFrame(columns=["yes_bid", "yes_ask", "no_bid", "no_ask", "volume"])

        rows = []
        for entry in history:
            rows.append({
                "ts": entry.get("ts") or entry.get("timestamp"),
                "yes_bid": entry.get("yes_bid", float("nan")),
                "yes_ask": entry.get("yes_ask", float("nan")),
                "no_bid": entry.get("no_bid", float("nan")),
                "no_ask": entry.get("no_ask", float("nan")),
                "volume": entry.get("volume", 0),
            })

        df = pd.DataFrame(rows)
        # ts may be seconds or milliseconds — normalise to ms
        ts_col = df["ts"].astype("int64")
        # If max value < 1e12, it's in seconds
        if ts_col.max() < 1_000_000_000_000:
            ts_col = ts_col * 1000
        df["datetime"] = pd.to_datetime(ts_col, unit="ms", utc=True)
        df = df.set_index("datetime")[["yes_bid", "yes_ask", "no_bid", "no_ask", "volume"]]
        df = df.astype(float)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache_file)
        logger.info(
            "Cached market history for %s to %s (%d rows)", ticker, cache_file, len(df)
        )
        return df

    def get_settled_markets(self, start_ms: int, end_ms: int) -> list[dict]:
        """Get BTC markets that resolved in a date range. Used for backtest.

        Returns markets with their resolution outcome ('yes'/'no').
        Filters by min_close_ts / max_close_ts.
        Paginates and caches results.
        """
        start_s = start_ms // 1000
        end_s = end_ms // 1000
        cache_key = f"kalshi_settled_{start_s}_{end_s}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading settled markets from cache: %s", cache_file)
            df = pd.read_parquet(cache_file)
            return df.to_dict(orient="records")

        markets: list[dict] = []
        cursor: str | None = None

        while True:
            params: dict = {
                "series_ticker": "KXBTC",
                "status": "settled",
                "min_close_ts": start_s,
                "max_close_ts": end_s,
                "limit": self._PAGE_LIMIT,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._get("markets", params)
            page_markets = data.get("markets", [])

            for m in page_markets:
                event_ticker = m.get("event_ticker", "")
                title = m.get("title", "")
                if "BTC" in event_ticker.upper() or "BTC" in title.upper():
                    markets.append(m)

            cursor = data.get("cursor")
            if not cursor or len(page_markets) < self._PAGE_LIMIT:
                break

            time.sleep(0.1)

        if markets:
            df = pd.DataFrame(markets)
            df.to_parquet(cache_file)
            logger.info(
                "Cached %d settled BTC markets to %s", len(markets), cache_file
            )

        return markets
