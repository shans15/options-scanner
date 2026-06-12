from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

_CG_API = "https://api.coingecko.com/api/v3"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "crypto"

logger = logging.getLogger(__name__)


class CoinGeckoSource:
    """CoinGecko free public API. No auth required.
    Free tier: ~10-30 calls/min, history up to 365 days for free tier.

    BTC dominance history is fetched via /coins/bitcoin/market_chart and
    /global/market_cap_chart. If the global endpoint requires Pro, we fall
    back to computing dominance from BTC market cap vs. total market cap
    fetched via /global (current snapshot repeated). If all else fails the
    caller should catch the exception and skip dominance features.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        """GET with one retry on transient error."""
        for attempt in range(2):
            try:
                resp = requests.get(url, params=params or {}, timeout=20)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    logger.warning("CoinGecko request failed (%s), retrying once…", exc)
                    time.sleep(2)
                else:
                    raise
        return {}

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def fetch_btc_dominance_history(self, days: int = 365) -> pd.DataFrame:
        """Fetch BTC dominance % history.

        Returns DataFrame indexed by UTC datetime with column ['btc_dominance'].
        Daily resolution. Cached to Parquet.

        Strategy:
        1. Try /global/market_cap_chart (may require Pro tier).
        2. Fall back to computing BTC market cap / total via /coins/bitcoin/market_chart
           and /coins/tether/market_chart etc — too complicated for Phase 1, so instead
           we use /global snapshot to get current dominance and forward-fill (degraded).
        3. Actually for Phase 1 we fetch BTC market_caps and total_market_caps from
           /coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily
           and assume total ≈ BTC / (current dominance fraction). That's circular.
           Best free approach: try the global chart endpoint and fail gracefully.
        """
        cache_file = self._cache_path(f"cg_btc_dominance_{days}d")
        if cache_file.exists():
            logger.info("Loading BTC dominance from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        # Try the global market cap chart endpoint first (may require Pro)
        try:
            data = self._get(
                f"{_CG_API}/global/market_cap_chart",
                params={"days": days, "vs_currency": "usd"},
            )
            # Response: {"market_cap_chart": {"market_cap": [[ts_ms, val], ...], ...}}
            # We need BTC dominance = btc_market_cap / total_market_cap * 100
            chart = data.get("market_cap_chart", {})
            total_mc = chart.get("market_cap", [])
            if not total_mc:
                raise ValueError("Empty market_cap in /global/market_cap_chart response")

            # Fetch BTC market cap separately
            time.sleep(0.5)
            btc_data = self._get(
                f"{_CG_API}/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": days, "interval": "daily"},
            )
            btc_mc = btc_data.get("market_caps", [])
            if not btc_mc:
                raise ValueError("Empty market_caps in /coins/bitcoin/market_chart")

            df_total = pd.DataFrame(total_mc, columns=["ts_ms", "total_market_cap_usd"])
            df_total["datetime"] = pd.to_datetime(df_total["ts_ms"], unit="ms", utc=True)
            df_total = df_total.set_index("datetime")[["total_market_cap_usd"]]

            df_btc = pd.DataFrame(btc_mc, columns=["ts_ms", "btc_market_cap_usd"])
            df_btc["datetime"] = pd.to_datetime(df_btc["ts_ms"], unit="ms", utc=True)
            df_btc = df_btc.set_index("datetime")[["btc_market_cap_usd"]]

            df = df_btc.join(df_total, how="inner")
            df["btc_dominance"] = df["btc_market_cap_usd"] / df["total_market_cap_usd"] * 100.0
            df = df[["btc_dominance"]].sort_index()
            df = df[~df.index.duplicated(keep="first")]

            df.to_parquet(cache_file)
            logger.info("Cached BTC dominance to %s (%d rows)", cache_file, len(df))
            return df

        except Exception as exc:
            logger.warning("Global market_cap_chart failed (%s); trying BTC-only fallback…", exc)

        # Fallback: use /global snapshot for current dominance, build a constant series
        # over the requested date range. This is a degraded approximation.
        global_data = self._get(f"{_CG_API}/global")
        dominance = (
            global_data.get("data", {})
            .get("market_cap_percentage", {})
            .get("btc", None)
        )
        if dominance is None:
            raise RuntimeError(
                "CoinGecko /global endpoint returned no BTC dominance. "
                "Cannot build btc_dominance feature."
            )

        # Build a constant daily series going back `days` from now
        idx = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").normalize(),
            periods=days,
            freq="D",
            tz="UTC",
        )
        df = pd.DataFrame({"btc_dominance": dominance}, index=idx)
        df.index.name = "datetime"
        df.to_parquet(cache_file)
        logger.info(
            "Cached BTC dominance (constant fallback %.2f%%) to %s (%d rows)",
            dominance, cache_file, len(df),
        )
        return df

    def fetch_total_market_cap_history(self, days: int = 365) -> pd.DataFrame:
        """Fetch total crypto market cap history.

        Returns DataFrame indexed by UTC datetime with column ['total_market_cap_usd'].
        Daily resolution. Cached.
        """
        cache_file = self._cache_path(f"cg_total_market_cap_{days}d")
        if cache_file.exists():
            logger.info("Loading total market cap from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        data = self._get(
            f"{_CG_API}/global/market_cap_chart",
            params={"days": days, "vs_currency": "usd"},
        )
        chart = data.get("market_cap_chart", {})
        total_mc = chart.get("market_cap", [])
        if not total_mc:
            raise ValueError("Empty market_cap in /global/market_cap_chart response")

        df = pd.DataFrame(total_mc, columns=["ts_ms", "total_market_cap_usd"])
        df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.set_index("datetime")[["total_market_cap_usd"]].sort_index()
        df = df[~df.index.duplicated(keep="first")]

        df.to_parquet(cache_file)
        logger.info("Cached total market cap to %s (%d rows)", cache_file, len(df))
        return df
