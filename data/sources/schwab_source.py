"""Charles Schwab market-data API source.

Read-only wrapper around Schwab's /marketdata/v1 endpoints, exposing the
same interface other DataSource implementations use (yahooquery, yfinance, stooq).

Requires Schwab developer-portal app credentials in the environment — see
data/sources/schwab_auth.SchwabAuth.

Endpoints used:
  GET /marketdata/v1/{symbol}/quotes        — single-symbol real-time quote
  GET /marketdata/v1/quotes?symbols=…       — batch quotes (more efficient)
  GET /marketdata/v1/pricehistory           — OHLCV history
  GET /marketdata/v1/chains                 — full option chain
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from data.sources.base import DataSource, RawContract
from data.sources.schwab_auth import SchwabAuth, SchwabAuthError


_API_BASE = "https://api.schwabapi.com/marketdata/v1"

logger = logging.getLogger(__name__)


class SchwabSource(DataSource):
    """Schwab market-data client.

    Uses an injected SchwabAuth for token management; pass `auth=None` to
    auto-construct one from environment variables.
    """

    def __init__(self, auth: Optional[SchwabAuth] = None) -> None:
        self._auth = auth if auth is not None else SchwabAuth()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, path: str, params: Optional[dict] = None) -> dict:
        token = self._auth.get_access_token()
        url = f"{_API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 401:
            # Token may be invalid; force a refresh and retry once.
            self._auth._refresh_access_token()
            token = self._auth.get_access_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def fetch_spot(self, ticker: str) -> float:
        """Latest mid quote for a single ticker."""
        data = self._request(f"/{ticker}/quotes")
        quote = data.get(ticker, {}).get("quote", {})
        # Prefer mid of bid/ask; fall back to lastPrice
        bid = quote.get("bidPrice") or quote.get("bid") or 0.0
        ask = quote.get("askPrice") or quote.get("ask") or 0.0
        if bid > 0 and ask > 0:
            return float((bid + ask) / 2)
        return float(quote.get("lastPrice", 0.0) or 0.0)

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        """Daily close series for the past `lookback_days` calendar days."""
        df = self.fetch_price_history_ohlcv(ticker, lookback_days)
        return df["close"] if not df.empty else pd.Series(dtype=float)

    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """OHLCV DataFrame (UTC DatetimeIndex)."""
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
        )
        params = {
            "symbol": ticker,
            "periodType": "year" if lookback_days > 60 else "month",
            "frequencyType": "daily",
            "frequency": 1,
            "startDate": start_ms,
            "endDate": end_ms,
            "needExtendedHoursData": "false",
        }
        data = self._request("/pricehistory", params)
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        return df.astype(float)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        """All option contracts for the underlying, flattened to RawContract."""
        params = {
            "symbol": ticker,
            "contractType": "ALL",
            "strikeCount": 50,
            "includeUnderlyingQuote": "true",
            "range": "ALL",
        }
        data = self._request("/chains", params)
        underlying = data.get("underlying", {}) or {}
        spot = float(underlying.get("last") or underlying.get("mark") or 0.0)
        today = date.today()
        out: list[RawContract] = []

        for opt_type_key, py_type in [("callExpDateMap", "call"), ("putExpDateMap", "put")]:
            exp_map = data.get(opt_type_key, {}) or {}
            for exp_key, strike_map in exp_map.items():
                # exp_key looks like '2026-06-20:7'
                exp_str = exp_key.split(":")[0]
                try:
                    exp_date = date.fromisoformat(exp_str)
                except ValueError:
                    continue
                dte = (exp_date - today).days
                if dte < 0:
                    continue
                for strike_str, contracts in strike_map.items():
                    if not contracts:
                        continue
                    c = contracts[0]
                    try:
                        strike = float(strike_str)
                        bid = float(c.get("bid") or 0.0)
                        ask = float(c.get("ask") or 0.0)
                        vol = int(c.get("totalVolume") or 0)
                        oi = int(c.get("openInterest") or 0)
                        iv_pct = c.get("volatility")
                        # Schwab returns IV in percent (e.g., 22.5 means 22.5%); normalise
                        iv = float(iv_pct) / 100.0 if iv_pct is not None else 0.0
                        out.append(RawContract(
                            ticker=ticker,
                            expiration=exp_date,
                            strike=strike,
                            option_type=py_type,
                            bid=bid,
                            ask=ask,
                            volume=vol,
                            open_interest=oi,
                            implied_volatility=iv,
                            dte=dte,
                            spot_price=spot,
                        ))
                    except (ValueError, TypeError):
                        continue
        return out
