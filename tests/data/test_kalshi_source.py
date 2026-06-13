from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from data.sources.kalshi_source import KalshiSource, _KALSHI_API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _market(ticker: str = "KXBTC-26JUN1316-T62500", status: str = "open") -> dict:
    """Minimal Kalshi market dict."""
    return {
        "ticker": ticker,
        "event_ticker": "KXBTC-26JUN1316",
        "title": "BTC > $62,500 at 4:00 PM ET",
        "status": status,
        "yes_bid": 8,
        "yes_ask": 10,
        "no_bid": 90,
        "no_ask": 92,
        "volume": 1240,
        "open_interest": 500,
        "close_time": "2026-06-13T20:00:00Z",
        "expiration_time": "2026-06-13T20:00:00Z",
    }


def _markets_url() -> str:
    return f"{_KALSHI_API}/markets"


def _orderbook_url(ticker: str) -> str:
    return f"{_KALSHI_API}/markets/{ticker}/orderbook"


def _history_url(ticker: str) -> str:
    return f"{_KALSHI_API}/markets/{ticker}/history"


# ---------------------------------------------------------------------------
# test_list_btc_markets_parses_response
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_list_btc_markets_parses_response(tmp_path):
    """list_btc_markets must return a list of dicts matching BTC filter."""
    payload = {
        "markets": [
            _market("KXBTC-26JUN1316-T62500"),
            _market("KXBTC-26JUN1316-T63000"),
            # Non-BTC market — should be excluded
            {"ticker": "KXGOLD-26JUN-T2000", "event_ticker": "KXGOLD-26JUN",
             "title": "Gold > $2,000", "status": "open",
             "yes_bid": 50, "yes_ask": 52, "no_bid": 48, "no_ask": 50,
             "volume": 100, "open_interest": 50,
             "close_time": "2026-06-13T20:00:00Z", "expiration_time": "2026-06-13T20:00:00Z"},
        ],
        "cursor": None,
    }
    resp_lib.add(resp_lib.GET, _markets_url(), json=payload, status=200)

    src = KalshiSource(cache_dir=tmp_path)
    markets = src.list_btc_markets(status="open")

    assert len(markets) == 2
    assert all("BTC" in m["event_ticker"].upper() or "BTC" in m["title"].upper() for m in markets)
    assert markets[0]["ticker"] == "KXBTC-26JUN1316-T62500"


# ---------------------------------------------------------------------------
# test_list_btc_markets_paginates
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_list_btc_markets_paginates(tmp_path):
    """list_btc_markets must concatenate results across pages."""
    page1_markets = [_market(f"KXBTC-26JUN1316-T{60000 + i * 1000}") for i in range(200)]
    page2_markets = [_market(f"KXBTC-26JUN1316-T{260000 + i * 1000}") for i in range(5)]

    resp_lib.add(
        resp_lib.GET,
        _markets_url(),
        json={"markets": page1_markets, "cursor": "abc123"},
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        _markets_url(),
        json={"markets": page2_markets, "cursor": None},
        status=200,
    )

    src = KalshiSource(cache_dir=tmp_path)
    markets = src.list_btc_markets(status="open")

    assert len(resp_lib.calls) == 2
    assert len(markets) == 205


# ---------------------------------------------------------------------------
# test_get_market_orderbook_returns_yes_no_sides
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_market_orderbook_returns_yes_no_sides(tmp_path):
    """get_market_orderbook must return a dict with 'yes' and 'no' sides."""
    ticker = "KXBTC-26JUN1316-T62500"
    orderbook_payload = {
        "orderbook": {
            "yes": [[8, 100], [7, 200]],
            "no": [[90, 150], [89, 300]],
        }
    }
    resp_lib.add(resp_lib.GET, _orderbook_url(ticker), json=orderbook_payload, status=200)

    src = KalshiSource(cache_dir=tmp_path)
    ob = src.get_market_orderbook(ticker)

    assert "yes" in ob
    assert "no" in ob
    assert isinstance(ob["yes"], list)
    assert isinstance(ob["no"], list)


# ---------------------------------------------------------------------------
# test_get_market_history_returns_datetime_indexed_df
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_market_history_returns_datetime_indexed_df(tmp_path):
    """get_market_history must return a DataFrame indexed by UTC datetime."""
    ticker = "KXBTC-26JUN1316-T62500"
    t0_s = int(datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc).timestamp())
    t1_s = t0_s + 3600
    history_payload = {
        "history": [
            {"ts": t0_s, "yes_bid": 8, "yes_ask": 10, "no_bid": 90, "no_ask": 92, "volume": 100},
            {"ts": t1_s, "yes_bid": 9, "yes_ask": 11, "no_bid": 89, "no_ask": 91, "volume": 120},
        ]
    }
    resp_lib.add(resp_lib.GET, _history_url(ticker), json=history_payload, status=200)

    start_ms = t0_s * 1000
    end_ms = t1_s * 1000 + 1000

    src = KalshiSource(cache_dir=tmp_path)
    df = src.get_market_history(ticker, start_ms, end_ms)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
    assert list(df.columns) == ["yes_bid", "yes_ask", "no_bid", "no_ask", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# test_get_settled_markets_filters_by_date_range
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_get_settled_markets_filters_by_date_range(tmp_path):
    """get_settled_markets must include the correct date-range params in its request."""
    t0_ms = _ms(datetime(2026, 1, 1, tzinfo=timezone.utc))
    t1_ms = _ms(datetime(2026, 5, 31, tzinfo=timezone.utc))

    settled = [_market(f"KXBTC-26JAN-T{50000 + i * 1000}", status="settled") for i in range(3)]
    resp_lib.add(
        resp_lib.GET,
        _markets_url(),
        json={"markets": settled, "cursor": None},
        status=200,
    )

    src = KalshiSource(cache_dir=tmp_path)
    markets = src.get_settled_markets(t0_ms, t1_ms)

    # Verify the request included the right params
    assert len(resp_lib.calls) == 1
    req = resp_lib.calls[0].request
    assert "min_close_ts" in req.url
    assert "max_close_ts" in req.url
    assert "settled" in req.url
    assert len(markets) == 3


# ---------------------------------------------------------------------------
# test_caches_to_parquet
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_caches_to_parquet(tmp_path):
    """list_btc_markets must write a Parquet file; second call reads cache."""
    payload = {
        "markets": [_market("KXBTC-26JUN1316-T62500")],
        "cursor": None,
    }
    resp_lib.add(resp_lib.GET, _markets_url(), json=payload, status=200)

    src = KalshiSource(cache_dir=tmp_path)
    markets1 = src.list_btc_markets(status="open")

    # Parquet file must exist
    cache_file = tmp_path / "kalshi_btc_markets_open.parquet"
    assert cache_file.exists()

    # Second call — must NOT hit the network
    markets2 = src.list_btc_markets(status="open")
    assert len(resp_lib.calls) == 1  # only one HTTP call total
    assert len(markets1) == len(markets2)
    assert markets1[0]["ticker"] == markets2[0]["ticker"]
