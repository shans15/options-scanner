"""Tests for scripts/kalshi_btc_watchlist.py

Uses mocked KalshiSource and CoinbaseSource to avoid network calls.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 800, base_price: float = 65_000.0) -> pd.DataFrame:
    """Synthetic hourly OHLCV DataFrame of length n, UTC-indexed."""
    now = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
    idx = pd.date_range(end=now, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0, 0.001, size=n)
    close = base_price * np.exp(np.cumsum(log_rets))
    df = pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.001, "low": close * 0.998,
         "close": close, "volume": rng.uniform(100, 500, size=n)},
        index=idx,
    )
    return df


def _make_raw_market(
    ticker: str = "KXBTC-26JUN1316-T62500",
    yes_bid: int = 38,
    yes_ask: int = 42,
    volume: int = 1000,
    expiration: str = "2026-06-13T20:00:00Z",
) -> dict:
    """Minimal raw Kalshi market dict."""
    return {
        "ticker": ticker,
        "event_ticker": "KXBTC-26JUN1316",
        "title": f"BTC > $62,500 at 4:00 PM ET",
        "status": "open",
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 100 - yes_ask,
        "no_ask": 100 - yes_bid,
        "volume": volume,
        "open_interest": 200,
        "close_time": expiration,
        "expiration_time": expiration,
    }


# ---------------------------------------------------------------------------
# test_watchlist_smoke_with_mocked_sources
# ---------------------------------------------------------------------------

def test_watchlist_smoke_with_mocked_sources(tmp_path):
    """main() runs without error and produces latest.json output."""
    ohlcv_30d = _make_ohlcv(n=800)
    ohlcv_1min = _make_ohlcv(n=60, base_price=65_000.0)

    # Market expiring in 2 hours from our frozen "now"
    now_str = "2026-06-13T16:00:00Z"
    future_exp = "2026-06-13T18:00:00Z"

    raw_markets = [_make_raw_market(
        ticker="KXBTC-26JUN1316-T62500",
        yes_bid=8,
        yes_ask=12,
        volume=1000,
        expiration=future_exp,
    )]

    mock_ks = MagicMock()
    mock_ks.list_btc_markets.return_value = raw_markets

    mock_cb = MagicMock()
    mock_cb.fetch_ohlcv.side_effect = lambda product, interval, start, end: (
        ohlcv_30d if interval == "1h" else ohlcv_1min
    )

    output_dir = tmp_path / "output" / "kalshi"

    with patch("scripts.kalshi_btc_watchlist.KalshiSource", return_value=mock_ks), \
         patch("scripts.kalshi_btc_watchlist.CoinbaseSource", return_value=mock_cb), \
         patch("scripts.kalshi_btc_watchlist.OUTPUT_DIR", output_dir), \
         patch("scripts.kalshi_btc_watchlist.datetime") as mock_dt:

        # Freeze "now" so hours_to_expiry > 0
        frozen_now = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = frozen_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        from scripts.kalshi_btc_watchlist import main
        rc = main()

    # Script must exit 0
    assert rc == 0

    # Output JSON must have been written
    out_file = output_dir / "latest.json"
    assert out_file.exists(), "latest.json was not created"

    data = json.loads(out_file.read_text())
    assert "btc_spot" in data
    assert "markets_analyzed" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


# ---------------------------------------------------------------------------
# test_edge_filter_keeps_only_above_threshold
# ---------------------------------------------------------------------------

def test_edge_filter_keeps_only_above_threshold(tmp_path):
    """Markets below MIN_EDGE or MIN_VOLUME_24H should be excluded from output."""
    ohlcv_30d = _make_ohlcv(n=800, base_price=65_000.0)
    ohlcv_1min = _make_ohlcv(n=60, base_price=65_000.0)

    # Market A: large edge — yes_bid=8, yes_ask=12 → mid=0.10
    # With spot=65000 and strike=62500 (OTM call), fair prob should be noticeably > 0.10
    market_a = _make_raw_market(
        ticker="KXBTC-26JUN1316-T62500",
        yes_bid=8, yes_ask=12, volume=2000,
        expiration="2026-06-13T20:00:00Z",
    )
    # Market B: tiny edge (market near 50%), volume ok
    market_b = _make_raw_market(
        ticker="KXBTC-26JUN1316-T65000",
        yes_bid=49, yes_ask=51, volume=2000,
        expiration="2026-06-13T20:00:00Z",
    )
    # Market C: low volume — should be filtered regardless of edge
    market_c = _make_raw_market(
        ticker="KXBTC-26JUN1316-T70000",
        yes_bid=5, yes_ask=8, volume=100,  # below MIN_VOLUME_24H=500
        expiration="2026-06-13T20:00:00Z",
    )

    mock_ks = MagicMock()
    mock_ks.list_btc_markets.return_value = [market_a, market_b, market_c]

    mock_cb = MagicMock()
    mock_cb.fetch_ohlcv.side_effect = lambda p, i, s, e: (
        ohlcv_30d if i == "1h" else ohlcv_1min
    )

    output_dir = tmp_path / "output" / "kalshi"

    with patch("scripts.kalshi_btc_watchlist.KalshiSource", return_value=mock_ks), \
         patch("scripts.kalshi_btc_watchlist.CoinbaseSource", return_value=mock_cb), \
         patch("scripts.kalshi_btc_watchlist.OUTPUT_DIR", output_dir), \
         patch("scripts.kalshi_btc_watchlist.datetime") as mock_dt:

        frozen_now = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = frozen_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        from scripts.kalshi_btc_watchlist import main
        rc = main()

    assert rc == 0
    out_file = output_dir / "latest.json"
    data = json.loads(out_file.read_text())

    recs = data["recommendations"]
    tickers = [r["ticker"] for r in recs]

    # Market C (low volume) must never appear
    assert "KXBTC-26JUN1316-T70000" not in tickers, "Low-volume market should be filtered"

    # All recommendations must have |edge| >= 0.05
    for rec in recs:
        assert abs(rec["edge"]) >= 0.05, f"Edge {rec['edge']} below min for {rec['ticker']}"


# ---------------------------------------------------------------------------
# test_watchlist_json_output_schema
# ---------------------------------------------------------------------------

def test_watchlist_json_output_schema(tmp_path):
    """Output JSON must contain required top-level keys and recommendation fields."""
    ohlcv_30d = _make_ohlcv(n=800)
    ohlcv_1min = _make_ohlcv(n=60)

    raw_markets = [_make_raw_market(
        yes_bid=8, yes_ask=12, volume=1500,
        expiration="2026-06-13T22:00:00Z",
    )]

    mock_ks = MagicMock()
    mock_ks.list_btc_markets.return_value = raw_markets
    mock_cb = MagicMock()
    mock_cb.fetch_ohlcv.side_effect = lambda p, i, s, e: (
        ohlcv_30d if i == "1h" else ohlcv_1min
    )
    output_dir = tmp_path / "output" / "kalshi"

    with patch("scripts.kalshi_btc_watchlist.KalshiSource", return_value=mock_ks), \
         patch("scripts.kalshi_btc_watchlist.CoinbaseSource", return_value=mock_cb), \
         patch("scripts.kalshi_btc_watchlist.OUTPUT_DIR", output_dir), \
         patch("scripts.kalshi_btc_watchlist.datetime") as mock_dt:

        frozen_now = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = frozen_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        from scripts.kalshi_btc_watchlist import main
        main()

    data = json.loads((output_dir / "latest.json").read_text())

    required_top = {"generated_at", "btc_spot", "markets_analyzed",
                    "markets_with_edge", "recommendations"}
    assert required_top.issubset(data.keys())

    for rec in data["recommendations"]:
        required_rec = {
            "ticker", "title", "expiration", "strike", "side",
            "market_implied_prob", "fair_prob_A", "fair_prob_B", "fair_prob_AB",
            "canonical_fair_prob", "edge", "expected_value_per_dollar",
            "suggested_position_dollars", "estimator_agreement",
            "adjustments_applied", "volume_24h", "spread",
        }
        assert required_rec.issubset(rec.keys()), (
            f"Missing keys in recommendation: {required_rec - rec.keys()}"
        )
