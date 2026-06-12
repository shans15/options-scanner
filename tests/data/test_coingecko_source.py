from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from data.sources.coingecko_source import CoinGeckoSource, _CG_API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _make_global_market_cap_chart_response(n_days: int = 5) -> dict:
    """Minimal /global/market_cap_chart response with `n_days` daily points."""
    base_ms = _ts_ms(datetime(2025, 1, 1, tzinfo=timezone.utc))
    day_ms = 86_400_000
    market_cap = [[base_ms + i * day_ms, 2_000_000_000_000 + i * 1_000_000] for i in range(n_days)]
    return {"market_cap_chart": {"market_cap": market_cap}}


def _make_btc_market_chart_response(n_days: int = 5) -> dict:
    """Minimal /coins/bitcoin/market_chart response."""
    base_ms = _ts_ms(datetime(2025, 1, 1, tzinfo=timezone.utc))
    day_ms = 86_400_000
    market_caps = [[base_ms + i * day_ms, 800_000_000_000 + i * 500_000] for i in range(n_days)]
    return {"market_caps": market_caps, "prices": [], "total_volumes": []}


def _make_global_response(dominance: float = 50.0) -> dict:
    return {"data": {"market_cap_percentage": {"btc": dominance, "eth": 17.0}}}


# ---------------------------------------------------------------------------
# test_fetch_btc_dominance_history_returns_expected_structure
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_btc_dominance_history_returns_expected_structure(tmp_path):
    """fetch_btc_dominance_history must return a DataFrame with 'btc_dominance' column
    and UTC-aware DatetimeIndex when the primary endpoints succeed."""
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global/market_cap_chart",
        json=_make_global_market_cap_chart_response(5),
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/coins/bitcoin/market_chart",
        json=_make_btc_market_chart_response(5),
        status=200,
    )

    src = CoinGeckoSource(cache_dir=tmp_path)
    df = src.fetch_btc_dominance_history(days=5)

    assert isinstance(df, pd.DataFrame)
    assert "btc_dominance" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert len(df) == 5
    # Dominance values must be percentages (> 0, < 100)
    assert (df["btc_dominance"] > 0).all()
    assert (df["btc_dominance"] < 100).all()


@resp_lib.activate
def test_fetch_btc_dominance_fallback_to_global_snapshot(tmp_path):
    """When the primary endpoint (market_cap_chart) returns empty data,
    the method should fall back to /global snapshot and build a constant series."""
    # Primary endpoint returns empty market_cap list → triggers fallback
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global/market_cap_chart",
        json={"market_cap_chart": {"market_cap": []}},
        status=200,
    )
    # Fallback /global call
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global",
        json=_make_global_response(dominance=48.5),
        status=200,
    )

    src = CoinGeckoSource(cache_dir=tmp_path)
    df = src.fetch_btc_dominance_history(days=30)

    assert "btc_dominance" in df.columns
    assert len(df) == 30
    # All values should be the constant dominance from /global
    assert (df["btc_dominance"] == 48.5).all()


@resp_lib.activate
def test_fetch_btc_dominance_caches_to_parquet(tmp_path):
    """Second call must not hit the network."""
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global/market_cap_chart",
        json=_make_global_market_cap_chart_response(3),
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/coins/bitcoin/market_chart",
        json=_make_btc_market_chart_response(3),
        status=200,
    )

    src = CoinGeckoSource(cache_dir=tmp_path)
    df1 = src.fetch_btc_dominance_history(days=3)

    # Second call — responses library would raise ConnectionError if network is hit
    df2 = src.fetch_btc_dominance_history(days=3)

    assert len(resp_lib.calls) == 2  # only 2 HTTP calls total (global_chart + btc_chart)
    pd.testing.assert_frame_equal(df1, df2)


@resp_lib.activate
def test_fetch_total_market_cap_history_returns_expected_structure(tmp_path):
    """fetch_total_market_cap_history must return DataFrame with 'total_market_cap_usd'."""
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global/market_cap_chart",
        json=_make_global_market_cap_chart_response(7),
        status=200,
    )

    src = CoinGeckoSource(cache_dir=tmp_path)
    df = src.fetch_total_market_cap_history(days=7)

    assert isinstance(df, pd.DataFrame)
    assert "total_market_cap_usd" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert len(df) == 7
    assert (df["total_market_cap_usd"] > 0).all()


@resp_lib.activate
def test_fetch_total_market_cap_caches_to_parquet(tmp_path):
    """Second call must not hit the network."""
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global/market_cap_chart",
        json=_make_global_market_cap_chart_response(4),
        status=200,
    )

    src = CoinGeckoSource(cache_dir=tmp_path)
    df1 = src.fetch_total_market_cap_history(days=4)
    df2 = src.fetch_total_market_cap_history(days=4)

    assert len(resp_lib.calls) == 1
    pd.testing.assert_frame_equal(df1, df2)


@resp_lib.activate
def test_fetch_btc_dominance_index_is_monotonic(tmp_path):
    """Returned index must be monotonically increasing."""
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/global/market_cap_chart",
        json=_make_global_market_cap_chart_response(10),
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        f"{_CG_API}/coins/bitcoin/market_chart",
        json=_make_btc_market_chart_response(10),
        status=200,
    )

    src = CoinGeckoSource(cache_dir=tmp_path)
    df = src.fetch_btc_dominance_history(days=10)

    assert df.index.is_monotonic_increasing
