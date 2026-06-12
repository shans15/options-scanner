from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from data.sources.hyperliquid_source import HyperliquidSource, _HL_API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _make_candle(open_time_ms: int, close_time_ms: int, price: float = 30_000.0) -> dict:
    """Minimal Hyperliquid candleSnapshot candle."""
    return {
        "t": open_time_ms,
        "T": close_time_ms,
        "s": "BTC",
        "i": "15m",
        "o": str(price),
        "h": str(price + 10),
        "l": str(price - 10),
        "c": str(price + 1),
        "v": "100.5",
        "n": 150,
    }


def _make_funding_entry(time_ms: int, rate: float = 0.0000125) -> dict:
    return {
        "coin": "BTC",
        "fundingRate": str(rate),
        "premium": "0.0000050",
        "time": time_ms,
    }


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_converts_ms_to_datetime
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_converts_ms_to_datetime(tmp_path):
    """Index of fetch_ohlcv result must be timezone-aware UTC datetime."""
    bar_ms = 15 * 60 * 1000
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    t1_ms = t0_ms + bar_ms  # close_time of bar 0
    t2_ms = t1_ms           # open_time of bar 1
    t3_ms = t2_ms + bar_ms  # close_time of bar 1

    candles = [
        _make_candle(t0_ms, t1_ms - 1),
        _make_candle(t2_ms, t3_ms - 1),
    ]
    resp_lib.add(resp_lib.POST, _HL_API, json=candles, status=200)

    src = HyperliquidSource(cache_dir=tmp_path)
    df = src.fetch_ohlcv("BTC", "15m", t0_ms, t3_ms)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_paginates
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_paginates(tmp_path):
    """When API returns 5000 rows (max page), a second request must be made."""
    bar_ms = 15 * 60 * 1000
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))

    # Page 1: 5000 bars
    page1 = [
        _make_candle(t0_ms + i * bar_ms, t0_ms + (i + 1) * bar_ms - 1)
        for i in range(5000)
    ]
    last_close = t0_ms + 5000 * bar_ms - 1
    next_start = last_close + 1

    # Page 2: 3 bars (final page, fewer than 5000)
    page2 = [
        _make_candle(next_start + i * bar_ms, next_start + (i + 1) * bar_ms - 1)
        for i in range(3)
    ]

    resp_lib.add(resp_lib.POST, _HL_API, json=page1, status=200)
    resp_lib.add(resp_lib.POST, _HL_API, json=page2, status=200)

    src = HyperliquidSource(cache_dir=tmp_path)
    end_ms = next_start + 3 * bar_ms
    df = src.fetch_ohlcv("BTC", "15m", t0_ms, end_ms)

    assert len(resp_lib.calls) == 2
    assert len(df) == 5003


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_deduplicates_overlap
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_deduplicates_overlap(tmp_path):
    """Duplicate index entries at page boundaries must be dropped (keep first)."""
    bar_ms = 15 * 60 * 1000
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))

    # Build page 1 and page 2 where the last candle of page1 == first candle of page2
    page1 = [
        _make_candle(t0_ms + i * bar_ms, t0_ms + (i + 1) * bar_ms - 1)
        for i in range(3)
    ]
    # page2 starts with the same open_time as page1[-1]
    overlap_t = t0_ms + 2 * bar_ms
    page2 = [
        _make_candle(overlap_t + i * bar_ms, overlap_t + (i + 1) * bar_ms - 1)
        for i in range(2)
    ]

    resp_lib.add(resp_lib.POST, _HL_API, json=page1, status=200)
    resp_lib.add(resp_lib.POST, _HL_API, json=page2, status=200)

    src = HyperliquidSource(cache_dir=tmp_path)
    end_ms = overlap_t + 2 * bar_ms + 1
    df = src.fetch_ohlcv("BTC", "15m", t0_ms, end_ms)

    # Without dedup: 5 rows; with dedup: 4 rows (overlap removed)
    assert not df.index.duplicated().any()
    assert len(df) == 4


# ---------------------------------------------------------------------------
# test_fetch_perp_funding_returns_expected_columns
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_perp_funding_returns_expected_columns(tmp_path):
    """fetch_perp_funding must return a DataFrame with funding_rate and mark_price."""
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    hour_ms = 60 * 60 * 1000
    entries = [_make_funding_entry(t0_ms + i * hour_ms) for i in range(5)]

    resp_lib.add(resp_lib.POST, _HL_API, json=entries, status=200)

    src = HyperliquidSource(cache_dir=tmp_path)
    t_end_ms = t0_ms + 5 * hour_ms
    df = src.fetch_perp_funding("BTC", t0_ms, t_end_ms)

    assert "funding_rate" in df.columns
    assert "mark_price" in df.columns
    assert len(df) == 5
    assert df["funding_rate"].dtype == float
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# test_fetch_perp_funding_pagination
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_perp_funding_pagination(tmp_path):
    """When API returns 500 entries (max page), a second request must be made."""
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    hour_ms = 60 * 60 * 1000

    page1 = [_make_funding_entry(t0_ms + i * hour_ms) for i in range(500)]
    last_time = t0_ms + 499 * hour_ms
    next_start = last_time + 1

    page2 = [_make_funding_entry(next_start + i * hour_ms) for i in range(10)]

    resp_lib.add(resp_lib.POST, _HL_API, json=page1, status=200)
    resp_lib.add(resp_lib.POST, _HL_API, json=page2, status=200)

    src = HyperliquidSource(cache_dir=tmp_path)
    end_ms = next_start + 10 * hour_ms
    df = src.fetch_perp_funding("BTC", t0_ms, end_ms)

    assert len(resp_lib.calls) == 2
    assert len(df) == 510


# ---------------------------------------------------------------------------
# test_caches_to_parquet
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_caches_to_parquet(tmp_path):
    """fetch_ohlcv must write a Parquet file and serve it on the second call."""
    bar_ms = 15 * 60 * 1000
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    candles = [_make_candle(t0_ms, t0_ms + bar_ms - 1)]

    resp_lib.add(resp_lib.POST, _HL_API, json=candles, status=200)

    src = HyperliquidSource(cache_dir=tmp_path)
    end_ms = t0_ms + bar_ms

    df1 = src.fetch_ohlcv("BTC", "15m", t0_ms, end_ms)

    # Parquet file must exist
    cache_key = f"hl_ohlcv_BTC_15m_{t0_ms}_{end_ms}"
    cache_file = tmp_path / f"{cache_key}.parquet"
    assert cache_file.exists()

    # Second call must NOT hit the network (responses would raise if it did)
    df2 = src.fetch_ohlcv("BTC", "15m", t0_ms, end_ms)
    assert len(resp_lib.calls) == 1  # only one HTTP call total
    assert df1.equals(df2)
