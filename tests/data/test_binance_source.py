from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from data.sources.binance_source import BinanceSource, _BINANCE_API, _BINANCE_FAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _make_kline(open_time_ms: int, close_time_ms: int, price: float = 30_000.0) -> list:
    """Minimal Binance kline row."""
    return [
        open_time_ms,          # 0 open_time
        str(price),            # 1 open
        str(price + 10),       # 2 high
        str(price - 10),       # 3 low
        str(price + 1),        # 4 close
        "100.5",               # 5 volume
        close_time_ms,         # 6 close_time
        "3015000",             # 7 quote_asset_volume
        150,                   # 8 num_trades
        "50.0",                # 9 taker_buy_base_vol
        "1500000",             # 10 taker_buy_quote_vol
        "0",                   # 11 ignore
    ]


def _make_funding_entry(funding_time_ms: int, rate: float = 0.0001) -> dict:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": funding_time_ms,
        "fundingRate": str(rate),
        "markPrice": "30000.0",
    }


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_converts_ms_to_datetime
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_converts_ms_to_datetime(tmp_path):
    """Index of fetch_ohlcv result must be timezone-aware UTC datetime."""
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    t1_ms = t0_ms + 15 * 60 * 1000  # 15-min later (close_time of bar 0)
    t2_ms = t1_ms + 15 * 60 * 1000  # open_time of bar 1
    t3_ms = t2_ms + 15 * 60 * 1000  # close_time of bar 1

    klines = [
        _make_kline(t0_ms, t1_ms - 1),
        _make_kline(t2_ms, t3_ms - 1),
    ]
    resp_lib.add(
        resp_lib.GET,
        f"{_BINANCE_API}/api/v3/klines",
        json=klines,
        status=200,
    )

    src = BinanceSource(cache_dir=tmp_path)
    df = src.fetch_ohlcv("BTCUSDT", "15m", t0_ms, t3_ms)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None  # timezone-aware
    assert str(df.index.tz) == "UTC"
    assert len(df) == 2


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_paginates_when_more_than_1000
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_paginates_when_more_than_1000(tmp_path):
    """When API returns exactly 1000 rows, a second request must be made."""
    bar_ms = 15 * 60 * 1000  # 15-min in ms
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))

    # Page 1: 1000 bars
    page1 = [
        _make_kline(t0_ms + i * bar_ms, t0_ms + (i + 1) * bar_ms - 1)
        for i in range(1000)
    ]
    # Last bar's close_time determines the next startTime
    last_close = t0_ms + 1000 * bar_ms - 1
    next_start = last_close + 1

    # Page 2: 3 bars
    page2 = [
        _make_kline(next_start + i * bar_ms, next_start + (i + 1) * bar_ms - 1)
        for i in range(3)
    ]

    resp_lib.add(
        resp_lib.GET,
        f"{_BINANCE_API}/api/v3/klines",
        json=page1,
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        f"{_BINANCE_API}/api/v3/klines",
        json=page2,
        status=200,
    )

    src = BinanceSource(cache_dir=tmp_path)
    end_ms = next_start + 3 * bar_ms
    df = src.fetch_ohlcv("BTCUSDT", "15m", t0_ms, end_ms)

    # Two pages were called
    assert len(resp_lib.calls) == 2
    assert len(df) == 1003


# ---------------------------------------------------------------------------
# test_fetch_perp_funding_returns_expected_columns
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_perp_funding_returns_expected_columns(tmp_path):
    """fetch_perp_funding must return a DataFrame with funding_rate and mark_price."""
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    entries = [_make_funding_entry(t0_ms + i * 8 * 3600 * 1000) for i in range(5)]

    resp_lib.add(
        resp_lib.GET,
        f"{_BINANCE_FAPI}/fapi/v1/fundingRate",
        json=entries,
        status=200,
    )

    src = BinanceSource(cache_dir=tmp_path)
    t_end_ms = t0_ms + 5 * 8 * 3600 * 1000
    df = src.fetch_perp_funding("BTCUSDT", t0_ms, t_end_ms)

    assert "funding_rate" in df.columns
    assert "mark_price" in df.columns
    assert len(df) == 5
    assert df["funding_rate"].dtype == float
    assert df.index.tz is not None


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_returns_empty_on_no_data
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_returns_empty_on_no_data(tmp_path):
    """Empty API response must produce an empty DataFrame without raising."""
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))

    resp_lib.add(
        resp_lib.GET,
        f"{_BINANCE_API}/api/v3/klines",
        json=[],
        status=200,
    )

    src = BinanceSource(cache_dir=tmp_path)
    df = src.fetch_ohlcv("BTCUSDT", "15m", t0_ms, t0_ms + 1000)

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_uses_cache_on_second_call
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_uses_cache_on_second_call(tmp_path):
    """A second call with the same parameters must read from cache, not HTTP."""
    t0_ms = _ms(datetime(2024, 1, 1, tzinfo=timezone.utc))
    bar_ms = 15 * 60 * 1000
    klines = [_make_kline(t0_ms, t0_ms + bar_ms - 1)]

    resp_lib.add(
        resp_lib.GET,
        f"{_BINANCE_API}/api/v3/klines",
        json=klines,
        status=200,
    )

    src = BinanceSource(cache_dir=tmp_path)
    end_ms = t0_ms + bar_ms

    # First call — hits HTTP
    df1 = src.fetch_ohlcv("BTCUSDT", "15m", t0_ms, end_ms)
    # Second call — should read from Parquet cache
    df2 = src.fetch_ohlcv("BTCUSDT", "15m", t0_ms, end_ms)

    assert len(resp_lib.calls) == 1  # only one HTTP call total
    assert df1.equals(df2)
