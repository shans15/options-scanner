from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from data.sources.coinbase_source import CoinbaseSource, _CB_API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _s(dt: datetime) -> int:
    return int(dt.timestamp())


def _make_candle(time_s: int, price: float = 30_000.0) -> list:
    """Minimal Coinbase candle: [time_s, low, high, open, close, volume]."""
    return [time_s, price - 10, price + 10, price, price + 1, 100.5]


def _product_url(product_id: str = "BTC-USD") -> str:
    return f"{_CB_API}/products/{product_id}/candles"


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_converts_seconds_to_datetime
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_converts_seconds_to_datetime(tmp_path):
    """Index of fetch_ohlcv result must be timezone-aware UTC datetime."""
    gran_s = 900  # 15m
    t0_s = _s(datetime(2024, 1, 1, tzinfo=timezone.utc))
    t1_s = t0_s + gran_s

    # Coinbase returns descending: latest first
    candles = [_make_candle(t1_s), _make_candle(t0_s)]
    resp_lib.add(resp_lib.GET, _product_url(), json=candles, status=200)

    src = CoinbaseSource(cache_dir=tmp_path)
    start_ms = t0_s * 1000
    end_ms = (t1_s + gran_s) * 1000  # ensure loop covers both candles

    df = src.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_handles_descending_response_order
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_handles_descending_response_order(tmp_path):
    """Coinbase returns DESC; final DataFrame must be sorted ASC by datetime."""
    gran_s = 900  # 15m
    t0_s = _s(datetime(2024, 1, 1, tzinfo=timezone.utc))
    t1_s = t0_s + gran_s
    t2_s = t1_s + gran_s

    # Coinbase returns newest first
    candles = [_make_candle(t2_s, 31_000.0), _make_candle(t1_s, 30_500.0), _make_candle(t0_s, 30_000.0)]
    resp_lib.add(resp_lib.GET, _product_url(), json=candles, status=200)

    src = CoinbaseSource(cache_dir=tmp_path)
    start_ms = t0_s * 1000
    end_ms = (t2_s + gran_s) * 1000

    df = src.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)

    # Index must be strictly ascending
    assert df.index.is_monotonic_increasing
    assert len(df) == 3
    # First row should be the earliest candle
    assert df.index[0] == pd.Timestamp(t0_s, unit="s", tz="UTC")
    assert df.index[-1] == pd.Timestamp(t2_s, unit="s", tz="UTC")


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_paginates
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_paginates(tmp_path):
    """When range spans more than _PAGE_LIMIT bars, multiple requests are made."""
    gran_s = 900  # 15m
    page_limit = CoinbaseSource._PAGE_LIMIT  # 300
    t0_s = _s(datetime(2024, 1, 1, tzinfo=timezone.utc))

    # Page 1: 300 candles, descending (latest first)
    page1 = [_make_candle(t0_s + (page_limit - 1 - i) * gran_s) for i in range(page_limit)]
    page1_latest_s = t0_s + (page_limit - 1) * gran_s

    # Page 2: 5 candles, starting right after page 1
    page2_start_s = page1_latest_s + gran_s
    page2 = [_make_candle(page2_start_s + (4 - i) * gran_s) for i in range(5)]

    resp_lib.add(resp_lib.GET, _product_url(), json=page1, status=200)
    resp_lib.add(resp_lib.GET, _product_url(), json=page2, status=200)

    src = CoinbaseSource(cache_dir=tmp_path)
    start_ms = t0_s * 1000
    end_ms = (page2_start_s + 5 * gran_s) * 1000

    df = src.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)

    assert len(resp_lib.calls) == 2
    assert len(df) == page_limit + 5
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_invalid_interval_raises
# ---------------------------------------------------------------------------

def test_fetch_ohlcv_invalid_interval_raises(tmp_path):
    """Unsupported interval must raise ValueError immediately, before any HTTP call."""
    src = CoinbaseSource(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unsupported interval"):
        src.fetch_ohlcv("BTC-USD", "2m", 0, 1_000_000)


# ---------------------------------------------------------------------------
# test_caches_to_parquet
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_caches_to_parquet(tmp_path):
    """fetch_ohlcv must write a Parquet file and serve it on the second call."""
    gran_s = 900
    t0_s = _s(datetime(2024, 1, 1, tzinfo=timezone.utc))
    candles = [_make_candle(t0_s)]
    resp_lib.add(resp_lib.GET, _product_url(), json=candles, status=200)

    src = CoinbaseSource(cache_dir=tmp_path)
    start_ms = t0_s * 1000
    end_ms = (t0_s + gran_s) * 1000

    df1 = src.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)

    cache_key = f"cb_ohlcv_BTC-USD_15m_{start_ms}_{end_ms}"
    cache_file = tmp_path / f"{cache_key}.parquet"
    assert cache_file.exists()

    # Second call must NOT hit the network
    df2 = src.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
    assert len(resp_lib.calls) == 1  # only one HTTP call total
    assert df1.equals(df2)


# ---------------------------------------------------------------------------
# test_fetch_ohlcv_handles_empty_batch
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_ohlcv_handles_empty_batch(tmp_path):
    """When API returns [], cursor advances and function returns empty DataFrame
    without looping forever."""
    resp_lib.add(resp_lib.GET, _product_url(), json=[], status=200)

    src = CoinbaseSource(cache_dir=tmp_path)
    gran_s = 900
    t0_s = _s(datetime(2024, 1, 1, tzinfo=timezone.utc))
    # Request a single-page window so the loop terminates after one empty response
    start_ms = t0_s * 1000
    end_ms = (t0_s + gran_s) * 1000

    df = src.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
