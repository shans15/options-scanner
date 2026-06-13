from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from data.sources.yahoo_macro_source import YahooMacroSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yf_df(tz="UTC") -> pd.DataFrame:
    """Return a minimal yfinance-shaped DataFrame (Title-case columns, tz-aware index)."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz=tz)
    return pd.DataFrame(
        {
            "Open":   [1.0, 1.1, 1.2, 1.3, 1.4],
            "High":   [1.5, 1.6, 1.7, 1.8, 1.9],
            "Low":    [0.9, 1.0, 1.1, 1.2, 1.3],
            "Close":  [1.2, 1.3, 1.4, 1.5, 1.6],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# test_fetch_history_returns_datetime_index
# ---------------------------------------------------------------------------

def test_fetch_history_returns_datetime_index(tmp_path):
    """fetch_history must return a DataFrame with a UTC DatetimeIndex."""
    mock_df = _make_yf_df(tz="UTC")

    with patch("data.sources.yahoo_macro_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = mock_df

        src = YahooMacroSource(cache_dir=tmp_path)
        result = src.fetch_history("DX-Y.NYB", period="2y")

    assert isinstance(result.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert result.index.tz is not None, "Index must be timezone-aware"
    assert str(result.index.tz) == "UTC", f"Expected UTC, got {result.index.tz}"
    assert result.index.name == "datetime", f"Expected index.name='datetime', got '{result.index.name}'"
    assert len(result) == 5


# ---------------------------------------------------------------------------
# test_fetch_history_handles_empty_response
# ---------------------------------------------------------------------------

def test_fetch_history_handles_empty_response(tmp_path):
    """When yfinance returns an empty DataFrame, fetch_history returns an empty
    DataFrame with the expected columns and does not raise."""
    with patch("data.sources.yahoo_macro_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = pd.DataFrame()

        src = YahooMacroSource(cache_dir=tmp_path)
        result = src.fetch_history("MISSING", period="2y")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    expected_cols = {"open", "high", "low", "close", "volume"}
    assert set(result.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {set(result.columns)}"
    )


# ---------------------------------------------------------------------------
# test_fetch_history_lowercases_columns
# ---------------------------------------------------------------------------

def test_fetch_history_lowercases_columns(tmp_path):
    """fetch_history must return lowercase column names regardless of yfinance output."""
    mock_df = _make_yf_df(tz="UTC")

    with patch("data.sources.yahoo_macro_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = mock_df

        src = YahooMacroSource(cache_dir=tmp_path)
        result = src.fetch_history("^VIX", period="2y")

    assert list(result.columns) == ["open", "high", "low", "close", "volume"], (
        f"Columns not lowercased: {list(result.columns)}"
    )


# ---------------------------------------------------------------------------
# test_caches_to_parquet
# ---------------------------------------------------------------------------

def test_caches_to_parquet(tmp_path):
    """First call writes a Parquet file; second call loads from cache without
    hitting yfinance."""
    mock_df = _make_yf_df(tz="UTC")

    with patch("data.sources.yahoo_macro_source.yf.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = mock_df

        src = YahooMacroSource(cache_dir=tmp_path)

        # First fetch — hits yfinance
        result1 = src.fetch_history("DX-Y.NYB", period="2y")
        assert MockTicker.return_value.history.call_count == 1

        # Parquet file must now exist
        cache_key = "yahoo_DX-Y_NYB_DX-Y_NYB_2y".replace("=", "_").replace("^", "_").replace(".", "_")
        # Build expected cache key the same way the source does
        ticker = "DX-Y.NYB"
        period = "2y"
        expected_key = f"yahoo_{ticker.replace('=','_').replace('^','_').replace('.','_')}_{period}"
        cache_file = tmp_path / f"{expected_key}.parquet"
        assert cache_file.exists(), f"Cache file not found: {cache_file}"

        # Second fetch — must NOT call yfinance again
        result2 = src.fetch_history("DX-Y.NYB", period="2y")
        assert MockTicker.return_value.history.call_count == 1, (
            "yfinance was called again; cache was not used"
        )

    # Parquet round-trip drops freq metadata and may change UTC tz object type;
    # compare values and index timestamps only.
    pd.testing.assert_frame_equal(
        result1.reset_index(drop=True),
        result2.reset_index(drop=True),
        check_freq=False,
    )
