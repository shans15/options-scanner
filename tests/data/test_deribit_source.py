from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_lib

from data.sources.deribit_source import DeribitSource, _DERIBIT_API


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _make_hv_response(n_days: int = 10, currency: str = "BTC") -> dict:
    """Minimal Deribit get_historical_volatility response.

    Format: {"jsonrpc": "2.0", "result": [[timestamp_ms, hv_value], ...]}
    """
    base_ms = _ts_ms(datetime(2025, 1, 1, tzinfo=timezone.utc))
    day_ms = 86_400_000
    result = [[base_ms + i * day_ms, 60.0 + i * 0.5] for i in range(n_days)]
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def _hv_url() -> str:
    return f"{_DERIBIT_API}/get_historical_volatility"


# ---------------------------------------------------------------------------
# test_fetch_historical_volatility_returns_expected_structure
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_historical_volatility_returns_expected_structure(tmp_path):
    """fetch_historical_volatility must return DataFrame with 'hv_30d' column
    and UTC-aware DatetimeIndex."""
    resp_lib.add(resp_lib.GET, _hv_url(), json=_make_hv_response(10), status=200)

    src = DeribitSource(cache_dir=tmp_path)
    df = src.fetch_historical_volatility("BTC")

    assert isinstance(df, pd.DataFrame)
    assert "hv_30d" in df.columns
    assert list(df.columns) == ["hv_30d"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert len(df) == 10
    assert df["hv_30d"].dtype == float


@resp_lib.activate
def test_fetch_historical_volatility_index_is_monotonic(tmp_path):
    """Returned index must be sorted ascending (monotonically increasing)."""
    # Return data in shuffled order to test that sorting is applied
    base_ms = _ts_ms(datetime(2025, 1, 1, tzinfo=timezone.utc))
    day_ms = 86_400_000
    result = [
        [base_ms + 2 * day_ms, 62.0],
        [base_ms + 0 * day_ms, 60.0],
        [base_ms + 1 * day_ms, 61.0],
    ]
    resp_lib.add(resp_lib.GET, _hv_url(), json={"result": result}, status=200)

    src = DeribitSource(cache_dir=tmp_path)
    df = src.fetch_historical_volatility("BTC")

    assert df.index.is_monotonic_increasing
    assert df["hv_30d"].iloc[0] == 60.0   # earliest first
    assert df["hv_30d"].iloc[-1] == 62.0  # latest last


@resp_lib.activate
def test_fetch_historical_volatility_deduplicates(tmp_path):
    """Duplicate timestamps must be dropped (keep first)."""
    base_ms = _ts_ms(datetime(2025, 1, 1, tzinfo=timezone.utc))
    result = [
        [base_ms, 60.0],
        [base_ms, 60.5],   # duplicate timestamp
        [base_ms + 86_400_000, 61.0],
    ]
    resp_lib.add(resp_lib.GET, _hv_url(), json={"result": result}, status=200)

    src = DeribitSource(cache_dir=tmp_path)
    df = src.fetch_historical_volatility("BTC")

    assert not df.index.duplicated().any()
    assert len(df) == 2
    # First occurrence kept
    assert df["hv_30d"].iloc[0] == 60.0


@resp_lib.activate
def test_fetch_historical_volatility_caches_to_parquet(tmp_path):
    """Second call must not hit the network."""
    resp_lib.add(resp_lib.GET, _hv_url(), json=_make_hv_response(5), status=200)

    src = DeribitSource(cache_dir=tmp_path)
    df1 = src.fetch_historical_volatility("BTC")
    # Second call — should load from parquet, not network
    df2 = src.fetch_historical_volatility("BTC")

    assert len(resp_lib.calls) == 1
    pd.testing.assert_frame_equal(df1, df2)


@resp_lib.activate
def test_fetch_historical_volatility_raises_on_empty_result(tmp_path):
    """Empty result must raise ValueError (not silently return empty DataFrame)."""
    resp_lib.add(resp_lib.GET, _hv_url(), json={"result": []}, status=200)

    src = DeribitSource(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="no data"):
        src.fetch_historical_volatility("BTC")


@resp_lib.activate
def test_fetch_historical_volatility_eth(tmp_path):
    """ETH currency should hit the same endpoint with currency=ETH param."""
    resp_lib.add(resp_lib.GET, _hv_url(), json=_make_hv_response(5, "ETH"), status=200)

    src = DeribitSource(cache_dir=tmp_path)
    df = src.fetch_historical_volatility("ETH")

    assert len(df) == 5
    # Verify the request was made with the correct currency param
    assert len(resp_lib.calls) == 1
    assert "currency=ETH" in resp_lib.calls[0].request.url
