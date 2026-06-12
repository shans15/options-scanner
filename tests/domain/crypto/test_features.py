from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain.crypto.features import build_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_merged(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic merged DataFrame with realistic column structure."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    close = 30_000 + np.cumsum(rng.normal(0, 50, n))
    merged = pd.DataFrame(
        {
            "open": close + rng.uniform(-20, 20, n),
            "high": close + rng.uniform(0, 50, n),
            "low": close - rng.uniform(0, 50, n),
            "close": close,
            "volume": rng.uniform(10, 200, n),
            "funding_rate": rng.uniform(-0.001, 0.001, n),
            "mark_price": close + rng.uniform(-5, 5, n),
        },
        index=idx,
    )
    return merged


# ---------------------------------------------------------------------------
# test_build_features_no_lookahead
# ---------------------------------------------------------------------------

def test_build_features_no_lookahead():
    """Features at time T must not change when future rows are removed.

    Strategy: build features on n rows, then on n-10 rows, and assert the
    overlapping rows produce identical feature values.
    """
    merged_full = _make_merged(n=800)
    merged_short = merged_full.iloc[:-10].copy()

    feat_full = build_features(merged_full)
    feat_short = build_features(merged_short)

    common_idx = feat_short.index
    # Compare overlapping slice — must be identical to within floating-point precision
    pd.testing.assert_frame_equal(
        feat_full.loc[common_idx],
        feat_short.loc[common_idx],
        check_exact=False,
        atol=1e-12,
        obj="features (no-lookahead check)",
    )


# ---------------------------------------------------------------------------
# test_build_features_handles_short_history
# ---------------------------------------------------------------------------

def test_build_features_handles_short_history():
    """build_features must not raise when history is shorter than rolling windows.

    The rolling windows go up to 672 bars; a 50-bar frame should produce NaNs
    for those features rather than erroring.
    """
    merged = _make_merged(n=50)
    feat = build_features(merged)

    assert feat is not None
    assert len(feat) == 50
    # Features with long windows should be all-NaN for short history
    assert feat["rv_24_pct_rank_7d"].isna().all()
    assert feat["fund_pct_rank_7d"].isna().all()
    assert feat["fund_zscore_7d"].isna().all()


# ---------------------------------------------------------------------------
# test_returns_are_log_returns
# ---------------------------------------------------------------------------

def test_returns_are_log_returns():
    """ret_1 must equal log(close_t / close_{t-1}) exactly."""
    merged = _make_merged(n=200)
    feat = build_features(merged)

    close = merged["close"]
    expected_ret1 = np.log(close / close.shift(1))

    pd.testing.assert_series_equal(
        feat["ret_1"].rename("ret_1"),
        expected_ret1.rename("ret_1"),
        check_exact=False,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# test_funding_zscore_is_zero_on_constant_input
# ---------------------------------------------------------------------------

def test_funding_zscore_is_zero_on_constant_input():
    """fund_zscore_7d must be 0 (or NaN) when funding rate is constant.

    A constant series has std=0, so z-score should be NaN (we replace 0 std with NaN
    to avoid division by zero). For the rows within the rolling window where std > 0
    (floating-point noise may cause tiny non-zero std), we allow a generous tolerance.
    """
    merged = _make_merged(n=800)
    merged["funding_rate"] = 0.0001  # constant

    feat = build_features(merged)
    zscore = feat["fund_zscore_7d"].dropna()

    # When std is truly 0, result is NaN. After dropna, any remaining values
    # must be very close to 0 (they arise only from floating-point representation).
    if len(zscore) > 0:
        assert (zscore.abs() < 1e-6).all(), (
            f"Expected fund_zscore_7d ~ 0 for constant funding, got max {zscore.abs().max()}"
        )


# ---------------------------------------------------------------------------
# test_cyclical_encoding_bounds
# ---------------------------------------------------------------------------

def test_cyclical_encoding_bounds():
    """sin/cos features must lie in [-1, 1]."""
    merged = _make_merged(n=200)
    feat = build_features(merged)

    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert feat[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


# ---------------------------------------------------------------------------
# test_bars_since_settlement_range
# ---------------------------------------------------------------------------

def test_bars_since_settlement_range():
    """bars_since_funding_settlement must be in [0, 31]."""
    merged = _make_merged(n=200)
    feat = build_features(merged)

    col = feat["bars_since_funding_settlement"]
    assert col.min() >= 0
    assert col.max() <= 31


# ---------------------------------------------------------------------------
# test_all_expected_columns_present
# ---------------------------------------------------------------------------

def test_all_expected_columns_present():
    """build_features must return exactly the documented feature set."""
    expected = {
        "ret_1", "ret_4", "ret_24",
        "rv_24", "rv_96",
        "rv_24_pct_rank_7d",
        "vol_zscore_24",
        "fund_current",
        "fund_delta_1h", "fund_delta_24h",
        "fund_pct_rank_7d", "fund_zscore_7d",
        "fund_x_price_div",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "bars_since_funding_settlement",
    }
    merged = _make_merged(n=200)
    feat = build_features(merged)

    assert set(feat.columns) == expected
