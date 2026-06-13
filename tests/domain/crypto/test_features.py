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
# test_all_expected_columns_present
# ---------------------------------------------------------------------------

def test_all_expected_columns_present():
    """build_features must return at least the documented base feature set.

    Pruned features (session_*, btc_dom_*, hv_30d, bars_since_funding_settlement,
    consec_down_bars, gap_pct, body_pct, intrabar_range, fund_x_dom,
    crypto_breadth_24) must NOT be present.
    """
    base_expected = {
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
        "upper_shadow_pct",
        "vol_zscore_96",
        "fund_x_vol", "fund_x_ret", "vol_x_ret",
        # Price action (pruned)
        "close_to_high_20", "close_to_low_20", "range_pct_20",
        "consec_up_bars", "range_expansion",
        # Volume
        "vol_ratio_4_24", "cvd_proxy_24", "vol_breakout", "up_vol_pct_24",
    }
    merged = _make_merged(n=200)
    feat = build_features(merged)

    assert base_expected.issubset(set(feat.columns)), (
        f"Missing columns: {base_expected - set(feat.columns)}"
    )

    # Pruned features must NOT appear
    pruned = {
        "session_asia", "session_eu", "session_us", "session_overnight",
        "btc_dom_current", "btc_dom_delta_24h",
        "hv_30d",
        "consec_down_bars", "gap_pct",
        "body_pct", "intrabar_range",
        "fund_x_dom",
        "crypto_breadth_24",
    }
    present_pruned = pruned & set(feat.columns)
    assert not present_pruned, f"Pruned features still present: {present_pruned}"

    # Without optional source columns, macro/dominance/HV features must NOT be present
    assert "btc_dom_current" not in feat.columns
    assert "hv_30d" not in feat.columns
    assert "dxy_ret_24" not in feat.columns
    assert "vix_current" not in feat.columns


def test_intrabar_features_non_negative():
    """upper_shadow_pct must be >= 0."""
    merged = _make_merged(n=400)
    feat = build_features(merged)

    assert (feat["upper_shadow_pct"] >= 0).all(), "upper_shadow_pct has negative values"


def test_no_temporal_leakage_in_features():
    """Compute features on a longer series and a truncated version.

    For each row in the common range, the feature values must be IDENTICAL.
    This catches any rolling/cumulative operation that accidentally uses future
    data (e.g., a rolling operation with center=True, or a percentile computed
    on the full history).

    Includes all optional source columns: eth_close, eth_volume, sol_close,
    sol_volume, dxy_close, vix_close.
    """
    np.random.seed(42)
    n_full = 500
    n_truncated = 300

    idx_full = pd.date_range("2025-01-01", periods=n_full, freq="15min", tz="UTC")
    close_full = 100.0 + np.cumsum(np.random.normal(0, 0.5, n_full))
    eth_close_full = 2000.0 + np.cumsum(np.random.normal(0, 5.0, n_full))
    sol_close_full = 100.0 + np.cumsum(np.random.normal(0, 1.0, n_full))
    dxy_full = 100.0 + np.cumsum(np.random.normal(0, 0.1, n_full))
    vix_full = 20.0 + np.cumsum(np.random.normal(0, 0.2, n_full))
    merged_full = pd.DataFrame(
        {
            "open": close_full * (1 + np.random.normal(0, 0.0005, n_full)),
            "high": close_full * (1 + np.abs(np.random.normal(0, 0.001, n_full))),
            "low": close_full * (1 - np.abs(np.random.normal(0, 0.001, n_full))),
            "close": close_full,
            "volume": np.random.uniform(50, 200, n_full),
            "funding_rate": np.random.normal(0, 0.0001, n_full),
            "mark_price": close_full,
            # Optional source columns
            "eth_close": eth_close_full,
            "eth_volume": np.random.uniform(100, 500, n_full),
            "sol_close": sol_close_full,
            "sol_volume": np.random.uniform(50, 300, n_full),
            "dxy_close": dxy_full,
            "vix_close": vix_full,
        },
        index=idx_full,
    )
    merged_truncated = merged_full.iloc[:n_truncated].copy()

    feat_full = build_features(merged_full, settlement_period_bars=4)
    feat_truncated = build_features(merged_truncated, settlement_period_bars=4)

    # The first n_truncated rows of feat_full must EXACTLY match feat_truncated
    common = feat_full.iloc[:n_truncated]
    pd.testing.assert_frame_equal(
        common,
        feat_truncated,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
        obj="No-leakage check: features at time T must not depend on data after T",
    )


# ---------------------------------------------------------------------------
# New feature tests
# ---------------------------------------------------------------------------

def test_close_to_high_20_in_unit_range():
    """close_to_high_20 should be in (0, 1] — close is never above its recent 20-bar high."""
    merged = _make_merged(n=200)
    feat = build_features(merged)

    col = feat["close_to_high_20"].dropna()
    assert (col > 0).all(), "close_to_high_20 should be > 0"
    assert (col <= 1.0 + 1e-9).all(), f"close_to_high_20 should be <= 1, max={col.max()}"


def test_consec_up_bars_capped_at_10():
    """15 consecutive up bars should yield consec_up_bars == 10 (cap enforced)."""
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    # All bars are clearly up: close > open for all of them
    open_prices = np.ones(n) * 100.0
    close_prices = np.ones(n) * 101.0
    merged = pd.DataFrame(
        {
            "open": open_prices,
            "high": close_prices + 1,
            "low": open_prices - 1,
            "close": close_prices,
            "volume": rng.uniform(10, 200, n),
            "funding_rate": rng.uniform(-0.001, 0.001, n),
            "mark_price": close_prices,
        },
        index=idx,
    )
    feat = build_features(merged)
    # By bar index 15 (0-based), we've had 15 consecutive up bars before it,
    # but capped at 10.
    assert feat["consec_up_bars"].iloc[-1] == 10, (
        f"Expected 10 (capped), got {feat['consec_up_bars'].iloc[-1]}"
    )


def test_cross_asset_features_skipped_when_columns_absent():
    """If merged lacks eth_close/sol_close, cross-asset features must not appear."""
    merged = _make_merged(n=200)
    # No eth_close, no sol_close
    feat = build_features(merged)

    cross_asset_cols = [
        "eth_ret_24", "sol_ret_24", "eth_btc_ratio", "eth_btc_ratio_delta_24",
        "correl_btc_eth_96", "btc_eth_div", "eth_vol_zscore_24",
        "vol_x_breadth",
    ]
    for col in cross_asset_cols:
        assert col not in feat.columns, f"'{col}' should not be present without eth/sol data"


def test_cross_asset_features_present_when_columns_provided():
    """If merged has eth_close + sol_close, cross-asset features ARE present."""
    rng = np.random.default_rng(7)
    merged = _make_merged(n=200)
    merged["eth_close"] = 2000.0 + np.cumsum(rng.normal(0, 5, 200))
    merged["eth_volume"] = rng.uniform(100, 500, 200)
    merged["sol_close"] = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    merged["sol_volume"] = rng.uniform(50, 300, 200)

    feat = build_features(merged)

    cross_asset_cols = [
        "eth_ret_24", "sol_ret_24", "eth_btc_ratio", "eth_btc_ratio_delta_24",
        "correl_btc_eth_96", "btc_eth_div", "eth_vol_zscore_24",
        "vol_x_breadth",
    ]
    for col in cross_asset_cols:
        assert col in feat.columns, f"'{col}' should be present when eth/sol data is available"


def test_eth_btc_ratio_correct():
    """eth_btc_ratio must equal eth_close / btc_close exactly."""
    merged = _make_merged(n=200)
    merged["eth_close"] = 2000.0
    merged["sol_close"] = 100.0

    feat = build_features(merged)
    expected = merged["eth_close"] / merged["close"]
    pd.testing.assert_series_equal(
        feat["eth_btc_ratio"].rename("eth_btc_ratio"),
        expected.rename("eth_btc_ratio"),
        check_exact=False,
        atol=1e-12,
    )


def test_correl_btc_eth_96_is_one_for_identical_series():
    """If ETH returns == BTC returns, rolling 96-bar correlation should be 1.0."""
    rng = np.random.default_rng(99)
    n = 200
    merged = _make_merged(n=n, seed=99)
    # Make ETH exactly equal to BTC (same close prices → same log returns)
    merged["eth_close"] = merged["close"].values.copy()
    merged["sol_close"] = merged["close"].values.copy()

    feat = build_features(merged)

    # After the 96-bar warm-up, correlation should be 1.0
    corr = feat["correl_btc_eth_96"].dropna()
    assert len(corr) > 0, "Expected some non-NaN correlation values"
    assert (corr.round(10) == 1.0).all(), (
        f"Expected correlation = 1.0 for identical series, got min={corr.min():.6f}"
    )


# ---------------------------------------------------------------------------
# Macro feature tests (Path 3)
# ---------------------------------------------------------------------------

def test_macro_features_absent_without_columns():
    """Macro features must not appear when dxy_close/vix_close are not in merged."""
    merged = _make_merged(n=200)
    feat = build_features(merged)

    macro_cols = [
        "dxy_ret_24", "dxy_vs_btc_24",
        "vix_current", "vix_zscore_30d",
        "correl_btc_vix_96", "correl_btc_dxy_96",
    ]
    for col in macro_cols:
        assert col not in feat.columns, f"'{col}' should not be present without macro columns"


def test_macro_features_present_when_dxy_provided():
    """dxy_ret_24 and dxy_vs_btc_24 appear when dxy_close is in merged."""
    rng = np.random.default_rng(5)
    merged = _make_merged(n=200)
    merged["dxy_close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, 200))

    feat = build_features(merged)

    assert "dxy_ret_24" in feat.columns, "dxy_ret_24 should be present when dxy_close is in merged"
    assert "dxy_vs_btc_24" in feat.columns, "dxy_vs_btc_24 should be present when dxy_close is in merged"


def test_macro_features_present_when_vix_provided():
    """vix_current and vix_zscore_30d appear when vix_close is in merged."""
    rng = np.random.default_rng(6)
    merged = _make_merged(n=200)
    merged["vix_close"] = 20.0 + np.abs(rng.normal(0, 2, 200))

    feat = build_features(merged)

    assert "vix_current" in feat.columns, "vix_current should be present when vix_close is in merged"
    assert "vix_zscore_30d" in feat.columns, "vix_zscore_30d should be present when vix_close is in merged"


def test_correlation_macro_features_present_when_both_provided():
    """correl_btc_vix_96 and correl_btc_dxy_96 appear only when both dxy_close
    AND vix_close are in merged."""
    rng = np.random.default_rng(7)
    n = 200
    merged = _make_merged(n=n)
    merged["dxy_close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    merged["vix_close"] = 20.0 + np.abs(rng.normal(0, 2, n))

    feat = build_features(merged)

    assert "correl_btc_vix_96" in feat.columns
    assert "correl_btc_dxy_96" in feat.columns


def test_dxy_vs_btc_24_sign():
    """dxy_vs_btc_24 = dxy_ret_24 * -ret_24. When DXY rises and BTC falls,
    the product should be positive (bearish macro alignment)."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    # Monotonically rising DXY and falling BTC prices
    dxy = np.linspace(100, 110, n)
    btc = np.linspace(50000, 40000, n)
    merged = pd.DataFrame(
        {
            "open": btc,
            "high": btc + 50,
            "low": btc - 50,
            "close": btc,
            "volume": np.ones(n) * 100,
            "funding_rate": np.zeros(n),
            "mark_price": btc,
            "dxy_close": dxy,
        },
        index=idx,
    )

    feat = build_features(merged)

    # After warm-up (24 bars), dxy_vs_btc_24 should be positive
    signal = feat["dxy_vs_btc_24"].dropna()
    assert len(signal) > 0
    assert (signal > 0).all(), (
        f"Expected dxy_vs_btc_24 > 0 when DXY up and BTC down, got min={signal.min():.6f}"
    )
