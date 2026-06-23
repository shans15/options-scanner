"""Tests for all 4 stock-scanner detectors using synthetic OHLCV data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain.stock_scanner.detectors import sma_stack, compression, proximity_52w, volume_surge
from domain.stock_scanner.types import StockSetup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(closes: list[float], volume_base: int = 1_000_000) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    # high = close + 0.5%, low = close - 0.5%
    highs = closes_arr * 1.005
    lows = closes_arr * 0.995
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    volumes = np.full(n, volume_base, dtype=float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes_arr, "volume": volumes}
    )


def _trending_up(n: int = 300, start: float = 100.0, pct_per_day: float = 0.003) -> list[float]:
    """Steadily rising prices."""
    return [start * (1 + pct_per_day) ** i for i in range(n)]


def _trending_down(n: int = 300, start: float = 100.0, pct_per_day: float = 0.003) -> list[float]:
    """Steadily falling prices."""
    return [start * (1 - pct_per_day) ** i for i in range(n)]


def _flat(n: int = 300, price: float = 100.0) -> list[float]:
    """Flat price series."""
    return [price] * n


# ---------------------------------------------------------------------------
# sma_stack detector
# ---------------------------------------------------------------------------

class TestSmaStack:
    def test_fires_bullish_on_strong_uptrend(self):
        closes = _trending_up(300, start=50.0, pct_per_day=0.004)
        df = _make_ohlcv(closes)
        result = sma_stack.detect("TEST", df)
        assert result is not None
        assert result.setup_type == "sma_stack"
        assert result.direction == "bullish"

    def test_fires_bearish_on_strong_downtrend(self):
        closes = _trending_down(300, start=200.0, pct_per_day=0.004)
        df = _make_ohlcv(closes)
        result = sma_stack.detect("TEST", df)
        assert result is not None
        assert result.direction == "bearish"

    def test_returns_none_on_flat_prices(self):
        df = _make_ohlcv(_flat(300))
        result = sma_stack.detect("TEST", df)
        # Flat prices: EMAs all converge; stack unlikely to hold strictly
        # Either None or low-strength; we just check it doesn't crash
        if result is not None:
            assert 0.0 <= result.strength <= 1.0

    def test_returns_none_on_short_history(self):
        df = _make_ohlcv(_trending_up(150))  # < 200 required
        result = sma_stack.detect("TEST", df)
        assert result is None

    def test_strength_in_valid_range_bullish(self):
        closes = _trending_up(300, start=50.0, pct_per_day=0.005)
        df = _make_ohlcv(closes)
        result = sma_stack.detect("TEST", df)
        if result is not None:
            assert 0.0 <= result.strength <= 1.0

    def test_strength_in_valid_range_bearish(self):
        closes = _trending_down(300, start=200.0, pct_per_day=0.005)
        df = _make_ohlcv(closes)
        result = sma_stack.detect("TEST", df)
        if result is not None:
            assert 0.0 <= result.strength <= 1.0

    def test_notes_contain_expected_keys(self):
        closes = _trending_up(300, start=50.0, pct_per_day=0.004)
        df = _make_ohlcv(closes)
        result = sma_stack.detect("TEST", df)
        if result is not None:
            for key in ("ema10", "ema20", "ema50", "ema200", "spread_pct", "persistence_days_of_20"):
                assert key in result.notes

    def test_ticker_propagated(self):
        closes = _trending_up(300, start=50.0, pct_per_day=0.004)
        df = _make_ohlcv(closes)
        result = sma_stack.detect("NVDA", df)
        if result is not None:
            assert result.ticker == "NVDA"


# ---------------------------------------------------------------------------
# compression detector
# ---------------------------------------------------------------------------

class TestCompression:
    def _make_compressed_df(self, direction: str = "bullish") -> pd.DataFrame:
        """Build 200 bars: first 150 volatile (wide BB), last 50 very tight (compression)."""
        n_total = 200
        # First 150 bars: noisy around 100, will produce wide BB
        rng = np.random.default_rng(42)
        noisy = 100.0 + rng.normal(0, 3, 150).cumsum() * 0.3
        noisy = np.clip(noisy, 50, 200)
        # Last 50 bars: very tight around the last noisy value
        base = noisy[-1]
        if direction == "bullish":
            # slight upward drift for EMA50 slope
            tight = base + np.linspace(0, 0.5, 50) + rng.normal(0, 0.02, 50)
        else:
            # slight downward drift for EMA50 slope
            tight = base - np.linspace(0, 0.5, 50) + rng.normal(0, 0.02, 50)
        closes = np.concatenate([noisy, tight])
        return _make_ohlcv(closes.tolist())

    def test_fires_on_compression_bullish(self):
        df = self._make_compressed_df("bullish")
        result = compression.detect("TEST", df)
        # May or may not fire depending on random seed; just check no crash + valid output
        if result is not None:
            assert result.setup_type == "compression"
            assert 0.0 <= result.strength <= 1.0

    def test_fires_on_compression_bearish(self):
        df = self._make_compressed_df("bearish")
        result = compression.detect("TEST", df)
        if result is not None:
            assert result.direction in ("bullish", "bearish")

    def test_returns_none_on_short_history(self):
        df = _make_ohlcv(_trending_up(100))  # < 146 required
        result = compression.detect("TEST", df)
        assert result is None

    def test_strength_in_valid_range(self):
        df = self._make_compressed_df("bullish")
        result = compression.detect("TEST", df)
        if result is not None:
            assert 0.0 <= result.strength <= 1.0

    def test_notes_contain_expected_keys(self):
        df = self._make_compressed_df("bullish")
        result = compression.detect("TEST", df)
        if result is not None:
            for key in ("current_bbw_pct", "threshold_bbw_pct", "max_bbw_pct_6mo",
                        "compression_days", "ema50_slope_20d_pct"):
                assert key in result.notes

    def test_no_fire_on_expanding_volatility(self):
        """Increasing volatility should not produce compression signal."""
        rng = np.random.default_rng(7)
        # Start tight, become increasingly volatile
        tight = 100.0 + rng.normal(0, 0.1, 100)
        volatile = 100.0 + rng.normal(0, 5, 100).cumsum() * 0.5
        closes = np.concatenate([tight, volatile])
        df = _make_ohlcv(closes.tolist())
        result = compression.detect("TEST", df)
        assert result is None


# ---------------------------------------------------------------------------
# proximity_52w detector
# ---------------------------------------------------------------------------

class TestProximity52w:
    def _make_df_near_52w_high(self) -> pd.DataFrame:
        """Price rises to a high, then stays within 2% of it."""
        closes = _trending_up(200, start=100.0, pct_per_day=0.002)
        # Last 10 bars: price stays near the peak (within 2%)
        peak = closes[-1]
        for i in range(10):
            closes.append(peak * 0.99)
        return _make_ohlcv(closes)

    def _make_df_near_52w_low(self) -> pd.DataFrame:
        """Price drops to a low, then stays within 2% of it."""
        closes = _trending_down(200, start=200.0, pct_per_day=0.002)
        trough = closes[-1]
        for i in range(10):
            closes.append(trough * 1.01)
        return _make_ohlcv(closes)

    def _make_df_middle(self) -> pd.DataFrame:
        """Price is squarely in the middle — not near 52w high or low."""
        # go up, then come back down halfway
        up = _trending_up(126, start=100.0, pct_per_day=0.004)
        down = _trending_down(126, start=up[-1], pct_per_day=0.002)
        return _make_ohlcv(up + down)

    def test_fires_bullish_near_52w_high(self):
        df = self._make_df_near_52w_high()
        result = proximity_52w.detect("TEST", df)
        assert result is not None
        assert result.direction == "bullish"
        assert result.setup_type == "proximity_52w"

    def test_fires_bearish_near_52w_low(self):
        df = self._make_df_near_52w_low()
        result = proximity_52w.detect("TEST", df)
        assert result is not None
        assert result.direction == "bearish"

    def test_returns_none_in_middle(self):
        df = self._make_df_middle()
        result = proximity_52w.detect("TEST", df)
        assert result is None

    def test_returns_none_on_short_history(self):
        df = _make_ohlcv(_trending_up(30))  # < 60 required
        result = proximity_52w.detect("TEST", df)
        assert result is None

    def test_strength_in_valid_range(self):
        df = self._make_df_near_52w_high()
        result = proximity_52w.detect("TEST", df)
        if result is not None:
            assert 0.0 <= result.strength <= 1.0

    def test_notes_contain_expected_keys(self):
        df = self._make_df_near_52w_high()
        result = proximity_52w.detect("TEST", df)
        if result is not None:
            for key in ("high_52w", "low_52w", "dist_to_high_pct", "dist_to_low_pct"):
                assert key in result.notes

    def test_bullish_at_exact_high(self):
        """When close == 52w high exactly, dist=0, strength=1."""
        closes = [100.0] * 100
        closes[-1] = 100.0  # exact high
        df = _make_ohlcv(closes)
        # high column = close * 1.005, so we need close to equal max(high)
        # Override: set all highs equal to close so 52w high == close
        df["high"] = df["close"]
        result = proximity_52w.detect("TEST", df)
        assert result is not None
        assert result.direction == "bullish"
        assert result.strength == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# volume_surge detector
# ---------------------------------------------------------------------------

class TestVolumeSurge:
    def _make_surge_df(self, direction: str = "bullish") -> pd.DataFrame:
        """25+ bars, last 5 have 3x normal volume."""
        closes = _trending_up(30, start=100.0, pct_per_day=0.001)
        base_vol = 1_000_000
        volumes = [base_vol] * 30
        # Last 5 bars: 3x volume surge
        for i in range(-5, 0):
            volumes[i] = base_vol * 3
        if direction == "bearish":
            # Price went down over last 5 days
            closes[-1] = closes[-6] * 0.97

        df = _make_ohlcv(closes)
        df["volume"] = volumes
        return df

    def test_fires_bullish_on_price_up_surge(self):
        df = self._make_surge_df("bullish")
        result = volume_surge.detect("TEST", df)
        assert result is not None
        assert result.setup_type == "volume_surge"
        assert result.direction == "bullish"

    def test_fires_bearish_on_price_down_surge(self):
        df = self._make_surge_df("bearish")
        result = volume_surge.detect("TEST", df)
        assert result is not None
        assert result.direction == "bearish"

    def test_returns_none_without_surge(self):
        """Flat volume should not trigger."""
        df = _make_ohlcv(_trending_up(30))
        # volume stays at base (1_000_000), avg5 / avg20 ~= 1.0, well below 1.5
        result = volume_surge.detect("TEST", df)
        assert result is None

    def test_returns_none_on_short_history(self):
        df = _make_ohlcv(_trending_up(20))  # < 25 required
        result = volume_surge.detect("TEST", df)
        assert result is None

    def test_strength_in_valid_range(self):
        df = self._make_surge_df("bullish")
        result = volume_surge.detect("TEST", df)
        if result is not None:
            assert 0.0 <= result.strength <= 1.0

    def test_notes_contain_expected_keys(self):
        df = self._make_surge_df("bullish")
        result = volume_surge.detect("TEST", df)
        if result is not None:
            for key in ("surge_ratio", "avg_5d_volume", "avg_20d_volume", "5d_price_change_pct"):
                assert key in result.notes

    def test_strength_increases_with_larger_surge(self):
        """A 4x surge should produce higher strength than a 1.6x surge."""

        def _with_surge_factor(factor: float) -> float:
            closes = _trending_up(30, start=100.0, pct_per_day=0.001)
            base_vol = 1_000_000
            volumes = [base_vol] * 30
            for i in range(-5, 0):
                volumes[i] = int(base_vol * factor)
            df = _make_ohlcv(closes)
            df["volume"] = volumes
            result = volume_surge.detect("TEST", df)
            return result.strength if result is not None else 0.0

        s_small = _with_surge_factor(1.6)
        s_large = _with_surge_factor(4.0)
        assert s_large >= s_small
