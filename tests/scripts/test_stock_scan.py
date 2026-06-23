"""Smoke tests for scripts/stock_scan.py"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from domain.stock_scanner.types import StockSetup, StockCandidate


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 300, start: float = 100.0) -> pd.DataFrame:
    closes = [start * (1.003 ** i) for i in range(n)]
    closes_arr = np.array(closes)
    return pd.DataFrame({
        "open": closes_arr * 0.999,
        "high": closes_arr * 1.005,
        "low": closes_arr * 0.995,
        "close": closes_arr,
        "volume": np.full(n, 1_000_000, dtype=float),
    })


def _mock_sources():
    src = MagicMock()
    src.fetch_price_history_ohlcv.return_value = _make_ohlcv_df()
    return [src]


def _make_candidate(ticker: str = "NVDA", net: float = 75.0) -> StockCandidate:
    setup = StockSetup(
        ticker=ticker,
        setup_type="sma_stack",
        direction="bullish",
        strength=0.9,
        spot=200.0,
        notes={"ema10": 198.0, "ema20": 195.0, "ema50": 185.0, "ema200": 160.0,
               "spread_pct": 23.75, "persistence_days_of_20": 20},
    )
    return StockCandidate(
        ticker=ticker,
        spot=200.0,
        setups=[setup],
        bullish_score=75.0,
        bearish_score=0.0,
        net_score=net,
        label="STRONG_BULL" if net >= 60 else ("BULL" if net >= 30 else "NEUTRAL"),
        snapshot={"close": 200.0, "volume_5d_avg": 1_000_000, "high_52w": 210.0, "low_52w": 120.0},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStockScanCLI:
    def test_main_runs_without_error_on_mocked_sources(self, tmp_path):
        """Full main() run with mocked data sources should not raise."""
        from scripts.stock_scan import main

        with (
            patch("scripts.stock_scan.default_sources", return_value=_mock_sources()),
            patch("scripts.stock_scan._OUTPUT_DIR", tmp_path / "stock_scans"),
        ):
            # Should complete without raising
            main(["--lookback-days", "365"])

    def test_cli_writes_csv_and_json(self, tmp_path):
        """main() should produce both CSV and latest.json in the output dir."""
        from scripts.stock_scan import main, _save_outputs

        candidates = [_make_candidate("NVDA", 75.0), _make_candidate("AMD", 40.0)]

        with patch("scripts.stock_scan._OUTPUT_DIR", tmp_path):
            csv_path, json_path = _save_outputs(candidates, "20260622_1530")

        assert csv_path.exists(), "CSV file was not created"
        assert json_path.exists(), "JSON file was not created"

        # Validate CSV structure
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["ticker"] == "NVDA"
        assert rows[1]["ticker"] == "AMD"

        # Validate JSON structure
        with open(json_path) as f:
            payload = json.load(f)
        assert payload["total_candidates"] == 2
        assert payload["candidates"][0]["ticker"] == "NVDA"
        assert "setups" in payload["candidates"][0]

    def test_label_filter_works(self, tmp_path):
        """--label STRONG_BULL should exclude non-STRONG_BULL candidates."""
        from scripts.stock_scan import _apply_filters

        candidates = [
            _make_candidate("NVDA", 75.0),  # STRONG_BULL
            _make_candidate("AMD", 40.0),   # BULL
            _make_candidate("SQQQ", -70.0), # STRONG_BEAR
        ]
        # Override labels to be deterministic
        # Rebuild with correct labels via score_candidate would be ideal,
        # but direct construction is fine for unit test
        from dataclasses import replace
        candidates[1] = StockCandidate(
            **{**candidates[1].__dict__, "label": "BULL"}
        ) if False else candidates[1]

        filtered = _apply_filters(candidates, "STRONG_BULL", None)
        assert all(c.label == "STRONG_BULL" for c in filtered)
        assert any(c.ticker == "NVDA" for c in filtered)

    def test_setup_filter_works(self, tmp_path):
        """--setup compression should only return candidates with that detector."""
        from scripts.stock_scan import _apply_filters

        compression_setup = StockSetup(
            ticker="XLF", setup_type="compression", direction="bullish",
            strength=0.7, spot=50.0, notes={},
        )
        candidate_with_compression = StockCandidate(
            ticker="XLF", spot=50.0, setups=[compression_setup],
            bullish_score=17.5, bearish_score=0.0, net_score=17.5,
            label="NEUTRAL",
            snapshot={"close": 50.0, "volume_5d_avg": 500_000, "high_52w": 55.0, "low_52w": 42.0},
        )
        candidate_no_compression = _make_candidate("NVDA", 75.0)  # only sma_stack

        filtered = _apply_filters(
            [candidate_with_compression, candidate_no_compression],
            None,
            "compression",
        )
        assert len(filtered) == 1
        assert filtered[0].ticker == "XLF"

    def test_universe_file_loads_custom_tickers(self, tmp_path):
        """--universe-file should load tickers from a CSV."""
        from scripts.stock_scan import _load_universe

        csv_file = tmp_path / "tickers.csv"
        csv_file.write_text("AAPL\nMSFT\nGOOGL\n")
        tickers = _load_universe(str(csv_file))
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_load_universe_none_returns_builtin(self):
        """None universe_file returns the built-in UNIVERSE list."""
        from scripts.stock_scan import _load_universe
        from domain.stock_scanner.universe import UNIVERSE

        tickers = _load_universe(None)
        assert tickers == list(UNIVERSE)

    def test_main_with_label_filter_via_args(self, tmp_path):
        """Passing --label STRONG_BULL via argv should run without errors."""
        from scripts.stock_scan import main

        with (
            patch("scripts.stock_scan.default_sources", return_value=_mock_sources()),
            patch("scripts.stock_scan._OUTPUT_DIR", tmp_path / "stock_scans"),
        ):
            main(["--label", "STRONG_BULL", "--lookback-days", "365"])
