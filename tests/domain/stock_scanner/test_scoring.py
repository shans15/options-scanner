"""Tests for domain/stock_scanner/scoring.py"""
from __future__ import annotations

import pytest

from domain.stock_scanner.scoring import score_candidate
from domain.stock_scanner.types import StockSetup, StockCandidate


def _make_setup(setup_type: str, direction: str, strength: float = 0.9) -> StockSetup:
    return StockSetup(
        ticker="TEST",
        setup_type=setup_type,
        direction=direction,
        strength=strength,
        spot=100.0,
        notes={},
    )


def _snapshot() -> dict:
    return {"close": 100.0, "volume_5d_avg": 1_000_000, "high_52w": 120.0, "low_52w": 80.0}


class TestScoreCandidate:
    def test_all_bullish_setups_returns_strong_bull(self):
        setups = [
            _make_setup("sma_stack", "bullish", 1.0),
            _make_setup("compression", "bullish", 1.0),
            _make_setup("proximity_52w", "bullish", 1.0),
            _make_setup("volume_surge", "bullish", 1.0),
        ]
        c = score_candidate("TEST", 100.0, setups, _snapshot())
        assert c.label == "STRONG_BULL"
        assert c.net_score > 60
        assert c.bullish_score > 0
        assert c.bearish_score == 0.0

    def test_all_bearish_setups_returns_strong_bear(self):
        setups = [
            _make_setup("sma_stack", "bearish", 1.0),
            _make_setup("compression", "bearish", 1.0),
            _make_setup("proximity_52w", "bearish", 1.0),
            _make_setup("volume_surge", "bearish", 1.0),
        ]
        c = score_candidate("TEST", 100.0, setups, _snapshot())
        assert c.label == "STRONG_BEAR"
        assert c.net_score < -60
        assert c.bearish_score > 0
        assert c.bullish_score == 0.0

    def test_empty_setups_returns_neutral(self):
        c = score_candidate("TEST", 100.0, [], _snapshot())
        assert c.label == "NEUTRAL"
        assert c.bullish_score == 0.0
        assert c.bearish_score == 0.0
        assert c.net_score == 0.0

    def test_mixed_setups_net_score_is_difference(self):
        setups = [
            _make_setup("sma_stack", "bullish", 1.0),   # weight 0.30 -> bull 30
            _make_setup("proximity_52w", "bearish", 1.0),  # weight 0.25 -> bear 25
        ]
        c = score_candidate("TEST", 100.0, setups, _snapshot())
        assert c.bullish_score == pytest.approx(30.0, abs=0.1)
        assert c.bearish_score == pytest.approx(25.0, abs=0.1)
        assert c.net_score == pytest.approx(5.0, abs=0.2)
        assert c.label == "NEUTRAL"

    def test_bull_label_for_moderate_bullish_net(self):
        # net >= 30 -> BULL
        setups = [
            _make_setup("sma_stack", "bullish", 1.0),       # 30
            _make_setup("compression", "bullish", 1.0),     # 25
        ]
        c = score_candidate("TEST", 100.0, setups, _snapshot())
        assert c.net_score >= 30
        assert c.label in ("BULL", "STRONG_BULL")

    def test_bear_label_for_moderate_bearish_net(self):
        setups = [
            _make_setup("sma_stack", "bearish", 1.0),       # -30
            _make_setup("compression", "bearish", 1.0),     # -25
        ]
        c = score_candidate("TEST", 100.0, setups, _snapshot())
        assert c.net_score <= -30
        assert c.label in ("BEAR", "STRONG_BEAR")

    def test_result_is_stock_candidate_type(self):
        c = score_candidate("AAPL", 150.0, [], _snapshot())
        assert isinstance(c, StockCandidate)
        assert c.ticker == "AAPL"
        assert c.spot == 150.0

    def test_scores_scaled_by_strength(self):
        """Half-strength should produce half the score of full-strength."""
        full_setups = [_make_setup("sma_stack", "bullish", 1.0)]
        half_setups = [_make_setup("sma_stack", "bullish", 0.5)]
        c_full = score_candidate("T", 100.0, full_setups, _snapshot())
        c_half = score_candidate("T", 100.0, half_setups, _snapshot())
        assert c_full.bullish_score == pytest.approx(c_half.bullish_score * 2, abs=0.5)

    def test_snapshot_preserved(self):
        snap = {"close": 99.5, "volume_5d_avg": 500_000, "high_52w": 110.0, "low_52w": 85.0}
        c = score_candidate("TEST", 99.5, [], snap)
        assert c.snapshot == snap
