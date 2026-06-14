"""Tests for scripts/aplus_paper_trade.py

Forward-test infrastructure for the A+ Confluence Scorer.

Coverage:
    1. test_ingest_opens_new_position
    2. test_ingest_skips_duplicate
    3. test_ingest_skips_b_plus_and_below
    4. test_mark_updates_pnl
    5. test_close_triggers_at_target
    6. test_close_triggers_at_stop
    7. test_close_triggers_at_expiration
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.aplus_paper_trade import (
    cmd_close,
    cmd_ingest,
    cmd_mark,
    cmd_summary,
    _make_position_id,
    _wilson_ci,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _minimal_scan_json(
    *,
    grade: str = "A",
    ticker: str = "MSFT",
    option_type: str = "call",
    strike: float = 510.0,
    expiration: str = "2026-07-25",
    dte: int = 41,
    spot_price: float = 505.10,
    bid: float = 0.45,
    ask: float = 0.55,
    mid: float = 0.50,
    account_size: float = 1000.0,
    sizing_pct: float = 0.075,
    composite_score: float = 82.4,
    strategy: str = "long_call",
    structure: str = "long_premium",
    timestamp: str = "2026-06-14",
) -> dict:
    """Return a minimal latest.json payload for one candidate.

    Default mid=0.50 so that with account_size=1000 and sizing_pct=0.075
    (max_risk=$75), floor(75 / (0.50*100)) = floor(75/50) = 1 contract fits.
    """
    max_risk_dollars = account_size * sizing_pct
    return {
        "timestamp": timestamp,
        "account_size": account_size,
        "graded_candidates": [
            {
                "ticker": ticker,
                "strategy": strategy,
                "grade": grade,
                "composite_score": composite_score,
                "category_scores": {
                    "technical": 9.0,
                    "vol_vix": 8.5,
                    "catalyst": 7.0,
                    "macro_breadth": 8.0,
                    "liquidity": 9.0,
                },
                "feature_scores": {},
                "structure": structure,
                "structure_rationale": "momentum + compression",
                "sizing_pct": sizing_pct,
                "max_risk_dollars": max_risk_dollars,
                "contract": {
                    "ticker": ticker,
                    "option_type": option_type,
                    "strike": strike,
                    "expiration": expiration,
                    "dte": dte,
                    "spot_price": spot_price,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "volume": 1200,
                    "open_interest": 5000,
                    "implied_volatility": 0.18,
                },
            }
        ],
    }


def _write_scan(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "latest.json"
    p.write_text(json.dumps(payload))
    return p


def _store_path(tmp_path: Path) -> Path:
    return tmp_path / "paper_trades.json"


def _write_store(tmp_path: Path, store: dict) -> Path:
    sp = _store_path(tmp_path)
    sp.write_text(json.dumps(store))
    return sp


def _empty_store() -> dict:
    return {"last_updated": None, "positions": []}


def _one_open_position(
    *,
    position_id: str = "2026-06-14-MSFT-call-510-2026-07-25",
    ticker: str = "MSFT",
    option_type: str = "call",
    strike: float = 510.0,
    expiration: str = "2026-07-25",
    entry_mid: float = 8.75,
    contracts_count: int = 1,
    pnl_pct: float = 0.0,
    pnl_dollars: float = 0.0,
    last_mark_mid: float | None = None,
    grade: str = "A",
    opened_at: str = "2026-06-14",
) -> dict:
    cost = round(contracts_count * 100 * entry_mid, 2)
    return {
        "id": position_id,
        "opened_at": opened_at,
        "ticker": ticker,
        "strategy": "long_call",
        "structure": "long_premium",
        "grade": grade,
        "composite_score": 82.4,
        "category_scores": {},
        "contract": {
            "option_type": option_type,
            "strike": strike,
            "expiration": expiration,
            "dte_at_entry": 41,
        },
        "entry_mid": entry_mid,
        "contracts_count": contracts_count,
        "entry_cost_dollars": cost,
        "max_risk_dollars": 75.0,
        "sizing_pct": 0.075,
        "status": "open",
        "last_mark_mid": last_mark_mid if last_mark_mid is not None else entry_mid,
        "last_marked_at": opened_at,
        "current_pnl_dollars": pnl_dollars,
        "current_pnl_pct": pnl_pct,
        "closed_at": None,
        "close_mid": None,
        "close_reason": None,
        "final_pnl_dollars": None,
        "final_pnl_pct": None,
    }


def _make_raw_contract(
    *,
    ticker: str = "MSFT",
    option_type: str = "call",
    strike: float = 510.0,
    expiration: str = "2026-07-25",
    bid: float = 8.50,
    ask: float = 9.00,
) -> SimpleNamespace:
    """Return a minimal object that looks like a RawContract."""
    return SimpleNamespace(
        ticker=ticker,
        option_type=option_type,
        strike=strike,
        expiration=date.fromisoformat(expiration),
        bid=bid,
        ask=ask,
        volume=100,
        open_interest=500,
        implied_volatility=0.28,
        dte=30,
        spot_price=505.0,
    )


# ---------------------------------------------------------------------------
# 1. test_ingest_opens_new_position
# ---------------------------------------------------------------------------

def test_ingest_opens_new_position(tmp_path):
    """Fresh store: ingest with one A-grade candidate opens exactly one position."""
    scan_payload = _minimal_scan_json(grade="A")
    scan_p = _write_scan(tmp_path, scan_payload)
    store_p = _store_path(tmp_path)

    today = date(2026, 6, 14)
    rc = cmd_ingest(scan_p, store_p, today=today)

    assert rc == 0
    assert store_p.exists()

    store = json.loads(store_p.read_text())
    positions = store["positions"]
    assert len(positions) == 1

    pos = positions[0]
    assert pos["ticker"] == "MSFT"
    assert pos["grade"] == "A"
    assert pos["status"] == "open"
    assert pos["entry_mid"] == pytest.approx(0.50)
    assert pos["contracts_count"] >= 1
    # entry_cost_dollars must not exceed max_risk_dollars
    assert pos["entry_cost_dollars"] <= pos["max_risk_dollars"] + 0.01

    expected_id = _make_position_id("2026-06-14", "MSFT", "call", 510.0, "2026-07-25")
    assert pos["id"] == expected_id


# ---------------------------------------------------------------------------
# 2. test_ingest_skips_duplicate
# ---------------------------------------------------------------------------

def test_ingest_skips_duplicate(tmp_path):
    """Re-running ingest with the same signal does not create a second position."""
    scan_payload = _minimal_scan_json(grade="A")
    scan_p = _write_scan(tmp_path, scan_payload)
    store_p = _store_path(tmp_path)

    today = date(2026, 6, 14)
    cmd_ingest(scan_p, store_p, today=today)
    cmd_ingest(scan_p, store_p, today=today)  # second call — same signal

    store = json.loads(store_p.read_text())
    assert len(store["positions"]) == 1, "Duplicate should not be appended"


# ---------------------------------------------------------------------------
# 3. test_ingest_skips_b_plus_and_below
# ---------------------------------------------------------------------------

def test_ingest_skips_b_plus_and_below(tmp_path):
    """Candidates with grade B+, B, F are not opened."""
    for bad_grade in ("B+", "B", "F"):
        sub_path = tmp_path / bad_grade
        sub_path.mkdir()
        scan_payload = _minimal_scan_json(grade=bad_grade)
        scan_p = _write_scan(sub_path, scan_payload)
        store_p = sub_path / "paper_trades.json"

        rc = cmd_ingest(scan_p, store_p, today=date(2026, 6, 14))
        assert rc == 0

        store = json.loads(store_p.read_text())
        assert store["positions"] == [], f"Grade {bad_grade} should not create a position"


# ---------------------------------------------------------------------------
# 4. test_mark_updates_pnl
# ---------------------------------------------------------------------------

def test_mark_updates_pnl(tmp_path):
    """mark fetches chain; contract moved from 8.75 to 10.00 → pnl updates correctly."""
    pos = _one_open_position(entry_mid=8.75, contracts_count=1, last_mark_mid=8.75)
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": [pos]})

    # Chain returns contract priced at bid=9.50, ask=10.50 → mid=10.00
    mock_contract = _make_raw_contract(bid=9.50, ask=10.50)

    with patch("data.fallback.fetch_with_fallback", return_value=[mock_contract]):
        rc = cmd_mark(store_p, today=date(2026, 6, 15))

    assert rc == 0
    store = json.loads(store_p.read_text())
    p = store["positions"][0]

    assert p["last_mark_mid"] == pytest.approx(10.00)
    expected_pnl_dollars = (10.00 - 8.75) * 100 * 1  # = 125.0
    assert p["current_pnl_dollars"] == pytest.approx(expected_pnl_dollars)
    expected_pnl_pct = (10.00 - 8.75) / 8.75 * 100  # ≈ 14.29%
    assert p["current_pnl_pct"] == pytest.approx(expected_pnl_pct, rel=0.01)


# ---------------------------------------------------------------------------
# 5. test_close_triggers_at_target
# ---------------------------------------------------------------------------

def test_close_triggers_at_target(tmp_path):
    """Position with current_pnl_pct >= 100.0 is closed with reason='target'."""
    pos = _one_open_position(pnl_pct=105.0, last_mark_mid=17.75)
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": [pos]})

    rc = cmd_close(store_p, target_pct=100.0, stop_pct=-50.0, today=date(2026, 6, 20))

    assert rc == 0
    store = json.loads(store_p.read_text())
    p = store["positions"][0]

    assert p["status"] == "closed"
    assert p["close_reason"] == "target"
    assert p["final_pnl_dollars"] is not None
    assert p["closed_at"] == "2026-06-20"


# ---------------------------------------------------------------------------
# 6. test_close_triggers_at_stop
# ---------------------------------------------------------------------------

def test_close_triggers_at_stop(tmp_path):
    """Position with current_pnl_pct <= -50.0 is closed with reason='stop'."""
    pos = _one_open_position(pnl_pct=-60.0, last_mark_mid=3.50)
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": [pos]})

    rc = cmd_close(store_p, target_pct=100.0, stop_pct=-50.0, today=date(2026, 6, 20))

    assert rc == 0
    store = json.loads(store_p.read_text())
    p = store["positions"][0]

    assert p["status"] == "closed"
    assert p["close_reason"] == "stop"
    # P&L should be negative (lost money)
    assert p["final_pnl_dollars"] < 0


# ---------------------------------------------------------------------------
# 7. test_close_triggers_at_expiration
# ---------------------------------------------------------------------------

def test_close_triggers_at_expiration(tmp_path):
    """Position where today >= expiration is closed with reason='expiration'."""
    # Expiry was yesterday
    pos = _one_open_position(expiration="2026-06-13")
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": [pos]})

    rc = cmd_close(store_p, target_pct=100.0, stop_pct=-50.0, today=date(2026, 6, 14))

    assert rc == 0
    store = json.loads(store_p.read_text())
    p = store["positions"][0]

    assert p["status"] == "closed"
    assert p["close_reason"] == "expiration"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_ingest_a_plus_grade_also_opens(tmp_path):
    """A+ grade (highest) is also ingested."""
    scan_payload = _minimal_scan_json(grade="A+", sizing_pct=0.125)
    scan_p = _write_scan(tmp_path, scan_payload)
    store_p = _store_path(tmp_path)

    rc = cmd_ingest(scan_p, store_p, today=date(2026, 6, 14))
    assert rc == 0

    store = json.loads(store_p.read_text())
    assert len(store["positions"]) == 1
    assert store["positions"][0]["grade"] == "A+"


def test_ingest_missing_scan_file_returns_nonzero(tmp_path):
    """ingest returns non-zero exit code when scan file doesn't exist."""
    store_p = _store_path(tmp_path)
    rc = cmd_ingest(tmp_path / "nonexistent.json", store_p, today=date(2026, 6, 14))
    assert rc != 0


def test_mark_graceful_when_no_matching_contract(tmp_path):
    """mark logs a warning but doesn't crash if the contract is missing from chain."""
    pos = _one_open_position()
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": [pos]})

    # Chain returns a contract with a different strike — no match
    wrong_contract = _make_raw_contract(strike=999.0)

    with patch("data.fallback.fetch_with_fallback", return_value=[wrong_contract]):
        rc = cmd_mark(store_p, today=date(2026, 6, 15))

    assert rc == 0
    store = json.loads(store_p.read_text())
    # last_mark_mid should be unchanged (stale)
    assert store["positions"][0]["last_mark_mid"] == pos["entry_mid"]


def test_mark_graceful_on_data_fetch_error(tmp_path):
    """mark logs per-position error but doesn't crash on DataFetchError."""
    from data.fallback import DataFetchError

    pos = _one_open_position()
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": [pos]})

    with patch("data.fallback.fetch_with_fallback", side_effect=DataFetchError("all sources failed")):
        rc = cmd_mark(store_p, today=date(2026, 6, 15))

    assert rc == 0  # must not crash


def test_close_no_open_positions_exits_cleanly(tmp_path):
    """close with no open positions prints a message and returns 0."""
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": []})
    rc = cmd_close(store_p, target_pct=100.0, stop_pct=-50.0, today=date(2026, 6, 20))
    assert rc == 0


def test_summary_empty_store(tmp_path):
    """summary handles an empty store gracefully."""
    store_p = _write_store(tmp_path, {"last_updated": None, "positions": []})
    rc = cmd_summary(store_p)
    assert rc == 0


def test_summary_wilson_ci_prints_when_enough_closed(tmp_path, capsys):
    """summary prints Wilson CI line when there are ≥5 closed positions."""
    positions = []
    for i in range(6):
        p = _one_open_position(
            position_id=f"2026-06-{i+1:02d}-MSFT-call-510-2026-07-25",
            opened_at=f"2026-06-{i+1:02d}",
        )
        # Override to closed state
        p["status"] = "closed"
        p["final_pnl_dollars"] = 50.0 if i % 2 == 0 else -30.0
        p["final_pnl_pct"] = 57.14 if i % 2 == 0 else -34.29
        p["close_reason"] = "target" if i % 2 == 0 else "stop"
        p["closed_at"] = f"2026-06-{i+10:02d}"
        positions.append(p)

    store_p = _write_store(tmp_path, {"last_updated": None, "positions": positions})
    rc = cmd_summary(store_p)
    assert rc == 0

    captured = capsys.readouterr()
    assert "Wilson" in captured.out


def test_wilson_ci_bounds():
    """Wilson CI is within [0, 1] and lower < upper."""
    lo, hi = _wilson_ci(7, 10)
    assert 0.0 <= lo < hi <= 1.0


def test_contracts_count_floored_correctly(tmp_path):
    """contracts_count = floor(max_risk / (entry_mid * 100)); entry_cost <= max_risk."""
    # mid = 8.75, max_risk = 75  → floor(75 / 875) = 0 → too expensive → skip
    scan_payload = _minimal_scan_json(grade="A", mid=8.75, sizing_pct=0.075, account_size=1000.0)
    # max_risk = 75, entry_mid=8.75, cost per contract = 875 → floor(75/875)=0 → skip
    scan_p = _write_scan(tmp_path, scan_payload)
    store_p = _store_path(tmp_path)

    # With default account_size=1000, sizing_pct=0.075 → max_risk=75
    # entry_mid=8.75 → floor(75/875) = 0 → should be skipped (too expensive)
    rc = cmd_ingest(scan_p, store_p, today=date(2026, 6, 14))
    assert rc == 0
    store = json.loads(store_p.read_text())
    # Position should be skipped because floor is 0
    assert len(store["positions"]) == 0


def test_contracts_count_with_larger_account(tmp_path):
    """With a larger account, contracts_count > 0 and entry_cost <= max_risk."""
    # entry_mid = 8.75, account_size = 100_000, sizing_pct = 0.075
    # max_risk = 7500, cost per contract = 875, floor(7500/875) = 8
    scan_payload = _minimal_scan_json(
        grade="A",
        mid=8.75,
        sizing_pct=0.075,
        account_size=100_000.0,
    )
    scan_p = _write_scan(tmp_path, scan_payload)
    store_p = _store_path(tmp_path)

    rc = cmd_ingest(scan_p, store_p, today=date(2026, 6, 14))
    assert rc == 0
    store = json.loads(store_p.read_text())
    assert len(store["positions"]) == 1

    pos = store["positions"][0]
    expected_count = math.floor(7500 / (8.75 * 100))
    assert pos["contracts_count"] == expected_count
    assert pos["entry_cost_dollars"] <= 7500.0 + 0.01
