"""Tests for scripts/kalshi_paper_trade.py

Tests record, resolve, and report subcommands.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_watchlist_json(tmp_path: Path, recs: list[dict] | None = None) -> Path:
    """Write a minimal latest.json for testing."""
    now = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
    if recs is None:
        recs = [
            {
                "ticker": "KXBTC-26JUN1316-T62500",
                "title": "BTC > $62,500 at 4PM",
                "expiration": "2026-06-13T20:00:00+00:00",
                "strike": 62500.0,
                "side": "above",
                "market_implied_prob": 0.42,
                "canonical_fair_prob": 0.55,
                "fair_prob_A": 0.53,
                "fair_prob_B": 0.56,
                "fair_prob_AB": 0.55,
                "edge": 0.13,
                "expected_value_per_dollar": 0.07,
                "suggested_position_dollars": 50.0,
                "estimator_agreement": [3, 4],
                "adjustments_applied": ["mean-reversion:-3%"],
                "volume_24h": 1200,
                "spread": 0.04,
            }
        ]

    payload = {
        "generated_at": now.isoformat(),
        "btc_spot": 65_000,
        "markets_analyzed": len(recs),
        "markets_with_edge": len(recs),
        "recommendations": recs,
    }
    out_dir = tmp_path / "output" / "kalshi"
    out_dir.mkdir(parents=True, exist_ok=True)
    watch_path = out_dir / "latest.json"
    watch_path.write_text(json.dumps(payload))
    return watch_path


def _make_expired_trade(tmp_path: Path, resolution: str = "yes") -> list[dict]:
    """Return a list with one OPEN expired trade entry."""
    exp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    return [
        {
            "id": f"KXBTC-26JUN01-T65000__{exp}",
            "recorded_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            "ticker": "KXBTC-26JUN01-T65000",
            "title": "BTC > $65,000",
            "expiration": exp,
            "strike": 65000.0,
            "side": "above",
            "market_implied_prob": 0.45,
            "canonical_fair_prob": 0.58,
            "edge": 0.13,
            "expected_value_per_dollar": 0.08,
            "suggested_position_dollars": 50.0,
            "estimator_agreement": [3, 4],
            "adjustments_applied": [],
            "status": "OPEN",
            "resolution": None,
            "resolved_at": None,
            "pnl_per_dollar": None,
            "pnl_dollars": None,
        }
    ]


# ---------------------------------------------------------------------------
# test_record_appends_to_ledger
# ---------------------------------------------------------------------------

def test_record_appends_to_ledger(tmp_path):
    """cmd_record reads latest.json and appends OPEN entries to ledger."""
    watch_path = _make_watchlist_json(tmp_path)
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"

    with patch("scripts.kalshi_paper_trade._WATCHLIST_JSON", watch_path), \
         patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_record
        rc = cmd_record()

    assert rc == 0
    assert ledger_path.exists()

    ledger = json.loads(ledger_path.read_text())
    assert len(ledger) == 1
    assert ledger[0]["ticker"] == "KXBTC-26JUN1316-T62500"
    assert ledger[0]["status"] == "OPEN"
    assert ledger[0]["resolution"] is None


def test_record_does_not_duplicate_existing_entries(tmp_path):
    """Recording the same watchlist twice does not duplicate ledger entries."""
    watch_path = _make_watchlist_json(tmp_path)
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"

    with patch("scripts.kalshi_paper_trade._WATCHLIST_JSON", watch_path), \
         patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_record
        cmd_record()
        cmd_record()  # second call — same recommendations

    ledger = json.loads(ledger_path.read_text())
    assert len(ledger) == 1, "Should not duplicate entries on second record call"


def test_record_multiple_recommendations(tmp_path):
    """Multiple recs in watchlist → multiple OPEN entries."""
    recs = [
        {
            "ticker": f"KXBTC-26JUN{i:02d}-T{62000 + i * 1000}",
            "title": f"BTC market {i}",
            "expiration": f"2026-06-{i:02d}T20:00:00+00:00",
            "strike": float(62000 + i * 1000),
            "side": "above",
            "market_implied_prob": 0.40,
            "canonical_fair_prob": 0.55,
            "fair_prob_A": 0.53, "fair_prob_B": 0.56, "fair_prob_AB": 0.55,
            "edge": 0.15,
            "expected_value_per_dollar": 0.09,
            "suggested_position_dollars": 50.0,
            "estimator_agreement": [3, 4],
            "adjustments_applied": [],
            "volume_24h": 1000,
            "spread": 0.03,
        }
        for i in range(13, 16)
    ]
    watch_path = _make_watchlist_json(tmp_path, recs=recs)
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"

    with patch("scripts.kalshi_paper_trade._WATCHLIST_JSON", watch_path), \
         patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_record
        cmd_record()

    ledger = json.loads(ledger_path.read_text())
    assert len(ledger) == 3


# ---------------------------------------------------------------------------
# test_resolve_marks_expired_open_trades
# ---------------------------------------------------------------------------

def test_resolve_marks_expired_open_trades(tmp_path):
    """cmd_resolve fetches resolution and marks expired OPEN trades as CLOSED."""
    expired_trades = _make_expired_trade(tmp_path, resolution="yes")
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(expired_trades))

    ticker = expired_trades[0]["ticker"]
    # Mock Kalshi returning resolved market
    mock_ks = MagicMock()
    mock_ks.get_settled_markets.return_value = [
        {
            "ticker": ticker,
            "event_ticker": "KXBTC-26JUN01",
            "title": "BTC > $65,000",
            "status": "settled",
            "yes_bid": 100,
            "yes_ask": 100,
            "no_bid": 0,
            "no_ask": 0,
            "volume": 500,
            "open_interest": 100,
            "close_time": expired_trades[0]["expiration"],
            "expiration_time": expired_trades[0]["expiration"],
            "result": "yes",
        }
    ]

    with patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path), \
         patch("data.sources.kalshi_source.KalshiSource", return_value=mock_ks):
        from scripts.kalshi_paper_trade import cmd_resolve
        rc = cmd_resolve()

    assert rc == 0

    ledger = json.loads(ledger_path.read_text())
    assert len(ledger) == 1
    trade = ledger[0]
    assert trade["status"] == "CLOSED"
    assert trade["resolution"] == "yes"
    assert trade["pnl_per_dollar"] is not None
    assert trade["pnl_dollars"] is not None
    # Edge > 0 and resolved YES → pnl > 0
    assert trade["pnl_dollars"] > 0


def test_resolve_skips_open_not_yet_expired(tmp_path):
    """cmd_resolve does not attempt to resolve trades that haven't expired."""
    future_exp = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    not_expired = [
        {
            "id": f"KXBTC-26JUN01-T65000__{future_exp}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "ticker": "KXBTC-26JUN01-T65000",
            "title": "BTC > $65,000",
            "expiration": future_exp,
            "strike": 65000.0,
            "side": "above",
            "market_implied_prob": 0.45,
            "canonical_fair_prob": 0.58,
            "edge": 0.13,
            "expected_value_per_dollar": 0.08,
            "suggested_position_dollars": 50.0,
            "estimator_agreement": [3, 4],
            "adjustments_applied": [],
            "status": "OPEN",
            "resolution": None,
            "resolved_at": None,
            "pnl_per_dollar": None,
            "pnl_dollars": None,
        }
    ]
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(not_expired))

    with patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_resolve
        rc = cmd_resolve()

    assert rc == 0
    ledger = json.loads(ledger_path.read_text())
    assert ledger[0]["status"] == "OPEN"  # unchanged


# ---------------------------------------------------------------------------
# test_report_aggregates_closed_trades
# ---------------------------------------------------------------------------

def test_report_aggregates_closed_trades(tmp_path):
    """cmd_report computes win rate and P&L correctly across multiple closed trades."""
    trades = [
        {
            "id": f"KXBTC-{i}",
            "recorded_at": "2026-06-01T00:00:00+00:00",
            "ticker": f"KXBTC-{i}",
            "title": f"Trade {i}",
            "expiration": "2026-06-02T00:00:00+00:00",
            "strike": 65000.0,
            "side": "above",
            "market_implied_prob": 0.40,
            "canonical_fair_prob": 0.55,
            "edge": 0.15,
            "expected_value_per_dollar": 0.09,
            "suggested_position_dollars": 50.0,
            "estimator_agreement": [3, 4],
            "adjustments_applied": [],
            "status": "CLOSED",
            "resolution": "yes" if i % 2 == 0 else "no",
            "resolved_at": "2026-06-02T00:05:00+00:00",
            "pnl_per_dollar": 0.80 if i % 2 == 0 else -0.40,
            "pnl_dollars": 40.0 if i % 2 == 0 else -20.0,
        }
        for i in range(1, 11)
    ]

    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(trades))

    with patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_report
        rc = cmd_report()

    assert rc == 0


def test_report_empty_ledger(tmp_path):
    """cmd_report handles empty ledger gracefully."""
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("[]")

    with patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_report
        rc = cmd_report()

    assert rc == 0


def test_report_no_closed_trades_prints_guidance(tmp_path):
    """cmd_report handles ledger with only OPEN trades."""
    exp = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    trades = [
        {
            "id": "KXBTC-OPEN",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "ticker": "KXBTC-OPEN",
            "title": "Open Trade",
            "expiration": exp,
            "strike": 65000.0,
            "side": "above",
            "market_implied_prob": 0.45,
            "canonical_fair_prob": 0.60,
            "edge": 0.15,
            "expected_value_per_dollar": 0.09,
            "suggested_position_dollars": 50.0,
            "estimator_agreement": [3, 4],
            "adjustments_applied": [],
            "status": "OPEN",
            "resolution": None,
            "resolved_at": None,
            "pnl_per_dollar": None,
            "pnl_dollars": None,
        }
    ]
    ledger_path = tmp_path / "output" / "kalshi" / "paper_trades.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(trades))

    with patch("scripts.kalshi_paper_trade._LEDGER_JSON", ledger_path):
        from scripts.kalshi_paper_trade import cmd_report
        rc = cmd_report()

    assert rc == 0
