"""Paper trade tracker. Records watchlist signals, back-fills P&L on resolution.

CLI
---
    python3 -m scripts.kalshi_paper_trade record   # record latest watchlist signals
    python3 -m scripts.kalshi_paper_trade resolve  # check settled markets, compute P&L
    python3 -m scripts.kalshi_paper_trade report   # print summary of closed trades

State persisted in output/kalshi/paper_trades.json (append-only ledger).

Ledger schema (one entry per recommendation):
    {
      "id": "KXBTC-26JUN1316-T62500__2026-06-13T14:00:00+00:00",
      "recorded_at": "2026-06-13T14:05:00+00:00",
      "ticker": "KXBTC-26JUN1316-T62500",
      "title": "BTC > $62,500 at 4:00 PM ET",
      "expiration": "2026-06-13T16:00:00+00:00",
      "strike": 62500.0,
      "side": "above",
      "market_implied_prob": 0.42,
      "canonical_fair_prob": 0.55,
      "edge": 0.13,
      "expected_value_per_dollar": 0.09,
      "suggested_position_dollars": 50.0,
      "estimator_agreement": [3, 4],
      "adjustments_applied": ["mean-reversion:-3%"],
      "status": "OPEN",          # OPEN | CLOSED
      "resolution": null,        # "yes" | "no" when CLOSED
      "resolved_at": null,
      "pnl_per_dollar": null,
      "pnl_dollars": null
    }
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_WATCHLIST_JSON = _ROOT / "output" / "kalshi" / "latest.json"
_LEDGER_JSON = _ROOT / "output" / "kalshi" / "paper_trades.json"
_KALSHI_FEE_RATE = 0.07


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------

def _load_ledger() -> list[dict]:
    if not _LEDGER_JSON.exists():
        return []
    try:
        return json.loads(_LEDGER_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_ledger(trades: list[dict]) -> None:
    _LEDGER_JSON.parent.mkdir(parents=True, exist_ok=True)
    _LEDGER_JSON.write_text(json.dumps(trades, indent=2, default=str))


def _trade_id(ticker: str, expiration: str) -> str:
    return f"{ticker}__{expiration}"


def _pnl_per_dollar(edge: float, market_prob: float, resolved_yes: bool) -> float:
    """Realized P&L per dollar risked given resolution.

    If edge > 0: we bought YES at market_prob.
    If edge < 0: we bought NO (effectively at 1 - market_prob).
    """
    if edge > 0:  # BUY YES
        if resolved_yes:
            return (1.0 - market_prob) * (1.0 - _KALSHI_FEE_RATE)
        else:
            return -market_prob
    else:  # BUY NO
        no_price = 1.0 - market_prob
        if not resolved_yes:
            return market_prob * (1.0 - _KALSHI_FEE_RATE)
        else:
            return -no_price


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_record() -> int:
    """Read latest.json and append new OPEN entries to ledger."""
    if not _WATCHLIST_JSON.exists():
        print(f"ERROR: Watchlist not found at {_WATCHLIST_JSON}", file=sys.stderr)
        print("Run `python3 -m scripts.kalshi_btc_watchlist` first.", file=sys.stderr)
        return 1

    watchlist = json.loads(_WATCHLIST_JSON.read_text())
    recs = watchlist.get("recommendations", [])
    if not recs:
        print("No recommendations in latest watchlist. Nothing to record.")
        return 0

    ledger = _load_ledger()
    existing_ids = {t["id"] for t in ledger}

    recorded_at = datetime.now(timezone.utc).isoformat()
    new_count = 0

    for rec in recs:
        trade_id = _trade_id(rec["ticker"], rec["expiration"])
        if trade_id in existing_ids:
            continue  # already recorded

        entry = {
            "id": trade_id,
            "recorded_at": recorded_at,
            "ticker": rec["ticker"],
            "title": rec.get("title", ""),
            "expiration": rec["expiration"],
            "strike": rec["strike"],
            "side": rec["side"],
            "market_implied_prob": rec["market_implied_prob"],
            "canonical_fair_prob": rec["canonical_fair_prob"],
            "edge": rec["edge"],
            "expected_value_per_dollar": rec["expected_value_per_dollar"],
            "suggested_position_dollars": rec["suggested_position_dollars"],
            "estimator_agreement": rec.get("estimator_agreement", [0, 0]),
            "adjustments_applied": rec.get("adjustments_applied", []),
            "status": "OPEN",
            "resolution": None,
            "resolved_at": None,
            "pnl_per_dollar": None,
            "pnl_dollars": None,
        }
        ledger.append(entry)
        new_count += 1

    _save_ledger(ledger)
    print(f"Recorded {new_count} new open trade(s). Ledger now has {len(ledger)} total entries.")
    return 0


def cmd_resolve() -> int:
    """For OPEN trades past expiration, fetch resolution from Kalshi, mark CLOSED."""
    ledger = _load_ledger()
    if not ledger:
        print("Ledger is empty. Run `record` first.")
        return 0

    now = datetime.now(timezone.utc)
    open_trades = [t for t in ledger if t["status"] == "OPEN"]
    expired = [
        t for t in open_trades
        if datetime.fromisoformat(t["expiration"]) <= now
    ]

    if not expired:
        print(f"No expired OPEN trades to resolve (checked {len(open_trades)} open trades).")
        return 0

    print(f"Resolving {len(expired)} expired trade(s) …")

    from data.sources.kalshi_source import KalshiSource
    ks = KalshiSource()

    resolved_count = 0
    error_count = 0

    # Build lookup: ticker → ledger entry
    id_to_entry = {t["id"]: t for t in ledger}

    for trade in expired:
        ticker = trade["ticker"]
        try:
            # Fetch settled market info for this ticker
            # Use a narrow window around expiration
            exp_dt = datetime.fromisoformat(trade["expiration"])
            start_ms = int((exp_dt.timestamp() - 3600) * 1000)
            end_ms = int((exp_dt.timestamp() + 3600) * 1000)
            settled = ks.get_settled_markets(start_ms, end_ms)
            raw = next((m for m in settled if m.get("ticker") == ticker), None)

            if raw is None:
                print(f"  WARN: {ticker} not found in settled markets — may still be open or ticker mismatch")
                error_count += 1
                continue

            resolution = (raw.get("result") or raw.get("resolution") or "").lower()
            if resolution not in ("yes", "no"):
                print(f"  WARN: {ticker} resolution unclear: {resolution!r}")
                error_count += 1
                continue

            resolved_yes = resolution == "yes"
            pnl_per_dollar = _pnl_per_dollar(
                trade["edge"], trade["market_implied_prob"], resolved_yes
            )
            pnl_dollars = pnl_per_dollar * trade["suggested_position_dollars"]

            entry = id_to_entry[trade["id"]]
            entry["status"] = "CLOSED"
            entry["resolution"] = resolution
            entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
            entry["pnl_per_dollar"] = pnl_per_dollar
            entry["pnl_dollars"] = pnl_dollars

            direction = "WIN" if pnl_dollars > 0 else "LOSS"
            print(f"  {ticker}: resolved {resolution.upper()} → {direction}  "
                  f"P&L: ${pnl_dollars:+.2f}")
            resolved_count += 1

        except Exception as exc:
            print(f"  ERROR resolving {ticker}: {exc}")
            error_count += 1

    _save_ledger(ledger)
    print(f"\nResolved {resolved_count} trade(s). Errors: {error_count}.")
    return 0


def cmd_report() -> int:
    """Aggregate CLOSED trades and print win rate, mean edge, total P&L."""
    ledger = _load_ledger()
    if not ledger:
        print("Ledger is empty.")
        return 0

    open_trades = [t for t in ledger if t["status"] == "OPEN"]
    closed_trades = [t for t in ledger if t["status"] == "CLOSED"]

    print(f"\n{'='*60}")
    print("=== Kalshi BTC Paper Trade Report ===")
    print(f"{'='*60}\n")
    print(f"Total entries: {len(ledger)}")
    print(f"  OPEN:   {len(open_trades)}")
    print(f"  CLOSED: {len(closed_trades)}\n")

    if not closed_trades:
        print("No closed trades yet. Run `resolve` after markets expire.")
        return 0

    wins = [t for t in closed_trades if (t["pnl_dollars"] or 0) > 0]
    losses = [t for t in closed_trades if (t["pnl_dollars"] or 0) <= 0]
    win_rate = len(wins) / len(closed_trades) * 100
    total_pnl = sum(t["pnl_dollars"] or 0.0 for t in closed_trades)
    mean_expected_ev = np.mean([t["expected_value_per_dollar"] for t in closed_trades])
    mean_realized_pnl_per_dollar = np.mean([t["pnl_per_dollar"] or 0.0 for t in closed_trades])
    mean_edge = np.mean([t["edge"] for t in closed_trades])

    pnl_series = [t["pnl_dollars"] or 0.0 for t in closed_trades]
    pnl_per_dollar_series = [t["pnl_per_dollar"] or 0.0 for t in closed_trades]

    # Sharpe (per-trade)
    pnl_arr = np.array(pnl_per_dollar_series)
    sharpe = float(pnl_arr.mean() / pnl_arr.std()) if pnl_arr.std() > 0 else 0.0

    # Max drawdown
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())

    print(f"Performance summary ({len(closed_trades)} closed trades):")
    print(f"  Win rate:              {win_rate:.1f}%  ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Mean expected edge:    {mean_edge:+.1%}")
    print(f"  Mean exp EV/dollar:    {mean_expected_ev:+.3f}")
    print(f"  Mean realized P&L/\$:  {mean_realized_pnl_per_dollar:+.3f}")
    print(f"  Total P&L:             ${total_pnl:+.2f}")
    print(f"  Sharpe (per trade):    {sharpe:.2f}")
    print(f"  Max drawdown:          ${max_dd:+.2f}")

    # Recent closed trades
    recent = sorted(closed_trades, key=lambda t: t.get("resolved_at") or "", reverse=True)[:10]
    print(f"\nRecent closed trades (last {len(recent)}):")
    print(f"  {'Ticker':<35} {'Edge':>7} {'PnL':>8} {'Resolution'}")
    print(f"  {'─'*65}")
    for t in recent:
        print(
            f"  {t['ticker']:<35} {t['edge']:>+6.1%} "
            f"${t['pnl_dollars'] or 0:>+7.2f}   {t['resolution']}"
        )

    print(f"\n{'='*60}\n")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    commands = {
        "record": cmd_record,
        "resolve": cmd_resolve,
        "report": cmd_report,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python3 -m scripts.kalshi_paper_trade <{' | '.join(commands)}>")
        return 1

    return commands[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
