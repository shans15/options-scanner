"""A+ Confluence paper trade tracker.

Forward-test infrastructure that records real A+/A signals, marks them to
market daily, closes at target/stop/expiration, and prints a P&L summary.

Concurrency assumption: single-writer; no file locking is applied.  Run
only one instance at a time (cron or manual).

CLI
---
    python3 -m scripts.aplus_paper_trade [ingest|mark|close|summary|run]

    ingest   — read latest.json, open new positions
    mark     — fetch live chains, update unrealised P&L
    close    — close positions at target / stop / expiration
    summary  — print P&L table
    run      — ingest + mark + close + summary  (default)

Flags
-----
    --scan       path to aplus latest.json  (default: output/aplus/latest.json)
    --store      path to paper_trades.json  (default: output/aplus/paper_trades.json)
    --target-pct double-or-more trigger     (default: 100.0)
    --stop-pct   half-loss trigger          (default: -50.0)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

_EMPTY_STORE: dict = {"last_updated": None, "positions": []}


def _load_store(store_path: Path) -> dict:
    if not store_path.exists():
        return {"last_updated": None, "positions": []}
    try:
        return json.loads(store_path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("Could not parse %s — starting with empty store.", store_path)
        return {"last_updated": None, "positions": []}


def _save_store(store_path: Path, store: dict) -> None:
    store["last_updated"] = datetime.now().isoformat(timespec="seconds")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2, default=str))


# ---------------------------------------------------------------------------
# Position ID
# ---------------------------------------------------------------------------

def _make_position_id(
    opened_at: str,
    ticker: str,
    option_type: str,
    strike: float,
    expiration: str,
) -> str:
    return f"{opened_at}-{ticker}-{option_type}-{strike:.0f}-{expiration}"


# ---------------------------------------------------------------------------
# Wilson 95% CI for a proportion
# ---------------------------------------------------------------------------

def _wilson_ci(wins: int, n: int) -> tuple[float, float]:
    """Return (lower, upper) Wilson score interval at 95% confidence."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p_hat = wins / n
    centre = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    margin = (z / (1 + z * z / n)) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def cmd_ingest(scan_path: Path, store_path: Path, today: Optional[date] = None) -> int:
    today = today or date.today()
    today_str = today.isoformat()

    if not scan_path.exists():
        print(f"ERROR: scan file not found: {scan_path}", file=sys.stderr)
        return 1

    scan_data = json.loads(scan_path.read_text())
    candidates = scan_data.get("graded_candidates", [])

    store = _load_store(store_path)
    existing_ids = {p["id"] for p in store["positions"]}

    new_positions: list[dict] = []
    skipped_grade = 0
    skipped_dup = 0
    skipped_too_expensive = 0

    for cand in candidates:
        grade = cand.get("grade", "")
        if grade not in ("A+", "A"):
            skipped_grade += 1
            continue

        contract = cand.get("contract", {})
        ticker = cand.get("ticker", contract.get("ticker", ""))
        option_type = contract.get("option_type", "")
        strike = float(contract.get("strike", 0))
        expiration = str(contract.get("expiration", ""))
        strategy = cand.get("strategy", "")
        structure = cand.get("structure", "")

        # entry mid
        bid = float(contract.get("bid", 0))
        ask = float(contract.get("ask", 0))
        # prefer pre-computed mid; fall back to average
        if "mid" in contract and contract["mid"] is not None:
            entry_mid = float(contract["mid"])
        else:
            entry_mid = (bid + ask) / 2.0

        if entry_mid <= 0:
            log.warning("Skipping %s — entry_mid=%.2f (zero or negative)", ticker, entry_mid)
            continue

        max_risk_dollars = float(cand.get("max_risk_dollars", 0))
        sizing_pct = float(cand.get("sizing_pct", 0))
        composite_score = float(cand.get("composite_score", 0))
        dte_at_entry = int(contract.get("dte", 0))

        # contracts_count: floor(max_risk / (entry_mid * 100)), min 1
        contracts_count = int(math.floor(max_risk_dollars / (entry_mid * 100)))
        if contracts_count < 1:
            log.warning(
                "Skipping %s — entry_mid=%.2f too expensive for max_risk=$%.0f",
                ticker, entry_mid, max_risk_dollars,
            )
            skipped_too_expensive += 1
            continue

        entry_cost_dollars = round(contracts_count * 100 * entry_mid, 2)

        position_id = _make_position_id(today_str, ticker, option_type, strike, expiration)

        if position_id in existing_ids:
            skipped_dup += 1
            continue

        position: dict = {
            "id": position_id,
            "opened_at": today_str,
            "ticker": ticker,
            "strategy": strategy,
            "structure": structure,
            "grade": grade,
            "composite_score": composite_score,
            "category_scores": cand.get("category_scores", {}),
            "contract": {
                "option_type": option_type,
                "strike": strike,
                "expiration": expiration,
                "dte_at_entry": dte_at_entry,
            },
            "entry_mid": entry_mid,
            "contracts_count": contracts_count,
            "entry_cost_dollars": entry_cost_dollars,
            "max_risk_dollars": max_risk_dollars,
            "sizing_pct": sizing_pct,
            "status": "open",
            "last_mark_mid": entry_mid,
            "last_marked_at": today_str,
            "current_pnl_dollars": 0.0,
            "current_pnl_pct": 0.0,
            "closed_at": None,
            "close_mid": None,
            "close_reason": None,
            "final_pnl_dollars": None,
            "final_pnl_pct": None,
        }

        store["positions"].append(position)
        existing_ids.add(position_id)
        new_positions.append(position)

    _save_store(store_path, store)

    print(f"Opened {len(new_positions)} new position(s).")
    for p in new_positions:
        print(
            f"  [{p['grade']}] {p['id']}  entry={p['entry_mid']:.2f} "
            f"x{p['contracts_count']}  cost=${p['entry_cost_dollars']:.2f}"
        )
    if skipped_dup:
        print(f"  (skipped {skipped_dup} duplicate id(s))")
    if skipped_too_expensive:
        print(f"  (skipped {skipped_too_expensive} position(s) too expensive for risk budget)")

    return 0


# ---------------------------------------------------------------------------
# mark
# ---------------------------------------------------------------------------

def cmd_mark(store_path: Path, today: Optional[date] = None) -> int:
    today = today or date.today()
    today_str = today.isoformat()

    store = _load_store(store_path)
    open_positions = [p for p in store["positions"] if p["status"] == "open"]

    if not open_positions:
        print("No open positions to mark.")
        return 0

    # Import data sources lazily so tests can mock them
    from data.sources.factory import default_sources
    from data.fallback import fetch_with_fallback, DataFetchError

    sources = default_sources()

    marked_count = 0
    for pos in open_positions:
        ticker = pos["ticker"]
        option_type = pos["contract"]["option_type"]
        strike = float(pos["contract"]["strike"])
        expiration_str = pos["contract"]["expiration"]

        try:
            chain = fetch_with_fallback(sources, "fetch_option_chain", ticker)
        except DataFetchError as exc:
            log.error("Could not fetch chain for %s: %s", ticker, exc)
            print(f"  WARN: Could not fetch chain for {ticker}: {exc}")
            continue
        except Exception as exc:
            log.error("Unexpected error fetching chain for %s: %s", ticker, exc)
            print(f"  WARN: Unexpected error for {ticker}: {exc}")
            continue

        # Find matching contract
        matching = None
        for contract in chain:
            c_exp = (
                contract.expiration.isoformat()
                if hasattr(contract.expiration, "isoformat")
                else str(contract.expiration)
            )
            if (
                contract.option_type == option_type
                and abs(contract.strike - strike) < 0.01
                and c_exp == expiration_str
            ):
                matching = contract
                break

        if matching is None:
            print(f"  WARN: No matching contract found for {pos['id']} — mark stale")
            continue

        current_mid = (matching.bid + matching.ask) / 2.0
        entry_mid = pos["entry_mid"]
        contracts_count = pos["contracts_count"]

        pnl_dollars = round((current_mid - entry_mid) * 100 * contracts_count, 2)
        pnl_pct = (
            round((current_mid - entry_mid) / entry_mid * 100, 4)
            if entry_mid > 0
            else 0.0
        )

        pos["last_mark_mid"] = current_mid
        pos["last_marked_at"] = today_str
        pos["current_pnl_dollars"] = pnl_dollars
        pos["current_pnl_pct"] = pnl_pct
        marked_count += 1

    _save_store(store_path, store)
    print(f"Marked {marked_count} position(s).")
    return 0


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

def cmd_close(
    store_path: Path,
    target_pct: float = 100.0,
    stop_pct: float = -50.0,
    today: Optional[date] = None,
) -> int:
    today = today or date.today()
    today_str = today.isoformat()

    store = _load_store(store_path)
    open_positions = [p for p in store["positions"] if p["status"] == "open"]

    if not open_positions:
        print("No open positions to close.")
        return 0

    closed_target = 0
    closed_stop = 0
    closed_expiration = 0

    for pos in open_positions:
        expiration_str = pos["contract"]["expiration"]
        try:
            expiration_date = date.fromisoformat(expiration_str[:10])
        except ValueError:
            expiration_date = None

        pnl_pct = pos.get("current_pnl_pct", 0.0)

        reason: Optional[str] = None
        if expiration_date is not None and today >= expiration_date:
            reason = "expiration"
            closed_expiration += 1
        elif pnl_pct >= target_pct:
            reason = "target"
            closed_target += 1
        elif pnl_pct <= stop_pct:
            reason = "stop"
            closed_stop += 1

        if reason is None:
            continue

        close_mid = pos.get("last_mark_mid", pos["entry_mid"])
        entry_mid = pos["entry_mid"]
        contracts_count = pos["contracts_count"]

        final_pnl_dollars = round((close_mid - entry_mid) * 100 * contracts_count, 2)
        final_pnl_pct = (
            round((close_mid - entry_mid) / entry_mid * 100, 4)
            if entry_mid > 0
            else 0.0
        )

        pos["status"] = "closed"
        pos["closed_at"] = today_str
        pos["close_mid"] = close_mid
        pos["close_reason"] = reason
        pos["final_pnl_dollars"] = final_pnl_dollars
        pos["final_pnl_pct"] = final_pnl_pct

    total_closed = closed_target + closed_stop + closed_expiration
    _save_store(store_path, store)
    print(
        f"Closed {total_closed} position(s) "
        f"[target={closed_target}, stop={closed_stop}, expiration={closed_expiration}]"
    )
    return 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def cmd_summary(store_path: Path) -> int:
    store = _load_store(store_path)
    positions = store.get("positions", [])

    if not positions:
        print("No positions in store.")
        return 0

    open_pos = [p for p in positions if p["status"] == "open"]
    closed_pos = [p for p in positions if p["status"] == "closed"]

    # Header
    print()
    print("=" * 90)
    print("  A+ Paper Trade Summary")
    print("=" * 90)

    # --- Position table ---
    col = "{:<36} {:<5} {:<6} {:<11} {:>9} {:>9} {:>9} {:>8} {:<8} {:>9}"
    print(col.format("ID (short)", "Grade", "Ticker", "Strategy", "Entry", "Mark", "P&L $", "P&L %", "Status", "Days"))
    print("  " + "-" * 88)

    today = date.today()
    for p in sorted(positions, key=lambda x: x.get("opened_at", ""), reverse=True):
        id_short = p["id"][-32:] if len(p["id"]) > 32 else p["id"]
        try:
            days_held = (today - date.fromisoformat(p["opened_at"])).days
        except (ValueError, KeyError):
            days_held = 0

        pnl_d = p.get("final_pnl_dollars") if p["status"] == "closed" else p.get("current_pnl_dollars", 0.0)
        pnl_p = p.get("final_pnl_pct") if p["status"] == "closed" else p.get("current_pnl_pct", 0.0)

        print(col.format(
            id_short,
            p.get("grade", "?"),
            p.get("ticker", "?"),
            (p.get("strategy", "?") or "?")[:11],
            f"{p['entry_mid']:.2f}",
            f"{p.get('last_mark_mid', p['entry_mid']):.2f}",
            f"${pnl_d:+.2f}" if pnl_d is not None else "n/a",
            f"{pnl_p:+.1f}%" if pnl_p is not None else "n/a",
            p["status"],
            str(days_held),
        ))

    # --- Aggregates ---
    print()
    if open_pos:
        unrealised = sum(p.get("current_pnl_dollars", 0.0) or 0.0 for p in open_pos)
        cost_basis = sum(p.get("entry_cost_dollars", 0.0) for p in open_pos)
        print(f"  Open positions: {len(open_pos)}")
        print(f"    Total unrealised P&L : ${unrealised:+.2f}")
        print(f"    Total cost basis     : ${cost_basis:.2f}")

    if closed_pos:
        wins = [p for p in closed_pos if (p.get("final_pnl_dollars") or 0.0) > 0]
        win_rate = len(wins) / len(closed_pos) * 100
        total_realised = sum(p.get("final_pnl_dollars") or 0.0 for p in closed_pos)

        by_reason: dict[str, int] = {}
        for p in closed_pos:
            r = p.get("close_reason") or "unknown"
            by_reason[r] = by_reason.get(r, 0) + 1

        print(f"\n  Closed positions: {len(closed_pos)}")
        print(f"    Win rate          : {win_rate:.1f}%  ({len(wins)}/{len(closed_pos)})")
        print(f"    Total realised P&L: ${total_realised:+.2f}")
        reason_str = "  ".join(f"{k}={v}" for k, v in by_reason.items())
        print(f"    By close reason   : {reason_str}")

        # Wilson CI when sample is large enough
        if len(closed_pos) >= 5:
            lo, hi = _wilson_ci(len(wins), len(closed_pos))
            print(
                f"\n  A+/A win rate so far: {win_rate:.1f}%  "
                f"Wilson 95% CI [{lo*100:.1f}%, {hi*100:.1f}%]  (n={len(closed_pos)})"
            )

    print()
    return 0


# ---------------------------------------------------------------------------
# run (default)
# ---------------------------------------------------------------------------

def cmd_run(
    scan_path: Path,
    store_path: Path,
    target_pct: float,
    stop_pct: float,
    today: Optional[date] = None,
) -> int:
    today = today or date.today()
    rc = cmd_ingest(scan_path, store_path, today=today)
    if rc != 0:
        return rc
    cmd_mark(store_path, today=today)
    cmd_close(store_path, target_pct=target_pct, stop_pct=stop_pct, today=today)
    cmd_summary(store_path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A+ Confluence paper trade tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=["ingest", "mark", "close", "summary", "run"],
        default="run",
        help="Action to perform",
    )
    parser.add_argument(
        "--scan",
        default="output/aplus/latest.json",
        help="Path to aplus latest.json",
    )
    parser.add_argument(
        "--store",
        default="output/aplus/paper_trades.json",
        help="Path to paper_trades.json store",
    )
    parser.add_argument(
        "--target-pct",
        type=float,
        default=100.0,
        help="P&L %% threshold for target close (e.g. 100 = doubled)",
    )
    parser.add_argument(
        "--stop-pct",
        type=float,
        default=-50.0,
        help="P&L %% threshold for stop-loss close (e.g. -50 = lost half)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    scan_path = Path(args.scan)
    store_path = Path(args.store)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sub = args.subcommand
    if sub == "ingest":
        return cmd_ingest(scan_path, store_path)
    elif sub == "mark":
        return cmd_mark(store_path)
    elif sub == "close":
        return cmd_close(store_path, target_pct=args.target_pct, stop_pct=args.stop_pct)
    elif sub == "summary":
        return cmd_summary(store_path)
    else:  # "run"
        return cmd_run(scan_path, store_path, args.target_pct, args.stop_pct)


if __name__ == "__main__":
    raise SystemExit(main())
