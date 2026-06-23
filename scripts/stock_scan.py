"""Stock swing-trade scanner CLI.

Usage:
    python -m scripts.stock_scan
    python -m scripts.stock_scan --label STRONG_BULL
    python -m scripts.stock_scan --setup compression
    python -m scripts.stock_scan --label BEAR --setup sma_stack
    python -m scripts.stock_scan --universe-file tickers.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path when run as script
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.factory import default_sources
from domain.stock_scanner.scanner import scan
from domain.stock_scanner.universe import UNIVERSE
from domain.stock_scanner.types import StockCandidate

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_VALID_LABELS = {"STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"}
_VALID_SETUPS = {"sma_stack", "compression", "proximity_52w", "volume_surge"}
_OUTPUT_DIR = _ROOT / "output" / "stock_scans"


def _load_universe(universe_file: str | None) -> list[str]:
    if universe_file is None:
        return list(UNIVERSE)
    path = Path(universe_file)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_file}")
    tickers: list[str] = []
    with open(path) as f:
        # Support CSV with first column as ticker or plain newline-separated list
        reader = csv.reader(f)
        for row in reader:
            if row:
                t = row[0].strip().upper()
                if t:
                    tickers.append(t)
    return tickers


def _apply_filters(
    candidates: list[StockCandidate],
    label: str | None,
    setup: str | None,
) -> list[StockCandidate]:
    result = candidates
    if label:
        result = [c for c in result if c.label == label]
    if setup:
        result = [c for c in result if any(s.setup_type == setup for s in c.setups)]
    return result


def _detector_summary(candidate: StockCandidate) -> str:
    parts = []
    for s in sorted(candidate.setups, key=lambda x: -x.strength):
        parts.append(f"{s.setup_type}({s.strength:.2f})")
    return ", ".join(parts)


def _print_table(candidates: list[StockCandidate], title: str) -> None:
    print(f"\n{title}")
    header = f"  {'Tkr':<8} {'Spot':>8}  {'Bull':>6}  {'Bear':>6}  {'Net':>7}  {'Label':<12}  Detectors firing"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in candidates:
        net_str = f"+{c.net_score:.1f}" if c.net_score >= 0 else f"{c.net_score:.1f}"
        det_str = _detector_summary(c)
        print(
            f"  {c.ticker:<8} ${c.spot:>7.2f}  {c.bullish_score:>6.1f}  {c.bearish_score:>6.1f}  {net_str:>7}  {c.label:<12}  {det_str}"
        )


def _save_outputs(candidates: list[StockCandidate], timestamp: str) -> tuple[Path, Path]:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _OUTPUT_DIR / f"scan_{timestamp}.csv"
    json_path = _OUTPUT_DIR / "latest.json"

    # CSV
    fieldnames = [
        "ticker", "spot", "label", "bullish_score", "bearish_score", "net_score",
        "setups_firing", "snapshot_close", "snapshot_high_52w", "snapshot_low_52w",
        "snapshot_volume_5d_avg",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "ticker": c.ticker,
                "spot": c.spot,
                "label": c.label,
                "bullish_score": c.bullish_score,
                "bearish_score": c.bearish_score,
                "net_score": c.net_score,
                "setups_firing": _detector_summary(c),
                "snapshot_close": c.snapshot.get("close"),
                "snapshot_high_52w": c.snapshot.get("high_52w"),
                "snapshot_low_52w": c.snapshot.get("low_52w"),
                "snapshot_volume_5d_avg": c.snapshot.get("volume_5d_avg"),
            })

    # JSON — serialize setups (frozen dataclasses) manually
    def _candidate_to_dict(c: StockCandidate) -> dict:
        return {
            "ticker": c.ticker,
            "spot": c.spot,
            "label": c.label,
            "bullish_score": c.bullish_score,
            "bearish_score": c.bearish_score,
            "net_score": c.net_score,
            "snapshot": c.snapshot,
            "setups": [
                {
                    "ticker": s.ticker,
                    "setup_type": s.setup_type,
                    "direction": s.direction,
                    "strength": s.strength,
                    "spot": s.spot,
                    "notes": s.notes,
                }
                for s in c.setups
            ],
        }

    payload = {
        "scan_timestamp": timestamp,
        "total_candidates": len(candidates),
        "candidates": [_candidate_to_dict(c) for c in candidates],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stock swing-trade scanner")
    parser.add_argument(
        "--universe-file",
        default=None,
        help="Path to CSV/text file with tickers (one per line, first column). "
             "Defaults to built-in 290-ticker universe.",
    )
    parser.add_argument(
        "--label",
        choices=list(_VALID_LABELS),
        default=None,
        help="Filter output to candidates with this label.",
    )
    parser.add_argument(
        "--setup",
        choices=list(_VALID_SETUPS),
        default=None,
        help="Filter output to candidates with this detector firing.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Days of OHLCV history to fetch per ticker (default: 365).",
    )
    args = parser.parse_args(argv)

    universe = _load_universe(args.universe_file)
    sources = default_sources()

    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n=== STOCK SCAN — {today_str} ===")
    print(f"Universe: {len(universe)} tickers | Fetching OHLCV ({args.lookback_days}d lookback) ...")

    t0 = time.time()
    all_candidates = scan(universe, sources, lookback_days=args.lookback_days)
    elapsed = time.time() - t0

    filtered = _apply_filters(all_candidates, args.label, args.setup)
    strong_net = [c for c in all_candidates if abs(c.net_score) >= 30]

    print(
        f"Universe: {len(universe)} tickers | "
        f"Setups detected on {len(all_candidates)} tickers | "
        f"Candidates >=|30| net: {len(strong_net)}"
    )
    if args.label or args.setup:
        filters_desc = " + ".join(filter(None, [args.label, args.setup]))
        print(f"Filter applied: {filters_desc} -> {len(filtered)} candidates")

    # Top 20 bullish (highest net_score)
    display_candidates = filtered if (args.label or args.setup) else all_candidates
    bulls = sorted(
        [c for c in display_candidates if c.net_score > 0],
        key=lambda c: -c.net_score,
    )[:20]
    bears = sorted(
        [c for c in display_candidates if c.net_score < 0],
        key=lambda c: c.net_score,
    )[:20]

    _print_table(bulls, "TOP 20 BULLISH (sorted by net_score desc)")
    _print_table(bears, "TOP 20 BEARISH (sorted by net_score asc)")

    csv_path, json_path = _save_outputs(all_candidates, timestamp)
    print(f"\nSaved {len(all_candidates)} candidates -> {csv_path}")
    print(f"                   -> {json_path}")
    print(f"Scan completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
