"""Refresh the committed S&P 500 constituents file.

Run monthly (or when the index changes are worth picking up).  Pulls the
list from datahub.io's GitHub-backed CSV and writes a plain JSON list to
``data/universe/sp500_constituents.json``.

Usage:
    python -m scripts.refresh_sp500
    python -m scripts.refresh_sp500 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SOURCE_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)
OUTPUT_PATH = Path("data/universe/sp500_constituents.json")


def fetch_constituents(url: str = SOURCE_URL) -> list[str]:
    """Pull tickers from the datahub CSV.  Normalises class-share dots to dashes
    (e.g. ``BRK.B`` → ``BRK-B``) to match Yahoo's symbol convention used by the
    rest of the scanner."""
    df = pd.read_csv(url)
    if "Symbol" not in df.columns:
        raise RuntimeError(
            f"Unexpected CSV schema; columns were {list(df.columns)}"
        )
    symbols = df["Symbol"].astype(str).str.strip()
    return [s.replace(".", "-") for s in symbols if s]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Refresh committed S&P 500 list")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written but don't touch the file.",
    )
    parser.add_argument(
        "--url", default=SOURCE_URL,
        help="Override CSV source URL.",
    )
    parser.add_argument(
        "--out", default=str(OUTPUT_PATH),
        help="Override output path.",
    )
    args = parser.parse_args(argv)

    try:
        tickers = fetch_constituents(args.url)
    except Exception as exc:
        print(f"ERROR: constituent fetch failed — {exc}", file=sys.stderr)
        return 1

    if not tickers:
        print("ERROR: source returned 0 tickers; aborting.", file=sys.stderr)
        return 1

    print(f"Fetched {len(tickers)} tickers from {args.url}")
    print(f"  first: {tickers[:5]}")
    print(f"  last:  {tickers[-5:]}")

    if args.dry_run:
        print("Dry run — not writing.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tickers, indent=2) + "\n")
    print(f"Wrote {out} ({len(tickers)} tickers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
