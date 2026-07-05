"""Refresh the committed Russell 2000 constituents file.

Pulls from iShares IWM holdings CSV.  Writes a plain JSON list to
data/universe/russell2000_constituents.json.

Usage:
    python -m scripts.refresh_russell2000
    python -m scripts.refresh_russell2000 --dry-run
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SOURCE_URL = (
    'https://www.ishares.com/us/products/239710/'
    'ishares-russell-2000-etf/1467271812596.ajax'
    '?fileType=csv&fileName=IWM_holdings&dataType=fund'
)
OUTPUT_PATH = Path('data/universe/russell2000_constituents.json')

_TICKER_COLUMN_CANDIDATES = ('Ticker', 'Symbol', 'ticker', 'symbol')


def fetch_constituents(url: str = SOURCE_URL) -> list[str]:
    """Pull tickers from the iShares CSV, dedup, normalise dots to dashes."""
    df = pd.read_csv(url)
    col = None
    for c in _TICKER_COLUMN_CANDIDATES:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise RuntimeError(f"No ticker column found; got {list(df.columns)}")

    syms = df[col].astype(str).str.strip()
    syms = [s.replace('.', '-') for s in syms if s and s != 'nan']
    seen: set[str] = set()
    out: list[str] = []
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Refresh Russell 2000 list')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--url', default=SOURCE_URL)
    parser.add_argument('--out', default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)

    try:
        tickers = fetch_constituents(args.url)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not tickers:
        print("ERROR: 0 tickers returned", file=sys.stderr)
        return 1

    print(f"Fetched {len(tickers)} tickers")
    print(f"  first: {tickers[:5]}")
    print(f"  last:  {tickers[-5:]}")

    if args.dry_run:
        print("Dry run — not writing.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tickers, indent=2) + '\n')
    print(f"Wrote {out} ({len(tickers)} tickers).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
