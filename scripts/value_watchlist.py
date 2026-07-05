"""Value Watchlist — monthly full-universe scan.

Usage:
    python -m scripts.value_watchlist                    # russell2000 default
    python -m scripts.value_watchlist --universe sp500
    python -m scripts.value_watchlist --top 30
    python -m scripts.value_watchlist --no-cache
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from data.sources.yahooquery_fundamentals import fetch_fundamentals_batch
from domain.value.fundamentals import normalize_fundamentals
from domain.value.sector_relative import compute_sector_medians, compute_sector_relative
from domain.value.fundamental_divergence import compute_fundamental_divergence
from domain.value.volume_overlay import compute_volume_overlay
from domain.value.short_overlay import compute_short_overlay
from domain.value.scoring import assign_role1, assign_role2
from domain.value.report import build_report
from domain.value.types import ValuationSnapshot, MispricingReport


_UNIVERSE_FILES: dict[str, Path] = {
    'russell2000': Path('data/universe/russell2000_constituents.json'),
    'sp500': Path('data/universe/sp500_constituents.json'),
}


def _load_universe(name: str) -> list[str]:
    path = _UNIVERSE_FILES.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Universe {name} not found at {path}")
    return json.loads(path.read_text())


def _cache_path(cache_dir: Path) -> Path:
    ym = datetime.utcnow().strftime('%Y-%m')
    return cache_dir / f'fundamentals_{ym}.parquet'


def _prior_month_yyyy_mm(today) -> str:
    """Return the calendar year-month string for the month preceding `today`."""
    y, m = today.year, today.month - 1
    if m == 0:
        y -= 1
        m = 12
    return f'{y:04d}-{m:02d}'


def _load_or_fetch_fundamentals(
    constituents: list[str],
    cache_dir: Path,
    use_cache: bool,
) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir)
    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return {row['ticker']: row.to_dict() for _, row in df.iterrows()}

    rows = fetch_fundamentals_batch(constituents)
    if rows:
        df = pd.DataFrame(list(rows.values()))
        df.to_parquet(cache_file, index=False)
    return rows


def _score_ticker(
    inputs, sector_medians, prior_si_pct: Optional[float]
) -> ValuationSnapshot:
    """Compute all sub-results + Role1/Role2 for a single ValuationInputs."""
    sr = compute_sector_relative(inputs, sector_medians)
    fd = compute_fundamental_divergence(inputs)

    if sr is None and fd is None:
        return ValuationSnapshot(
            ticker=inputs.ticker, inputs=inputs, sector_relative=None,
            fundamental_divergence=None, volume=None, short_overlay=None,
            role1=None, role2=None, skip_reason='insufficient_data',
        )

    vol = compute_volume_overlay(inputs)
    scores = [x.score for x in (sr, fd) if x is not None]
    base_rank = sum(scores) / len(scores)
    short = compute_short_overlay(inputs, base_rank=base_rank,
                                   prior_short_interest_pct=prior_si_pct)

    role1 = assign_role1(sr, fd, vol, short)
    role2 = assign_role2(sr, fd, vol, short)
    return ValuationSnapshot(
        ticker=inputs.ticker, inputs=inputs,
        sector_relative=sr, fundamental_divergence=fd,
        volume=vol, short_overlay=short, role1=role1, role2=role2,
        skip_reason=None,
    )


def run_watchlist(
    constituents: list[str],
    out_dir: Path,
    cache_dir: Path,
    universe_name: str,
    top_n: int = 20,
    use_cache: bool = True,
    prior_report_path: Optional[Path] = None,
) -> MispricingReport:
    """Execute the full pipeline.  Returns MispricingReport."""
    raw_rows = _load_or_fetch_fundamentals(constituents, cache_dir, use_cache)

    prior_si: dict[str, float] = {}
    if prior_report_path and prior_report_path.exists():
        prior = json.loads(prior_report_path.read_text())
        for lst_key in ('role1_ranked_longs', 'role1_ranked_shorts',
                        'role2_ranked_longs', 'role2_ranked_shorts'):
            for row in prior.get(lst_key, []):
                if row.get('short_interest_pct') is not None:
                    prior_si[row['ticker']] = row['short_interest_pct']

    inputs_list = []
    normalize_failures: list[str] = []
    for t in constituents:
        raw = raw_rows.get(t)
        if not raw:
            normalize_failures.append(t)
            continue
        try:
            inputs_list.append(normalize_fundamentals(raw))
        except Exception:
            normalize_failures.append(t)

    sector_medians = compute_sector_medians(inputs_list)

    snapshots: list[ValuationSnapshot] = []
    for v in inputs_list:
        snap = _score_ticker(v, sector_medians,
                             prior_si_pct=prior_si.get(v.ticker))
        snapshots.append(snap)

    for t in normalize_failures:
        snapshots.append(ValuationSnapshot(
            ticker=t,
            inputs=normalize_fundamentals({'ticker': t}),
            sector_relative=None, fundamental_divergence=None,
            volume=None, short_overlay=None, role1=None, role2=None,
            skip_reason='missing_fundamentals',
        ))

    ts_utc = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    report = build_report(
        snapshots=snapshots, universe_name=universe_name,
        universe_size=len(constituents), run_timestamp_utc=ts_utc,
        top_n=top_n,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_timestamp_utc': report.run_timestamp_utc,
        'universe': report.universe,
        'universe_size': report.universe_size,
        'graded_size': report.graded_size,
        'skipped': report.skipped,
        'skipped_reasons': report.skipped_reasons,
        'role1_ranked_longs': report.role1_ranked_longs,
        'role1_ranked_shorts': report.role1_ranked_shorts,
        'role2_ranked_longs': report.role2_ranked_longs,
        'role2_ranked_shorts': report.role2_ranked_shorts,
    }
    (out_dir / 'latest.json').write_text(json.dumps(payload, indent=2, default=str))
    ym = datetime.utcnow().strftime('%Y-%m')
    (out_dir / f'value_report_{ym}.json').write_text(json.dumps(payload, indent=2, default=str))

    _print_summary(report)
    return report


def _print_summary(report):
    print(f"\n=== Value Watchlist — {report.universe} ===")
    print(f"Universe: {report.universe_size}  Graded: {report.graded_size}  Skipped: {report.skipped}")
    print(f"\nTOP 5 LONGS (Role 1 rank):")
    for i, row in enumerate(report.role1_ranked_longs[:5], 1):
        print(f"  {i} {row['ticker']:<7}  score {row.get('combined_rank_score',0):.1f}  "
              f"vol {row.get('volume_tag','?'):<10}  SI {row.get('short_interest_pct') or 0:.1f}%  "
              f"{row.get('priority','')}")
    print(f"\nTOP 5 LONGS (Role 2 composite):")
    for i, row in enumerate(report.role2_ranked_longs[:5], 1):
        print(f"  {i} {row['ticker']:<7}  composite {row.get('composite_score',0):.1f}")
    print(f"\nTOP 5 SHORTS (Role 1 rank):")
    for i, row in enumerate(report.role1_ranked_shorts[:5], 1):
        print(f"  {i} {row['ticker']:<7}  score {row.get('combined_rank_score',0):.1f}  "
              f"SI {row.get('short_interest_pct') or 0:.1f}%  {row.get('priority','')}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description='Value Watchlist — monthly stock mispricing scan')
    p.add_argument('--universe', choices=list(_UNIVERSE_FILES.keys()), default='russell2000')
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--out', type=Path, default=Path('output/value'))
    p.add_argument('--cache-dir', type=Path, default=Path('cache/value'))
    args = p.parse_args(argv)

    try:
        constituents = _load_universe(args.universe)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    from datetime import date
    prior_ym = _prior_month_yyyy_mm(date.today())
    prior_path = args.out / f'value_report_{prior_ym}.json'

    run_watchlist(
        constituents=constituents,
        out_dir=args.out, cache_dir=args.cache_dir,
        universe_name=args.universe, top_n=args.top,
        use_cache=not args.no_cache,
        prior_report_path=prior_path if prior_path.exists() else None,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
