"""value_check — on-demand single-ticker deep dive.

Loads the current month's fundamentals cache and prints a full
breakdown for one ticker.

Usage:
    python -m scripts.value_check INTC
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from domain.value.fundamentals import normalize_fundamentals
from domain.value.sector_relative import compute_sector_medians, compute_sector_relative
from domain.value.fundamental_divergence import compute_fundamental_divergence
from domain.value.volume_overlay import compute_volume_overlay
from domain.value.short_overlay import compute_short_overlay
from domain.value.scoring import assign_role1, assign_role2
from domain.value.report import snapshot_to_dict
from domain.value.types import ValuationSnapshot


def _current_cache_file(cache_dir: Path) -> Optional[Path]:
    ym = datetime.utcnow().strftime('%Y-%m')
    p = cache_dir / f'fundamentals_{ym}.parquet'
    return p if p.exists() else None


def check_ticker(ticker: str, cache_dir: Path = Path('cache/value')) -> Optional[dict]:
    """Return the flat dict for `ticker` if present in this month's cache."""
    cache_file = _current_cache_file(cache_dir)
    if cache_file is None:
        print(f"ERROR: no cached fundamentals for {datetime.utcnow().strftime('%Y-%m')} at {cache_dir}",
              file=sys.stderr)
        return None
    df = pd.read_parquet(cache_file)
    inputs_list = [
        normalize_fundamentals(r.to_dict())
        for _, r in df.iterrows()
    ]
    hit = next((v for v in inputs_list if v.ticker == ticker), None)
    if hit is None:
        print(f"ERROR: {ticker} not in current cache", file=sys.stderr)
        return None

    sector_medians = compute_sector_medians(inputs_list)
    sr = compute_sector_relative(hit, sector_medians)
    fd = compute_fundamental_divergence(hit)
    vol = compute_volume_overlay(hit)
    scores = [s.score for s in (sr, fd) if s is not None]
    base_rank = sum(scores) / len(scores) if scores else 5.0
    short = compute_short_overlay(hit, base_rank=base_rank, prior_short_interest_pct=None)
    role1 = assign_role1(sr, fd, vol, short) if (sr or fd) else None
    role2 = assign_role2(sr, fd, vol, short) if (sr or fd) else None

    snap = ValuationSnapshot(
        ticker=ticker, inputs=hit,
        sector_relative=sr, fundamental_divergence=fd,
        volume=vol, short_overlay=short, role1=role1, role2=role2,
        skip_reason=None if role1 else 'insufficient_data',
    )
    return snapshot_to_dict(snap)


def _pretty_print(d: dict) -> None:
    print(f"\n=== {d['ticker']} — {d.get('sector', '?')} ===")
    print(f"Spot: ${d.get('spot', 0):.2f}")
    if 'sector_relative_score' in d:
        print(f"\nSector-relative score: {d['sector_relative_score']:.2f} / 10")
        m = d.get('sector_multiples', {})
        for k in ('pe_fwd', 'ps_ttm', 'pb', 'ev_ebitda'):
            if m.get(k) is not None:
                pm = m.get(f'{k}_peer_median')
                dp = m.get(f'{k}_discount_pct')
                dp_s = f"({dp:+.1f}% vs peer)" if dp is not None else ""
                pm_s = f"{pm:.2f}" if pm is not None else "n/a"
                print(f"  {k}: {m[k]:.2f}  peer median {pm_s}  {dp_s}")
    if 'fundamental_divergence_score' in d:
        print(f"\nFundamental divergence: {d['fundamental_divergence_score']:.2f} / 10")
        t = d.get('fundamental_trend', {})
        if t.get('revenue_growth_yoy_pct') is not None:
            print(f"  Revenue growth YoY: {t['revenue_growth_yoy_pct']:+.1f}%")
        if t.get('eps_growth_yoy_pct') is not None:
            print(f"  EPS growth YoY:     {t['eps_growth_yoy_pct']:+.1f}%")
        if t.get('price_growth_yoy_pct') is not None:
            print(f"  Price growth YoY:   {t['price_growth_yoy_pct']:+.1f}%")
        if t.get('divergence_pp') is not None:
            print(f"  Divergence:         {t['divergence_pp']:+.1f} pp")
    if 'volume_tag' in d:
        v = d.get('volume', {})
        r = v.get('ratio')
        r_s = f" (5d/20d = {r:.2f})" if r is not None else ""
        print(f"\nVolume: {d['volume_tag']}{r_s}")
    if 'short_interest_pct' in d and d.get('short_interest_pct') is not None:
        print(f"Short interest: {d['short_interest_pct']:.1f}%   "
              f"Days to cover: {d.get('days_to_cover') or 0:.1f}")
    if 'priority' in d:
        print(f"\nRole 1 priority: {d['priority']}")
    if 'composite_score' in d:
        print(f"Role 2 composite: {d['composite_score']:.1f} / 100")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description='Value scanner — single ticker check')
    p.add_argument('ticker')
    p.add_argument('--cache-dir', type=Path, default=Path('cache/value'))
    args = p.parse_args(argv)

    out = check_ticker(args.ticker, cache_dir=args.cache_dir)
    if out is None:
        return 1
    _pretty_print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
