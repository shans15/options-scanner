"""A+ Confluence watchlist CLI.

Reads a scan JSON, grades every candidate, filters to A+/A, and writes
output/aplus/latest.json plus a console summary.

Usage:
    python -m scripts.aplus_watchlist
    python -m scripts.aplus_watchlist --scan output/scans/scan_20260614_0811.json
"""
from __future__ import annotations
import argparse
import functools
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from domain.aplus.features import extract_features
from domain.aplus.scoring import score_categories
from domain.aplus.grading import assign_grade
from domain.aplus.structure import select_structure
from domain.aplus.market_context import build_market_context
from domain.aplus.types import MarketContext, TradeStructure, GradedCandidate


_GRADE_RANK = {'A+': 5, 'A': 4, 'B+': 3, 'B': 2, 'F': 1}

# Per spec 2026-06-29-aplus-rebalance-design.md — $10k account sizing.
# Midpoints of the 7-10% (A+) and 4-5% (A) bands.
_SIZING_PCT = {'A+': 0.085, 'A': 0.045, 'B+': 0.0, 'B': 0.0, 'F': 0.0}

# Documented operating rules surfaced in the output JSON.  The script does
# not track open positions; the trader applies these rules manually.
_OPS_RULES: dict = {
    'max_concurrent_positions': 5,
    'max_daily_new_entries': 2,
    'max_same_ticker_positions': 2,
    'max_consecutive_losses_before_cooldown': 3,
    'cooldown_hours': 48,
    'max_daily_drawdown_pct': -0.03,
    'max_weekly_drawdown_pct': -0.07,
    'vix_expansion_blocks_new_entries': True,
}


def render_watchlist(
    scan_path: Path,
    out_dir: Path,
    account_size: float = 1000.0,
    today: Optional[date] = None,
) -> list[GradedCandidate]:
    """Read scan JSON, grade candidates, write output. Returns the graded list."""
    today = today or date.today()
    scan_data = json.loads(scan_path.read_text())
    candidates = scan_data.get('candidates', [])

    if not candidates:
        print("No candidates in scan — nothing to grade.")
        return []

    mc = _fetch_market_context(today)

    graded: list[GradedCandidate] = []
    for cand in candidates:
        ticker = cand.get('contract', {}).get('ticker', '')
        days_to_earn = _fetch_days_to_earnings(ticker)
        fs = extract_features(cand, mc, days_to_earnings=days_to_earn)
        cs = score_categories(fs)
        grade = assign_grade(cs, vix_regime=cand.get('vix_regime'))
        if grade not in ('A+', 'A'):
            continue
        structure, rationale = select_structure(fs)
        composite = cs.composite()
        sizing_pct = _SIZING_PCT[grade]
        max_risk = account_size * sizing_pct
        graded.append(GradedCandidate(
            ticker=ticker,
            strategy=cand.get('strategy', ''),
            composite_score=composite,
            grade=grade,
            category_scores=cs,
            feature_scores=fs,
            structure=structure,
            structure_rationale=rationale,
            sizing_pct=sizing_pct,
            max_risk_dollars=max_risk,
            raw_candidate=cand,
        ))

    # Sort: A+ first, then A; within grade, by composite descending.
    graded.sort(key=lambda g: (-_GRADE_RANK[g.grade], -g.composite_score))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_data = {
        'timestamp': today.isoformat(),
        'account_size': account_size,
        'graded_candidates': [_serialize(g) for g in graded],
        'operations_rules': _OPS_RULES,
    }
    (out_dir / 'latest.json').write_text(json.dumps(out_data, indent=2, default=str))

    print_summary(graded, account_size)
    return graded


def _serialize(g: GradedCandidate) -> dict:
    return {
        'ticker': g.ticker,
        'strategy': g.strategy,
        'grade': g.grade,
        'composite_score': g.composite_score,
        'category_scores': {
            'technical': g.category_scores.technical,
            'vol_vix': g.category_scores.vol_vix,
            'catalyst': g.category_scores.catalyst,
            'macro_breadth': g.category_scores.macro_breadth,
            'liquidity': g.category_scores.liquidity,
        },
        'feature_scores': g.feature_scores.values,
        'structure': g.structure.value,
        'structure_rationale': g.structure_rationale,
        'sizing_pct': g.sizing_pct,
        'max_risk_dollars': g.max_risk_dollars,
        'contract': g.raw_candidate.get('contract', {}),
    }


def print_summary(graded: list[GradedCandidate], account_size: float) -> None:
    print()
    print(f"=== A+ Watchlist — account ${account_size:,.0f} ===\n")
    if not graded:
        print("  No A+/A grade candidates today.\n")
        return
    print(f"  Grade  Ticker  Strategy     Composite  Structure       Max risk")
    print(f"  {'-'*65}")
    for g in graded[:10]:
        print(f"  {g.grade:<6} {g.ticker:<7} {g.strategy:<12} {g.composite_score:>7.1f}    "
              f"{g.structure.value:<15} ${g.max_risk_dollars:>5.0f}")
    print()


def _fetch_market_context(today: date) -> MarketContext:
    """Build market context using the canonical 'bullish' direction baseline.
    Direction-specific scores (SPX trend, sector rotation, DXY) are recomputed
    per candidate at feature time."""
    return build_market_context(today=today, setup_direction='bullish')


@functools.lru_cache(maxsize=None)
def _fetch_days_to_earnings(ticker: str) -> Optional[int]:
    """Return calendar days to next earnings, or None if unknown.
    Cached per-ticker so a 1000-candidate scan across N tickers makes N network
    calls, not 1000."""
    try:
        from pipeline.earnings import has_earnings_within
        for d in (3, 7, 14, 30, 60):
            if has_earnings_within(ticker, d):
                return d
        return None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="A+ confluence watchlist")
    p.add_argument('--scan', type=str, default='output/scans/latest.json')
    p.add_argument('--out', type=str, default='output/aplus')
    p.add_argument('--account-size', type=float, default=1000.0)
    args = p.parse_args(argv)
    render_watchlist(Path(args.scan), Path(args.out), args.account_size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
