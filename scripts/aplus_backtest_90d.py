"""A+ Confluence Scorer grading applied to historical 90-day scanner setups.

Reads the most-recent scanner_90d_*.csv (or a user-specified CSV),
reconstructs a historical MarketContext for each row's as_of date
(from pre-loaded cached OHLCV + macro calendar), runs the full
extract_features → score_categories → assign_grade pipeline,
then reports GRADE × WIN RATE correlation.

Usage:
    python3 -m scripts.aplus_backtest_90d
    python3 -m scripts.aplus_backtest_90d --csv output/backtest/scanner_90d_20260614_0819.csv
    python3 -m scripts.aplus_backtest_90d --output-dir /tmp/bt
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.macro_calendar import days_to_next_event
from data.sources.yahoo_macro_source import YahooMacroSource
from domain.aplus.features import extract_features
from domain.aplus.scoring import score_categories
from domain.aplus.grading import assign_grade
from domain.aplus.market_context import score_spx_trend, rank_sectors
from domain.aplus.types import MarketContext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_DIR = _ROOT / "output" / "backtest"
_CACHE_DIR = _ROOT / "cache" / "backtest_ohlcv"

_SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
]

# Neutral-ish liquidity fixtures for missing option chain data.
# bid=0.97 ask=1.03 → 3% spread → liq_bid_ask_spread ≈ 10
# oi=500              → liq_open_interest ≈ 4.4
# volume=100          → volume/oi=0.2 → liq_volume_oi_ratio ≈ 7
# implied_volatility=0.30 → liq_iv_percentile ≈ 7.5
_LIQ_FIXTURE: dict = {
    "bid": 0.97,
    "ask": 1.03,
    "open_interest": 500,
    "volume": 100,
    "implied_volatility": 0.30,
}

# ---------------------------------------------------------------------------
# Wilson CI (copied from scanner_90d_backtest.py)
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial win rate."""
    if trials == 0:
        return (0.0, 1.0)
    z = 1.96
    p = wins / trials
    n = trials
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------

def _load_parquet(ticker: str, cache_dir: Path) -> pd.DataFrame:
    """Load parquet from cache; fall back to yfinance if not present."""
    path = cache_dir / f"{ticker.replace('-', '_')}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = df.index.normalize()
        return df

    # Fallback: fetch and cache
    print(f"    {ticker}: parquet not in cache — fetching via yfinance …")
    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
        if raw.empty:
            raise ValueError("empty response")
        df = raw.rename(columns={c: c.lower() for c in raw.columns})
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = df.index.normalize()
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].astype(float)
        df = df.dropna(subset=["close"])
        df.to_parquet(path)
        print(f"    {ticker}: fetched {len(df)} rows → cached")
        return df
    except Exception as exc:
        print(f"    {ticker}: fetch failed ({exc}) — empty DataFrame")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _slice_to(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return rows with normalized date index <= as_of."""
    return df[df.index <= as_of]


# ---------------------------------------------------------------------------
# Pre-load helpers
# ---------------------------------------------------------------------------

def load_helper_data(cache_dir: Path) -> dict:
    """Load SPY, all 11 sector ETFs, and VIX into memory once."""
    print("\nPre-loading helper data …")

    spy = _load_parquet("SPY", cache_dir)
    print(f"  SPY: {len(spy)} rows ({spy.index[0].date()} → {spy.index[-1].date()})")

    sector_dfs: dict[str, pd.DataFrame] = {}
    for etf in _SECTOR_ETFS:
        df = _load_parquet(etf, cache_dir)
        if not df.empty:
            sector_dfs[etf] = df
            print(f"  {etf}: {len(df)} rows")
        else:
            print(f"  {etf}: MISSING — sector rotation will use neutral rank")

    # VIX — prefer cached macro parquet, else fetch
    vix_series = _load_vix_series()
    print(f"  VIX: {len(vix_series)} rows ({vix_series.index[0].date()} → {vix_series.index[-1].date()})")

    return {
        "spy": spy,
        "sector_dfs": sector_dfs,
        "vix_series": vix_series,
    }


def _load_vix_series() -> pd.Series:
    """Return tz-naive daily VIX close series.  Tries the 2y macro cache first."""
    macro = YahooMacroSource()
    df = macro.fetch_history("^VIX", period="2y")
    if not df.empty and "close" in df.columns:
        s = df["close"].copy()
        if s.index.tz is not None:
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        s.index = s.index.normalize()
        return s
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Historical MarketContext builder
# ---------------------------------------------------------------------------

def build_historical_market_context(
    as_of: pd.Timestamp,
    direction: str,
    helpers: dict,
) -> MarketContext:
    """Construct a MarketContext from cached data sliced to as_of."""
    spy_df = helpers["spy"]
    sector_dfs = helpers["sector_dfs"]

    spy_slice = _slice_to(spy_df, as_of)
    spx_trend = (
        score_spx_trend(spy_slice["close"], direction)
        if len(spy_slice) >= 50
        else 5.0
    )

    # 1-week sector returns: last close / close 5 bars prior - 1
    returns_1w: dict[str, float] = {}
    for etf, df in sector_dfs.items():
        s = _slice_to(df, as_of)["close"]
        if len(s) >= 6:
            returns_1w[etf] = float(s.iloc[-1] / s.iloc[-6] - 1)
        elif len(s) >= 1:
            returns_1w[etf] = 0.0
    sector_rank = rank_sectors(returns_1w) if returns_1w else {}

    # Macro days
    as_of_date: date = as_of.date()
    days_macro = days_to_next_event(as_of_date)

    return MarketContext(
        spx_trend_score=spx_trend,
        sector_rotation_rank=sector_rank,
        dxy_trend_score=5.0,    # not in cache — neutral
        yield_10y_score=5.0,    # not in cache — neutral
        vvix_score=7.0,         # not in cache — neutral (low-ish VVIX assumption)
        days_to_macro_event=days_macro,
    )


# ---------------------------------------------------------------------------
# vix_pct_vs_7d helper
# ---------------------------------------------------------------------------

def compute_vix_pct_vs_7d(
    vix_at_setup: float,
    as_of: pd.Timestamp,
    vix_series: pd.Series,
) -> float:
    """Return (vix_at_setup / trailing_7d_mean) - 1.  0.0 if not enough data."""
    if vix_series.empty:
        return 0.0
    s = vix_series[vix_series.index <= as_of].tail(7)
    if len(s) < 1:
        return 0.0
    mean_7d = float(s.mean())
    if mean_7d <= 0:
        return 0.0
    return float(vix_at_setup / mean_7d - 1)


# ---------------------------------------------------------------------------
# Grade a single CSV row
# ---------------------------------------------------------------------------

def grade_row(row: dict, helpers: dict) -> dict:
    """Return the row augmented with composite_score, grade, and category scores."""
    as_of = pd.Timestamp(row["as_of"])
    direction = str(row.get("direction", "bullish"))
    ticker = str(row.get("ticker", ""))
    spot = float(row.get("spot_at_setup") or 0.0)

    vix_at_setup = float(row.get("vix_at_setup") or 20.0)
    vix_pct = compute_vix_pct_vs_7d(vix_at_setup, as_of, helpers["vix_series"])

    mc = build_historical_market_context(as_of, direction, helpers)

    # Build candidate dict in the shape extract_features expects
    candidate: dict = {
        # Setup-level fields
        "setup_name": row.get("setup_name"),
        "setup_strength": row.get("strength"),
        "setup_direction": direction,
        "weekly_ribbon_agreement": row.get("weekly_ribbon_agreement"),
        "vix_regime": row.get("vix_regime"),
        "vix_pct_vs_7d": vix_pct,
        "pct_change_1w": row.get("pct_change_1w") or 0.0,
        "pct_change_4w": row.get("pct_change_4w") or 0.0,
        # contract sub-dict for liquidity features
        "contract": {
            "ticker": ticker,
            "spot_price": spot,
            **_LIQ_FIXTURE,
        },
    }

    fs = extract_features(candidate, mc, days_to_earnings=None)
    cs = score_categories(fs)
    grade = assign_grade(cs)

    return {
        **row,
        "composite_score": round(cs.composite(), 2),
        "grade": grade,
        "cat_technical": round(cs.technical, 3),
        "cat_vol_vix": round(cs.vol_vix, 3),
        "cat_catalyst": round(cs.catalyst, 3),
        "cat_macro_breadth": round(cs.macro_breadth, 3),
        "cat_liquidity": round(cs.liquidity, 3),
    }


# ---------------------------------------------------------------------------
# Find latest CSV
# ---------------------------------------------------------------------------

def find_latest_csv(output_dir: Path) -> Optional[Path]:
    candidates = sorted(output_dir.glob("scanner_90d_*.csv"), reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Win-rate helpers
# ---------------------------------------------------------------------------

def _win_rate(rows: list[dict], horizon: str) -> tuple[int, int, float]:
    """Return (n_valid, n_wins, win_pct) for the given horizon."""
    col = f"correct_{horizon}"
    valid = [r for r in rows if r.get(col) is not None and str(r.get(col)).lower() not in ("nan", "none", "")]
    wins = [r for r in valid if str(r.get(col)).lower() in ("true", "1")]
    n = len(valid)
    w = len(wins)
    return n, w, (w / n * 100 if n > 0 else 0.0)


def _ci_str(wins: int, n: int) -> str:
    lo, hi = wilson_ci(wins, n)
    return f"[{lo*100:.1f}, {hi*100:.1f}]"


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(graded: list[dict], total_input: int) -> None:
    sep = "=" * 56

    n_graded = len(graded)

    # Overall 1d
    n_all_1d, w_all_1d, pct_all_1d = _win_rate(graded, "1d")
    n_all_5d, w_all_5d, pct_all_5d = _win_rate(graded, "5d")

    print(f"\n{sep}")
    print(f"A+ BACKTEST — 90 DAYS, N={total_input} input / {n_graded} graded")
    print(f"{sep}\n")

    # -------------------------------------------------------------------
    # GRADE × WIN RATE
    # -------------------------------------------------------------------
    print("GRADE × WIN RATE")
    print(f"  {'Grade':<6} | {'n':>5} | {'1d Win% [95% CI]':<26} | {'5d Win% [95% CI]'}")
    print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*26}-+-{'-'*26}")

    grade_order = ["A+", "A", "B+", "B", "F"]
    for g in grade_order:
        rows = [r for r in graded if r.get("grade") == g]
        n1, w1, p1 = _win_rate(rows, "1d")
        n5, w5, p5 = _win_rate(rows, "5d")
        ci1 = _ci_str(w1, n1)
        ci5 = _ci_str(w5, n5)
        print(f"  {g:<6} | {len(rows):>5} | {p1:>5.1f}% {ci1:<20} | {p5:>5.1f}% {ci5}")

    # ALL row
    ci1_all = _ci_str(w_all_1d, n_all_1d)
    ci5_all = _ci_str(w_all_5d, n_all_5d)
    print(f"  {'ALL':<6} | {n_graded:>5} | {pct_all_1d:>5.1f}% {ci1_all:<20} | {pct_all_5d:>5.1f}% {ci5_all}")

    # -------------------------------------------------------------------
    # COMPOSITE SCORE × WIN RATE (deciles)
    # -------------------------------------------------------------------
    print(f"\nCOMPOSITE SCORE × WIN RATE (deciles)")
    scores = [r["composite_score"] for r in graded]
    decile_edges = np.percentile(scores, np.arange(0, 110, 10)) if scores else []

    print(f"  {'Decile':<7} | {'Score range':<18} | {'n':>5} | {'1d Win%':>8} | {'5d Win%':>8}")
    print(f"  {'-'*7}-+-{'-'*18}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}")

    for i in range(10):
        lo_v = decile_edges[i]
        hi_v = decile_edges[i + 1]
        if i == 9:
            bucket = [r for r in graded if r["composite_score"] >= lo_v]
        else:
            bucket = [r for r in graded if lo_v <= r["composite_score"] < hi_v]
        if not bucket:
            continue
        n1, w1, p1 = _win_rate(bucket, "1d")
        n5, w5, p5 = _win_rate(bucket, "5d")
        label = f"D{i+1:02d}"
        range_str = f"{lo_v:.1f}–{hi_v:.1f}"
        print(f"  {label:<7} | {range_str:<18} | {len(bucket):>5} | {p1:>7.1f}% | {p5:>7.1f}%")

    # -------------------------------------------------------------------
    # CATEGORY × WIN RATE (1d, top-tertile vs bottom-tertile)
    # -------------------------------------------------------------------
    print(f"\nCATEGORY × WIN RATE (1d, top-tertile vs bottom-tertile)")
    print(f"  {'category':<16} | {'top%':>6} | {'bot%':>6} | {'lift':>6}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    categories = [
        ("technical",     "cat_technical"),
        ("vol_vix",       "cat_vol_vix"),
        ("catalyst",      "cat_catalyst"),
        ("macro_breadth", "cat_macro_breadth"),
        ("liquidity",     None),   # neutralized — skip
    ]

    for cat_label, col in categories:
        if col is None:
            print(f"  {'liquidity':<16} | (skipped — neutralized in backtest)")
            continue
        vals = [r[col] for r in graded]
        t33 = float(np.percentile(vals, 66.7))
        b33 = float(np.percentile(vals, 33.3))
        top_rows = [r for r in graded if r[col] >= t33]
        bot_rows = [r for r in graded if r[col] <= b33]
        nt, wt, pt = _win_rate(top_rows, "1d")
        nb, wb, pb = _win_rate(bot_rows, "1d")
        lift = pt - pb
        print(
            f"  {cat_label:<16} | {pt:>5.1f}% | {pb:>5.1f}% | {lift:>+5.1f}%"
        )

    # -------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------
    aplus_rows = [r for r in graded if r.get("grade") == "A+"]
    n_aplus_1d, w_aplus_1d, p_aplus_1d = _win_rate(aplus_rows, "1d")

    print(f"\nVERDICT")
    print(f"  A+ setups: {len(aplus_rows)} ({len(aplus_rows)/n_graded*100:.1f}% of graded)")
    print(f"  A+ 1d win rate: {p_aplus_1d:.1f}% (n={n_aplus_1d})")
    if n_aplus_1d > 0:
        ci_lo, ci_hi = wilson_ci(w_aplus_1d, n_aplus_1d)
        print(f"  A+ 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print(f"  Overall baseline 1d win rate (all graded): {pct_all_1d:.1f}%")
        lift = p_aplus_1d - pct_all_1d
        print(f"  A+ lift vs baseline: {lift:+.1f}%")

    print()
    if n_aplus_1d == 0:
        verdict = "INSUFFICIENT DATA — no A+ setups in sample to evaluate."
    elif p_aplus_1d >= 60.0:
        verdict = "SCORER HAS EDGE (proceed to forward test)"
    elif p_aplus_1d >= 50.0:
        verdict = "MARGINAL (forward test with caution)"
    else:
        verdict = "SCORER HAS NO HISTORICAL EDGE (kill or rebuild)"

    print(f"  >>> {verdict}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def save_csv(graded: list[dict], output_dir: Path) -> Path:
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = output_dir / f"aplus_grades_90d_{ts_str}.csv"
    if not graded:
        return out_path
    fieldnames = list(graded[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(graded)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="A+ confluence scorer grading on 90d historical scanner setups"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to scanner_90d CSV. Defaults to most recent in output/backtest/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default {_DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Resolve input CSV
    input_csv: Optional[Path] = args.csv
    if input_csv is None:
        input_csv = find_latest_csv(_DEFAULT_OUTPUT_DIR)
    if input_csv is None or not input_csv.exists():
        print(f"ERROR: No input CSV found. Pass --csv <path> or run scanner_90d_backtest first.", file=sys.stderr)
        return 1

    print(f"\nInput CSV : {input_csv}")
    print(f"Output dir: {args.output_dir}")

    # Load CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as exc:
        print(f"ERROR: Cannot read CSV: {exc}", file=sys.stderr)
        return 1

    # Normalize bool columns that pandas may read as strings or mixed
    for col in ("correct_1d", "correct_5d", "weekly_ribbon_agreement"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: True if str(v).lower() == "true"
                else (False if str(v).lower() == "false" else None)
            )

    rows = df.to_dict(orient="records")
    print(f"  {len(rows)} rows loaded")

    # Pre-load helper data
    try:
        helpers = load_helper_data(_CACHE_DIR)
    except Exception as exc:
        print(f"ERROR: Failed to load helper data: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    # Grade each row
    print(f"\nGrading {len(rows)} setups …")
    graded: list[dict] = []
    errors = 0
    for i, row in enumerate(rows):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(rows)} graded …")
        try:
            graded.append(grade_row(row, helpers))
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  WARNING: row {i} ({row.get('ticker')} {row.get('as_of')}): {exc}")

    print(f"  Done: {len(graded)} graded, {errors} errors\n")

    if not graded:
        print("No rows graded — cannot produce summary.")
        return 1

    # Console summary
    print_summary(graded, total_input=len(rows))

    # Save output CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        out_path = save_csv(graded, args.output_dir)
        print(f"Per-setup grades saved → {out_path}")
    except Exception as exc:
        print(f"WARNING: CSV save failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
