"""90-trading-day backtest of stock scanner setup detection.

For each of the last N trading days:
  1. Pull OHLCV history ending on that date (fetched once per ticker, sliced in memory)
  2. Run detect_setups (the 4 detectors) on each ticker's history slice
  3. For each setup, compute the actual next-1d and next-5d % move
  4. Mark direction-correct if move sign matches setup direction

Aggregate metrics by:
  - Overall (all detectors, all tickers, all days)
  - Per detector type (compression vs pullback vs stage_2 vs failed_breakdown)
  - Per direction (bullish vs bearish setups)
  - Per ticker
  - Per setup strength bucket (0.5-0.7, 0.7-0.9, 0.9-1.0)
  - VIX regime at setup time (expansion/contraction/neutral)
  - weekly_ribbon_agreement at setup time (True/False)

Output: CSV per-setup detail + console summary table.

CLI:
    python3 -m scripts.scanner_90d_backtest --days 90 --universe MEGA
    python3 -m scripts.scanner_90d_backtest --days 90 --universe ETFS --use-cache
    python3 -m scripts.scanner_90d_backtest --days 90 --universe SP500 --output-dir /tmp/bt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from domain.technical_signals import detect_setups
from domain.market_context import compute_market_context
from domain.equity.vix_regime import compute_vix_context
from data.sources.yahoo_macro_source import YahooMacroSource

# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV",
    "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
]
_MEGA = _ETFS + [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "COST", "UNH", "AMD",
]
_SP500 = _MEGA + [
    "BRK-B", "LLY", "AVGO", "WMT", "HD", "PG", "MRK", "CVX",
    "ABBV", "BAC", "KO", "PEP", "ORCL", "NFLX", "CRM", "TMO",
    "CSCO", "ACN", "MCD", "LIN", "ABT", "GE", "NOW", "DHR",
    "PM", "IBM", "CAT", "GS", "TXN", "ISRG", "QCOM", "BKNG",
    "SPGI", "RTX", "MS", "AMGN", "SYK", "PLD", "INTU",
    "AMAT", "UNP", "BLK", "AXP", "C", "DE", "GILD", "ADI",
    "SCHW", "TJX", "MDLZ", "REGN", "SO", "ZTS", "MMC", "EOG",
    "WFC", "CME", "DUK", "SLB", "CL", "ITW", "APD", "PH",
    "PYPL", "HCA", "AON", "EQIX", "MCO", "KLAC", "MO", "NSC",
    "ADP", "F", "GM", "NKE", "EL", "SBUX", "FCX", "OXY",
    "MET", "EMR", "ICE", "MSI", "KMB", "EW", "PSA", "CARR",
    "MNST", "HLT", "PCAR", "LRCX", "CDNS", "SNPS", "WM", "GWW",
    "CMG", "IDXX", "CTAS", "CCI", "AMT", "EXC", "AEP", "COP",
    "D", "KHC", "VRTX", "MAR", "ORLY", "HPQ", "TGT", "CVS",
    "ALL", "CTSH", "HES", "STZ", "BDX", "ROK", "A", "SHW",
    "ROST", "MCHP", "DXCM", "KEYS", "BIIB", "KR", "VLO", "MPC",
    "PSX", "GIS", "AFL", "LHX", "CI", "GPC", "VRSK", "CSGP",
    "ILMN", "NXPI", "TROW", "DOW", "NEM", "DD", "BSX", "PPG",
    "O", "HSY", "MTD", "ZBRA", "FAST", "EXPD", "PAYX", "ETSY",
    "ANSS", "CPRT", "WTW", "CBOE", "TTWO", "FDS", "POOL", "TECH",
    "DG", "TDG", "HIG", "DLR", "FTV", "TSCO", "MPWR", "RMD",
    "WAT", "TT", "PWR", "ETN", "ACGL", "PEG", "MLM", "VMC",
    "FI", "FIS", "XYL", "WY", "CE", "LYB", "ALB", "CTVA",
    "NUE", "RS", "IP", "AVY", "SEE", "IFF", "CF", "MOS",
    "FMC", "RPM", "EMN", "WLK", "OLN", "TPC", "ASH", "HUN",
    "TSN", "HRL", "CPB", "SJM", "MKC", "CAG", "K", "GIS",
    "TAP", "RAD", "SAFM", "LANC", "JJSF", "THS", "CALM",
    "CHEF", "USFD", "SFM", "FRSH", "PFGC", "CASY", "DINE",
    "JACK", "TXRH", "DENN", "BLMN", "RBI", "YUM", "MCD",
    "WEN", "CMG", "DRI", "EAT", "CAKE", "CHUY", "KRUS",
    "ARCO", "PLNT", "FAT", "RRBI", "RRGB", "BJRI", "JBSS",
    "PZZA", "DPZ", "QSR", "SYY", "USM", "TMUS", "VZ", "T",
    "DISH", "SIRI", "WBD", "PARA", "FOX", "FOXA", "NYT",
    "NWS", "NWSA", "LYV", "MSGS", "ALS", "CZR", "MGM", "WYNN",
    "LVS", "PENN", "DKNG", "RSI", "GAN", "GENI", "EVRI", "AGS",
]

_UNIVERSES = {
    "ETFS": _ETFS,
    "MEGA": _MEGA,
    "SP500": _SP500,
}

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_DIR = _ROOT / "output" / "backtest"
_CACHE_DIR = _ROOT / "cache" / "backtest_ohlcv"


# ---------------------------------------------------------------------------
# NYSE trading-day calendar (approximation without pandas_market_calendars)
# ---------------------------------------------------------------------------

def _us_federal_holidays(year: int) -> set[date]:
    """Approximate US federal market holidays for a given year."""
    holidays: set[date] = set()

    def _nearest_weekday(d: date) -> date:
        """Move weekend dates to nearest weekday (Mon if Sun, Fri if Sat)."""
        wd = d.weekday()
        if wd == 5:  # Saturday → Friday
            return d - timedelta(days=1)
        if wd == 6:  # Sunday → Monday
            return d + timedelta(days=1)
        return d

    # New Year's Day
    holidays.add(_nearest_weekday(date(year, 1, 1)))
    # MLK Day (3rd Monday of January)
    d = date(year, 1, 1)
    mondays = [date(year, 1, d.day + i) for i in range(31) if date(year, 1, min(31, d.day + i)).weekday() == 0]
    mondays_jan = [date(year, 1, day) for day in range(1, 32) if date(year, 1, day).weekday() == 0]
    if len(mondays_jan) >= 3:
        holidays.add(mondays_jan[2])
    # Presidents Day (3rd Monday of February)
    mondays_feb = [date(year, 2, day) for day in range(1, 29 if year % 4 != 0 else 30) if date(year, 2, day).weekday() == 0]
    if len(mondays_feb) >= 3:
        holidays.add(mondays_feb[2])
    # Good Friday (Easter - 2 days) — approximate with known offsets
    # Use a simple Gauss Easter algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day_num = ((h + ll - 7 * m + 114) % 31) + 1
    easter = date(year, month, day_num)
    good_friday = easter - timedelta(days=2)
    holidays.add(good_friday)
    # Memorial Day (last Monday of May)
    mondays_may = [date(year, 5, day) for day in range(1, 32) if date(year, 5, day).weekday() == 0]
    if mondays_may:
        holidays.add(mondays_may[-1])
    # Juneteenth (June 19)
    if year >= 2022:
        holidays.add(_nearest_weekday(date(year, 6, 19)))
    # Independence Day (July 4)
    holidays.add(_nearest_weekday(date(year, 7, 4)))
    # Labor Day (first Monday of September)
    mondays_sep = [date(year, 9, day) for day in range(1, 31) if date(year, 9, day).weekday() == 0]
    if mondays_sep:
        holidays.add(mondays_sep[0])
    # Thanksgiving (4th Thursday of November)
    thursdays_nov = [date(year, 11, day) for day in range(1, 31) if date(year, 11, day).weekday() == 3]
    if len(thursdays_nov) >= 4:
        holidays.add(thursdays_nov[3])
    # Christmas (Dec 25)
    holidays.add(_nearest_weekday(date(year, 12, 25)))

    return holidays


def get_trading_days(start: date, end: date) -> list[date]:
    """Return NYSE trading days in [start, end] inclusive."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        return [d.date() for d in schedule.index]
    except Exception:
        # Incompatible version or not installed — fall through to manual calendar
        pass

    # Fallback: business days minus federal holidays
    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        holidays |= _us_federal_holidays(year)

    days: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5 and cur not in holidays:  # Mon-Fri, not a holiday
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Wilson confidence interval
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
# OHLCV fetch + cache
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker.replace('-', '_')}.parquet"


def fetch_ohlcv(
    ticker: str,
    lookback_days: int = 730,
    cache_dir: Path = _CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch full OHLCV for ticker (2 years), cache to parquet."""
    cp = _cache_path(ticker, cache_dir)
    if use_cache and cp.exists():
        df = pd.read_parquet(cp)
        print(f"    {ticker}: loaded {len(df)} rows from cache")
        return df

    # Try yfinance directly (most reliable for 2yr daily OHLCV)
    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
        if raw.empty:
            raise ValueError("empty")
        df = raw.rename(columns={c: c.lower() for c in raw.columns})
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].astype(float)
        df = df.dropna(subset=["close"])
        df.to_parquet(cp)
        print(f"    {ticker}: fetched {len(df)} rows → cached")
        return df
    except Exception as exc:
        print(f"    {ticker}: yfinance failed ({exc}), trying yahooquery …")

    # Fallback to yahooquery
    try:
        from yahooquery import Ticker as YqTicker
        raw = YqTicker(ticker).history(period="2y")
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(ticker, level="symbol", drop_level=True)
        rename = {c: c.lower() for c in raw.columns}
        df = raw.rename(columns=rename)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].astype(float)
        df = df.dropna(subset=["close"])
        df.to_parquet(cp)
        print(f"    {ticker}: fetched {len(df)} rows via yahooquery → cached")
        return df
    except Exception as exc2:
        print(f"    {ticker}: all fetches failed: {exc2}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# VIX fetch
# ---------------------------------------------------------------------------

def fetch_vix(use_cache: bool = True) -> pd.Series:
    """Return daily VIX close series with tz-naive DatetimeIndex."""
    macro = YahooMacroSource()
    df = macro.fetch_history("^VIX", period="2y")
    if df.empty:
        return pd.Series(dtype=float)
    close = df["close"].copy()
    if close.index.tz is not None:
        close.index = close.index.tz_convert("UTC").tz_localize(None)
    return close


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _win_rate_row(
    label: str,
    rows: list[dict],
    horizon: str = "1d",
) -> dict:
    col = f"correct_{horizon}"
    n = sum(1 for r in rows if r[col] is not None)
    wins = sum(1 for r in rows if r[col] is True)
    win_pct = (wins / n * 100) if n > 0 else 0.0
    ci = wilson_ci(wins, n)
    rets = [r[f"ret_{horizon}"] for r in rows if r[f"ret_{horizon}"] is not None]
    mean_ret = float(np.mean(rets) * 100) if rets else 0.0
    median_ret = float(np.median(rets) * 100) if rets else 0.0
    return {
        "label": label,
        "n": n,
        "wins": wins,
        "win_pct": win_pct,
        "ci_lo": ci[0] * 100,
        "ci_hi": ci[1] * 100,
        "mean_ret_pct": mean_ret,
        "median_ret_pct": median_ret,
    }


def _fmt_row(r: dict) -> str:
    ci_str = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
    lift = ""
    if "lift" in r:
        lift = f"  ← lift vs overall {r['lift']:+.1f}%"
    return (
        f"  {r['label']:<40} | n={r['n']:>5} | "
        f"Win %: {r['win_pct']:.1f}% {ci_str:<20}{lift}"
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    results: list[dict],
    universe_name: str,
    universe: list[str],
    start_date: date,
    end_date: date,
    trading_days: list[date],
) -> None:
    if not results:
        print("No setups detected — nothing to report.")
        return

    sep = "─" * 78

    # Count setups per detector
    by_detector: dict[str, list[dict]] = {}
    for r in results:
        by_detector.setdefault(r["setup_name"], []).append(r)

    detector_names = [
        "compression_breakout",
        "pullback_in_trend",
        "stage_2_breakout",
        "failed_breakdown_reversal",
    ]

    print(f"\n{'='*78}")
    print(f"=== 90-DAY SCANNER BACKTEST ({start_date} → {end_date}) ===")
    print(f"{'='*78}\n")
    print(f"Universe: {universe_name} ({len(universe)} tickers)")
    print(f"Trading days evaluated: {len(trading_days)}")
    print(f"Total setups detected: {len(results):,}")
    for d in detector_names:
        n = len(by_detector.get(d, []))
        print(f"  {d:<35}: {n}")

    # Overall
    print(f"\n{sep}")
    print("OVERALL DIRECTION WIN RATE")
    print(sep)
    print(f"  {'Horizon':<8} | {'Setups':>7} | {'Wins':>6} | {'Win %':>7} | {'95% CI':<22} | {'Mean ret':>9} | {'Median ret':>10}")

    for horizon in ("1d", "5d"):
        col = f"correct_{horizon}"
        valid = [r for r in results if r[col] is not None]
        n = len(valid)
        wins = sum(1 for r in valid if r[col] is True)
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        rets = [r[f"ret_{horizon}"] for r in valid if r[f"ret_{horizon}"] is not None]
        mean_ret = float(np.mean(rets) * 100) if rets else 0.0
        median_ret = float(np.median(rets) * 100) if rets else 0.0
        ci_str = f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"
        print(
            f"  {horizon:<8} | {n:>7,} | {wins:>6,} | {win_pct:>6.1f}% | {ci_str:<22} | "
            f"{mean_ret:>+8.2f}% | {median_ret:>+9.2f}%"
        )

    # Per detector
    print(f"\n{sep}")
    print("PER DETECTOR")
    print(sep)
    print(f"  {'Detector':<35} | {'1d Win % [CI]':<30} | {'5d Win % [CI]'}")

    for d in detector_names:
        rows = by_detector.get(d, [])
        if not rows:
            print(f"  {d:<35} | {'n/a':<30} | n/a")
            continue
        for_1d = [r for r in rows if r["correct_1d"] is not None]
        for_5d = [r for r in rows if r["correct_5d"] is not None]
        n1 = len(for_1d); w1 = sum(1 for r in for_1d if r["correct_1d"] is True)
        n5 = len(for_5d); w5 = sum(1 for r in for_5d if r["correct_5d"] is True)
        ci1 = wilson_ci(w1, n1); ci5 = wilson_ci(w5, n5)
        p1 = (w1 / n1 * 100) if n1 > 0 else 0.0
        p5 = (w5 / n5 * 100) if n5 > 0 else 0.0
        s1 = f"{p1:.1f}% (n={n1}) [{ci1[0]*100:.1f},{ci1[1]*100:.1f}]"
        s5 = f"{p5:.1f}% (n={n5}) [{ci5[0]*100:.1f},{ci5[1]*100:.1f}]"
        print(f"  {d:<35} | {s1:<30} | {s5}")

    # Per direction
    print(f"\n{sep}")
    print("PER DIRECTION (1-day win rate)")
    print(sep)
    for direction in ("bullish", "bearish"):
        rows = [r for r in results if r["direction"] == direction and r["correct_1d"] is not None]
        n = len(rows); wins = sum(1 for r in rows if r["correct_1d"] is True)
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        print(f"  {direction.capitalize():<10} | n={n:>5} | Win %: {win_pct:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    # Per strength bucket
    print(f"\n{sep}")
    print("PER STRENGTH BUCKET (1-day win rate)")
    print(sep)
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    bucket_labels = ["0.00-0.50", "0.50-0.70", "0.70-0.90", "0.90-1.00"]
    for (lo, hi), lbl in zip(buckets, bucket_labels):
        rows = [
            r for r in results
            if lo <= r["strength"] < hi and r["correct_1d"] is not None
        ]
        if not rows:
            continue
        n = len(rows); wins = sum(1 for r in rows if r["correct_1d"] is True)
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        print(f"  Strength {lbl} | n={n:>5} | Win %: {win_pct:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    # VIX regime
    print(f"\n{sep}")
    print("VIX REGIME AT SETUP TIME (1-day win rate)")
    print(sep)

    overall_1d_valid = [r for r in results if r["correct_1d"] is not None]
    overall_1d_wins = sum(1 for r in overall_1d_valid if r["correct_1d"] is True)
    overall_wr = (overall_1d_wins / len(overall_1d_valid) * 100) if overall_1d_valid else 0.0

    for regime in ("expansion", "neutral", "contraction"):
        rows = [r for r in results if r["vix_regime"] == regime and r["correct_1d"] is not None]
        n = len(rows); wins = sum(1 for r in rows if r["correct_1d"] is True)
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        lift = win_pct - overall_wr
        lift_str = f"  ← lift vs overall {lift:+.1f}%"
        print(f"  {regime.upper():<12} | n={n:>5} | Win %: {win_pct:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]{lift_str}")

    # Multi-TF ribbon agreement
    print(f"\n{sep}")
    print("MULTI-TIMEFRAME RIBBON AGREEMENT (1-day win rate)")
    print(sep)
    for agreed in (True, False):
        rows = [r for r in results if r["weekly_ribbon_agreement"] == agreed and r["correct_1d"] is not None]
        n = len(rows); wins = sum(1 for r in rows if r["correct_1d"] is True)
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        lbl = "weekly_ribbon_agreement=True " if agreed else "weekly_ribbon_agreement=False"
        print(f"  {lbl} | n={n:>5} | Win %: {win_pct:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    # Agreement lift
    rows_true = [r for r in results if r["weekly_ribbon_agreement"] is True and r["correct_1d"] is not None]
    rows_false = [r for r in results if r["weekly_ribbon_agreement"] is False and r["correct_1d"] is not None]
    nt = len(rows_true); wt = sum(1 for r in rows_true if r["correct_1d"] is True)
    nf = len(rows_false); wf = sum(1 for r in rows_false if r["correct_1d"] is True)
    pt = (wt / nt * 100) if nt > 0 else 0.0
    pf = (wf / nf * 100) if nf > 0 else 0.0
    if nt > 0 and nf > 0:
        print(f"\n  → multi-tf agreement adds {pt - pf:+.1f}% win rate (the v2 multiplier opportunity)")

    # Per ticker (top 10 by n)
    print(f"\n{sep}")
    print("PER TICKER — TOP 10 BY SETUP COUNT (1-day win rate)")
    print(sep)
    by_ticker: dict[str, list[dict]] = {}
    for r in results:
        by_ticker.setdefault(r["ticker"], []).append(r)
    ticker_stats = []
    for tkr, rows in by_ticker.items():
        valid = [r for r in rows if r["correct_1d"] is not None]
        n = len(valid); wins = sum(1 for r in valid if r["correct_1d"] is True)
        ticker_stats.append((tkr, n, wins))
    ticker_stats.sort(key=lambda x: x[1], reverse=True)
    for tkr, n, wins in ticker_stats[:15]:
        win_pct = (wins / n * 100) if n > 0 else 0.0
        ci = wilson_ci(wins, n)
        print(f"  {tkr:<8} | n={n:>4} | Win %: {win_pct:.1f}% [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")

    # Verdict
    print(f"\n{sep}")
    print("VERDICT")
    print(sep)

    overall_n = len(overall_1d_valid)
    overall_pct = overall_wr
    overall_ci = wilson_ci(overall_1d_wins, overall_n)

    best_detector = None
    best_wr_1d = 0.0
    for d in detector_names:
        rows = [r for r in by_detector.get(d, []) if r["correct_1d"] is not None]
        if not rows:
            continue
        n = len(rows); w = sum(1 for r in rows if r["correct_1d"] is True)
        wr = (w / n * 100) if n > 0 else 0.0
        if wr > best_wr_1d:
            best_wr_1d = wr
            best_detector = d

    edge_verdict = ""
    if overall_ci[0] * 100 > 50:
        edge_verdict = "REAL EDGE — lower bound of 95% CI exceeds 50%."
    elif overall_pct >= 58:
        edge_verdict = "POSSIBLE EDGE — win rate ≥58% but CI dips below 50%. Need more data."
    elif overall_pct >= 52:
        edge_verdict = "MARGINAL — win rate positive but CI overlaps 50%; not statistically confirmed."
    else:
        edge_verdict = "NO EDGE — win rate ≤52%, consistent with noise."

    print(f"\n  Overall 1d win rate: {overall_pct:.1f}% (n={overall_n:,}) {edge_verdict}")
    if best_detector:
        print(f"  Strongest detector: {best_detector} at {best_wr_1d:.1f}% 1d win rate")
    if nt > 0 and nf > 0 and abs(pt - pf) > 1:
        better = "with" if pt > pf else "without"
        print(f"  Weekly ribbon agreement ({better}) adds {abs(pt - pf):.1f}% lift — use as filter in v2")

    print(f"\n{'='*78}\n")


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    universe: list[str],
    universe_name: str,
    days: int,
    output_dir: Path,
    use_cache: bool,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today()
    end_date = today
    start_date = today - timedelta(days=days + 30)  # extra buffer for calendar calc

    # Get trading days
    trading_days_all = get_trading_days(start_date, end_date)
    # Keep last N
    trading_days = trading_days_all[-days:] if len(trading_days_all) > days else trading_days_all
    bt_start = trading_days[0] if trading_days else start_date
    bt_end = trading_days[-1] if trading_days else end_date

    print(f"\n{'='*78}")
    print(f"=== SCANNER 90-DAY BACKTEST ===")
    print(f"  Universe:      {universe_name} ({len(universe)} tickers)")
    print(f"  Backtest days: {len(trading_days)} trading days ({bt_start} → {bt_end})")
    print(f"  Cache dir:     {_CACHE_DIR}")
    print(f"  Output dir:    {output_dir}")
    print(f"{'='*78}\n")

    # Fetch VIX
    print("Fetching VIX history …")
    try:
        vix_series = fetch_vix(use_cache=use_cache)
        print(f"  VIX: {len(vix_series)} daily rows")
    except Exception as exc:
        print(f"  VIX fetch failed: {exc} — regime will be 'neutral' everywhere")
        vix_series = pd.Series(dtype=float)

    # Fetch OHLCV for all tickers
    print(f"\nFetching OHLCV for {len(universe)} tickers …")
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    for ticker in universe:
        try:
            df = fetch_ohlcv(ticker, lookback_days=730, cache_dir=_CACHE_DIR, use_cache=use_cache)
            if not df.empty:
                # Normalize index to tz-naive
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                ohlcv_cache[ticker] = df
        except Exception as exc:
            print(f"    {ticker}: fetch error — {exc}")

    print(f"\n  {len(ohlcv_cache)}/{len(universe)} tickers ready\n")

    # Main loop
    results: list[dict] = []
    _MIN_HISTORY = 220

    total_iterations = len(trading_days) * len(ohlcv_cache)
    n_done = 0
    print(f"Running detector loop: {len(trading_days)} days × {len(ohlcv_cache)} tickers = {total_iterations:,} iterations …\n")

    for day_idx, as_of in enumerate(trading_days):
        if day_idx % 10 == 0:
            pct = day_idx / len(trading_days) * 100
            print(f"  Day {day_idx+1}/{len(trading_days)} ({pct:.0f}%) — as_of={as_of} — setups so far: {len(results):,}")

        # VIX at this date
        vix_slice = vix_series[vix_series.index.normalize() <= pd.Timestamp(as_of)] if len(vix_series) > 0 else pd.Series(dtype=float)
        vix_context = compute_vix_context(vix_slice.tail(35)) if len(vix_slice) >= 10 else None

        for ticker, full_df in ohlcv_cache.items():
            n_done += 1

            # Slice history up to (and including) as_of
            as_of_ts = pd.Timestamp(as_of)
            history_slice = full_df[full_df.index.normalize() <= as_of_ts]

            if len(history_slice) < _MIN_HISTORY:
                continue

            # Run detectors
            try:
                setups = detect_setups(history_slice)
            except Exception:
                continue

            if not setups:
                continue

            spot_at_setup = float(history_slice["close"].iloc[-1])

            # Future bars for outcome
            future_bars = full_df[full_df.index.normalize() > as_of_ts].head(5)
            if future_bars.empty:
                continue

            ret_1d: Optional[float] = None
            ret_5d: Optional[float] = None
            correct_1d: Optional[bool] = None
            correct_5d: Optional[bool] = None

            if len(future_bars) >= 1 and spot_at_setup > 0:
                ret_1d = float(future_bars["close"].iloc[0] / spot_at_setup - 1)
            if len(future_bars) >= 5 and spot_at_setup > 0:
                ret_5d = float(future_bars["close"].iloc[-1] / spot_at_setup - 1)

            # Market context
            try:
                ctx = compute_market_context(history_slice["close"])
            except Exception:
                from domain.market_context import _NEUTRAL
                ctx = _NEUTRAL

            vix_regime = vix_context.regime if vix_context else "neutral"
            vix_now = vix_context.vix_now if vix_context else float("nan")

            for setup in setups:
                if ret_1d is not None:
                    correct_1d = (
                        (setup.direction == "bullish" and ret_1d > 0) or
                        (setup.direction == "bearish" and ret_1d < 0)
                    )
                if ret_5d is not None:
                    correct_5d = (
                        (setup.direction == "bullish" and ret_5d > 0) or
                        (setup.direction == "bearish" and ret_5d < 0)
                    )

                results.append({
                    "as_of": as_of.isoformat(),
                    "ticker": ticker,
                    "setup_name": setup.setup_name,
                    "direction": setup.direction,
                    "strength": round(setup.strength, 4),
                    "spot_at_setup": round(spot_at_setup, 4),
                    "ret_1d": round(ret_1d, 6) if ret_1d is not None else None,
                    "ret_5d": round(ret_5d, 6) if ret_5d is not None else None,
                    "correct_1d": correct_1d,
                    "correct_5d": correct_5d,
                    "vix_at_setup": round(vix_now, 2) if not math.isnan(vix_now) else None,
                    "vix_regime": vix_regime,
                    "weekly_trend": ctx.weekly_trend,
                    "weekly_ribbon_agreement": ctx.weekly_ribbon_agreement,
                    "pct_change_1w": round(ctx.pct_change_1w, 6),
                    "pct_change_4w": round(ctx.pct_change_4w, 6),
                })

    print(f"\n  Loop complete. {len(results):,} setup records collected.\n")
    return results, trading_days, bt_start, bt_end


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], output_dir: Path) -> Path:
    from datetime import datetime as dt
    ts_str = dt.now().strftime("%Y%m%d_%H%M")
    csv_path = output_dir / f"scanner_90d_{ts_str}.csv"
    fieldnames = [
        "as_of", "ticker", "setup_name", "direction", "strength", "spot_at_setup",
        "ret_1d", "ret_5d", "correct_1d", "correct_5d", "vix_at_setup", "vix_regime",
        "weekly_trend", "weekly_ribbon_agreement", "pct_change_1w", "pct_change_4w",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="90-day scanner setup-detection backtest"
    )
    parser.add_argument("--days", type=int, default=90, help="Trading days to backtest (default 90)")
    parser.add_argument(
        "--universe",
        choices=["ETFS", "MEGA", "SP500"],
        default="MEGA",
        help="Ticker universe (default MEGA)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached parquet OHLCV if available (default True)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Force re-fetch even if cache exists",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    universe = _UNIVERSES[args.universe]

    try:
        results, trading_days, bt_start, bt_end = run_backtest(
            universe=universe,
            universe_name=args.universe,
            days=args.days,
            output_dir=args.output_dir,
            use_cache=use_cache,
        )
    except Exception as exc:
        print(f"ERROR: Backtest failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    if not results:
        print("No setups detected in the backtest window. Check OHLCV data quality.")
        return 0

    # Console summary
    print_summary(
        results=results,
        universe_name=args.universe,
        universe=universe,
        start_date=bt_start,
        end_date=bt_end,
        trading_days=trading_days,
    )

    # CSV
    try:
        csv_path = save_csv(results, args.output_dir)
        print(f"Per-setup detail saved → {csv_path}")
    except Exception as exc:
        print(f"WARNING: CSV save failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
