"""3-strategy backtest comparison: A (vol-model only) vs B (research-paper ensemble)
vs A+B (combined).

For each historical Kalshi BTC market in the date range:
  1. At the market's open time, replay what each strategy would have predicted.
  2. Compare to actual resolution.
  3. Aggregate P&L per strategy assuming fixed position size.

CLI
---
    python3 -m scripts.kalshi_btc_backtest \\
        --start 2026-01-01 --end 2026-05-31 \\
        --position-size 50

Output
------
  Console: comparison table
  CSV:     output/kalshi/backtest_YYYYMMDD_HHMM.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.kalshi_source import KalshiSource
from data.sources.coinbase_source import CoinbaseSource
from domain.prediction.market import KalshiMarket, parse_kalshi_market
from engine.prediction.ensemble import (
    compute_strategy_A,
    compute_strategy_B,
    compute_strategy_AB,
    count_estimator_agreement,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_EDGE = 0.05
_KALSHI_FEE_RATE = 0.07
_OUTPUT_DIR = _ROOT / "output" / "kalshi"

# Historical vol quantile keys
_Q = {0.25: 0.25, 0.5: 0.5, 0.75: 0.75}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_ohlcv_at(ohlcv: pd.DataFrame, as_of: datetime, lookback_hours: int) -> pd.DataFrame:
    """Return hourly OHLCV rows ending at (but not after) `as_of`."""
    cutoff = pd.Timestamp(as_of).tz_localize(None) if as_of.tzinfo and ohlcv.index.tzinfo is None else as_of
    past = ohlcv[ohlcv.index <= cutoff]
    return past.iloc[-lookback_hours:] if len(past) >= 1 else past


def _vol_quantiles(hourly_returns: pd.Series) -> dict[float, float]:
    hv = hourly_returns.rolling(24).std() * np.sqrt(24)
    hv = hv.dropna()
    if hv.empty:
        return {0.25: 0.01, 0.5: 0.02, 0.75: 0.03}
    return {
        0.25: float(hv.quantile(0.25)),
        0.5:  float(hv.quantile(0.5)),
        0.75: float(hv.quantile(0.75)),
    }


def _pnl_per_dollar(fair_prob: float, market_prob: float, resolved_yes: bool) -> float:
    """Compute realized P&L per dollar risked.

    BUY YES if fair > market_prob, BUY NO (sell YES) if fair < market_prob.
    Payout = $1 on win (binary).  Fee = 7% of winnings.
    """
    if abs(fair_prob - market_prob) < _MIN_EDGE:
        return 0.0  # no trade

    if fair_prob > market_prob:  # BUY YES
        cost = market_prob
        if resolved_yes:
            pnl = (1.0 - market_prob) * (1.0 - _KALSHI_FEE_RATE)
        else:
            pnl = -market_prob
    else:  # BUY NO
        cost = 1.0 - market_prob
        if not resolved_yes:
            pnl = market_prob * (1.0 - _KALSHI_FEE_RATE)
        else:
            pnl = -(1.0 - market_prob)

    return pnl / cost if cost > 0 else 0.0


def _sharpe(pnl_series: list[float]) -> float:
    if len(pnl_series) < 2:
        return 0.0
    arr = np.array(pnl_series)
    std = arr.std()
    return float(arr.mean() / std) if std > 0 else 0.0


def _max_drawdown(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return float(drawdown.min())


def _print_table(results: list[dict]) -> None:
    print(f"\n{'─'*70}")
    print(f"{'Strategy':<22} {'Trades':>7} {'Win%':>7} {'MeanEdge':>10} "
          f"{'TotalPnL':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"{'─'*70}")
    for r in results:
        print(
            f"  {r['strategy']:<20} {r['n_trades']:>7} {r['win_pct']:>6.1f}% "
            f"{r['mean_edge']:>+9.1%} {r['total_pnl']:>+9.1%} "
            f"{r['sharpe']:>8.2f} {r['max_dd']:>+7.1%}"
        )
    print(f"{'─'*70}\n")


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    settled_markets: list[KalshiMarket],
    ohlcv_1h: pd.DataFrame,
    position_size: float,
) -> tuple[list[dict], list[dict]]:
    """Run 3-strategy backtest over settled_markets.

    Returns (per_trade_rows, summary_rows).
    Each per_trade row: strategy, ticker, fair_prob, market_prob, edge,
        position_size, resolved_yes_no, pnl_per_dollar, cumulative_pnl.
    """
    per_trade: list[dict] = []
    strategy_pnl: dict[str, list[float]] = {"A": [], "B": [], "AB": []}
    strategy_wins: dict[str, int] = {"A": 0, "B": 0, "AB": 0}
    strategy_trades: dict[str, int] = {"A": 0, "B": 0, "AB": 0}
    strategy_edges: dict[str, list[float]] = {"A": [], "B": [], "AB": []}

    total = len(settled_markets)
    for i, m in enumerate(settled_markets, 1):
        if m.resolution not in ("yes", "no"):
            continue

        resolved_yes = m.resolution == "yes"

        # Slice historical data at market open = close_time - hours_to_expiry
        # We don't have open_time, so use close_time as the "as of" proxy
        # and look back 30 days of hourly data
        as_of = m.close_time
        hourly_slice = _slice_ohlcv_at(ohlcv_1h, as_of, 30 * 24)
        if len(hourly_slice) < 50:
            continue

        spot = float(hourly_slice["close"].iloc[-1])
        hourly_returns = np.log(hourly_slice["close"]).diff().dropna()
        quantiles = _vol_quantiles(hourly_returns)
        last_hour_return = float(hourly_returns.iloc[-1]) if len(hourly_returns) > 0 else 0.0
        hv_24 = hourly_returns.rolling(24).std() * np.sqrt(24)
        hourly_sigma = float(hv_24.iloc[-1]) if len(hv_24) > 0 and not np.isnan(hv_24.iloc[-1]) else 0.02

        hours_to_expiry = (m.expiration - as_of).total_seconds() / 3600.0
        if hours_to_expiry <= 0:
            hours_to_expiry = 1.0  # settled market: use 1h as floor for computation

        market_prob = m.market_implied_prob
        if market_prob <= 0 or market_prob >= 1:
            continue

        # Empty 1-min series (not needed for backtest — RV estimator will return 0.5)
        empty_1min = pd.Series(dtype=float)

        # Strategy A
        out_A = compute_strategy_A(
            spot=spot, strike=m.strike, hours_to_expiry=hours_to_expiry,
            vol_proba=0.5,  # no vol model in backtest (model trained after data period)
            historical_vol_quantiles=quantiles,
            expiration_utc=m.expiration, last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        fp_A = out_A.fair_prob_adjusted
        if m.side == "below":
            fp_A = 1.0 - fp_A

        # Strategy B
        out_B = compute_strategy_B(
            spot=spot, strike=m.strike, hours_to_expiry=hours_to_expiry,
            hourly_returns=hourly_returns, minute_returns_last_60=empty_1min,
            expiration_utc=m.expiration, last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        fp_B = out_B.fair_prob_adjusted
        if m.side == "below":
            fp_B = 1.0 - fp_B

        # Strategy A+B
        out_AB = compute_strategy_AB(
            spot=spot, strike=m.strike, hours_to_expiry=hours_to_expiry,
            vol_proba=0.5, historical_vol_quantiles=quantiles,
            hourly_returns=hourly_returns, minute_returns_last_60=empty_1min,
            expiration_utc=m.expiration, last_hour_return=last_hour_return,
            hourly_sigma=hourly_sigma,
        )
        fp_AB = out_AB.fair_prob_adjusted
        if m.side == "below":
            fp_AB = 1.0 - fp_AB

        for strategy, fp in [("A", fp_A), ("B", fp_B), ("AB", fp_AB)]:
            edge = fp - market_prob
            if abs(edge) < _MIN_EDGE:
                continue

            pnl = _pnl_per_dollar(fp, market_prob, resolved_yes)
            strategy_trades[strategy] += 1
            strategy_pnl[strategy].append(pnl)
            strategy_edges[strategy].append(edge)
            if pnl > 0:
                strategy_wins[strategy] += 1

            per_trade.append({
                "strategy": strategy,
                "ticker": m.ticker,
                "as_of": as_of.isoformat(),
                "expiration": m.expiration.isoformat(),
                "strike": m.strike,
                "side": m.side,
                "market_prob": market_prob,
                "fair_prob": fp,
                "edge": edge,
                "position_size": position_size,
                "resolved_yes": resolved_yes,
                "pnl_per_dollar": pnl,
                "pnl_dollars": pnl * position_size,
            })

    # Compute cumulative P&L per strategy
    running: dict[str, float] = {"A": 0.0, "B": 0.0, "AB": 0.0}
    for row in per_trade:
        s = row["strategy"]
        running[s] += row["pnl_dollars"]
        row["cumulative_pnl_dollars"] = running[s]

    # Summary
    summary = []
    for strat_name, label in [("A", "A: Vol-model only"), ("B", "B: Research ensemble"), ("AB", "A+B: Combined")]:
        n = strategy_trades[strat_name]
        pnl_list = strategy_pnl[strat_name]
        total_invested = n * position_size
        total_pnl_pct = (sum(p * position_size for p in pnl_list) / total_invested) if total_invested > 0 else 0.0
        win_pct = (strategy_wins[strat_name] / n * 100) if n > 0 else 0.0
        mean_edge = float(np.mean(strategy_edges[strat_name])) if strategy_edges[strat_name] else 0.0
        sharpe = _sharpe(pnl_list)
        max_dd = _max_drawdown(pnl_list)
        summary.append({
            "strategy": label,
            "n_trades": n,
            "win_pct": win_pct,
            "mean_edge": mean_edge,
            "total_pnl": total_pnl_pct,
            "sharpe": sharpe,
            "max_dd": max_dd,
        })

    return per_trade, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Kalshi BTC 3-strategy backtest")
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-01", help="End date YYYY-MM-DD")
    parser.add_argument("--position-size", type=float, default=50.0, help="Dollars per trade")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    position_size = args.position_size

    print(f"\n{'='*65}")
    print(f"=== Kalshi BTC Backtest: {args.start} → {args.end} ===")
    print(f"{'='*65}\n")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # ------------------------------------------------------------------
    # Fetch settled markets
    # ------------------------------------------------------------------
    print("Fetching settled Kalshi BTC markets …")
    try:
        ks = KalshiSource()
        raw = ks.get_settled_markets(start_ms, end_ms)
    except Exception as exc:
        print(f"ERROR: Kalshi fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    markets = [parse_kalshi_market(m) for m in raw]
    resolved = [m for m in markets if m.resolution in ("yes", "no")]
    print(f"  {len(markets)} settled markets, {len(resolved)} with resolution")

    if not resolved:
        print("No resolved markets in range. Nothing to backtest.")
        return 0

    # ------------------------------------------------------------------
    # Fetch historical BTC hourly data for the full range + 30d warmup
    # ------------------------------------------------------------------
    print("Fetching historical BTC 1h OHLCV …")
    try:
        cb = CoinbaseSource()
        warmup_ms = int((start_dt - timedelta(days=31)).timestamp() * 1000)
        ohlcv_1h = cb.fetch_ohlcv("BTC-USD", "1h", warmup_ms, end_ms)
    except Exception as exc:
        print(f"ERROR: Coinbase fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    print(f"  OHLCV rows: {len(ohlcv_1h)}")

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    print(f"\nRunning backtest on {len(resolved)} markets with ${position_size:.0f} position size …")
    try:
        per_trade, summary = run_backtest(resolved, ohlcv_1h, position_size)
    except Exception as exc:
        print(f"ERROR: Backtest failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    total_trades = sum(r["n_trades"] for r in summary)
    print(f"  Total trades taken across all strategies: {total_trades}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    _print_table(summary)

    # Verdict
    best = max(summary, key=lambda r: r["total_pnl"])
    print(f"Verdict: {best['strategy']} wins with {best['total_pnl']:+.1%} total return, "
          f"Sharpe {best['sharpe']:.2f}, max DD {best['max_dd']:+.1%}\n")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    csv_path = _OUTPUT_DIR / f"backtest_{ts_str}.csv"

    fieldnames = [
        "strategy", "ticker", "as_of", "expiration", "strike", "side",
        "market_prob", "fair_prob", "edge", "position_size",
        "resolved_yes", "pnl_per_dollar", "pnl_dollars", "cumulative_pnl_dollars",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_trade)

    print(f"Per-trade detail saved → {csv_path}")
    print(f"{'='*65}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
