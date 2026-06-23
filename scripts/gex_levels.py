"""GEX Levels CLI — Dealer Gamma Exposure scalping tool for index options.

Fetches live option chains, computes per-strike dealer GEX, identifies the
gamma flip level, and prints supply/demand zone walls for intraday scalping.

Usage:
    python -m scripts.gex_levels SPY
    python -m scripts.gex_levels SPY QQQ IWM DIA
    python -m scripts.gex_levels SPY --watch 60       # refresh every 60s
    python -m scripts.gex_levels SPY --dte-max 2      # 0-2 DTE (pin risk)
    python -m scripts.gex_levels SPY --dte-max 30     # near-term regime view
    python -m scripts.gex_levels SPY --zones 10       # show top 10 zones
    python -m scripts.gex_levels SPY --max-dist 0.03  # only zones within 3%
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.factory import default_sources
from domain.gex.compute import GexContext, build_context


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_gex_dollars(value: float) -> str:
    """Format a GEX dollar value in human-readable units with sign.

    Examples:
        1.5e9  → "+$1.50B"
        -2.3e6 → "-$2.30M"
        5e3    → "+$5.00K"
        -500   → "-$500.00"
    """
    sign = '+' if value >= 0.0 else '-'
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.2f}K"
    return f"{sign}${abs_val:.2f}"


def _regime_label(total_gex: float) -> str:
    if total_gex >= 0.0:
        return "POSITIVE — vol-suppressed regime"
    return "NEGATIVE — vol-amplified regime, trending"


def _et_now() -> str:
    """Return current time in ET format (simple UTC-4 offset for EDT)."""
    try:
        from datetime import timezone as _tz, timedelta
        # Use UTC-4 for EDT (EST would be UTC-5; EDT is in effect Jun–Nov)
        et = datetime.now(_tz.utc).astimezone(_tz(timedelta(hours=-4)))
        return et.strftime('%H:%M:%S ET')
    except Exception:
        return datetime.now().strftime('%H:%M:%S')


# ---------------------------------------------------------------------------
# Render a single GexContext to stdout
# ---------------------------------------------------------------------------

def render_context(ctx: GexContext, n_zones: int = 5) -> None:
    """Print a formatted GEX report for one ticker."""
    from data.sources.market_session import current_session, is_extended_hours
    session = current_session()
    sep = '=' * 70
    print(sep)
    print(f"=== {ctx.ticker} @ {_et_now()} ===")
    print(sep)
    session_line = f"Session: {session.upper()}"
    if is_extended_hours(session):
        session_line += "  (Spot reflects extended-hours quote)"
    print(session_line)
    print(f"Spot:                ${ctx.spot:,.2f}")
    print(f"Total GEX:           {format_gex_dollars(ctx.total_gex_dollars)}"
          f"   ({_regime_label(ctx.total_gex_dollars)})")

    if ctx.gamma_flip_level is not None:
        gfl = ctx.gamma_flip_level
        dist_pct = (gfl - ctx.spot) / ctx.spot * 100.0
        direction = 'above' if dist_pct >= 0 else 'below'
        print(f"Gamma Flip Level:    ${gfl:,.2f}   "
              f"({dist_pct:+.1f}% {direction} spot)")
    else:
        print("Gamma Flip Level:    n/a (no crossing found in search range)")

    print(f"Contracts analyzed:  {ctx.chain_size:,}")
    print()

    # ---- SUPPLY (above spot) ----
    print("SUPPLY (resistance — above spot):")
    if not ctx.supply_zones:
        print("  No supply zones within range")
    else:
        print(f"  {'Strike':<10} {'GEX':<12} {'Distance':<14} {'Wall'}")
        print(f"  {'────────':<10} {'────────':<12} {'─────────':<14} {'────'}")
        top_supply = max(ctx.supply_zones, key=lambda z: abs(z.gex_dollars))
        for z in ctx.supply_zones[:n_zones]:
            tag = '← top' if z.strike == top_supply.strike else ''
            print(f"  ${z.strike:<9,.0f} {format_gex_dollars(z.gex_dollars):<12} "
                  f"{z.distance_pct:+.2%}{'':8} {tag}")

    print()

    # ---- DEMAND (below spot) ----
    print("DEMAND (support — below spot):")
    if not ctx.demand_zones:
        print("  No demand zones within range")
    else:
        print(f"  {'Strike':<10} {'GEX':<12} {'Distance':<14} {'Wall'}")
        print(f"  {'────────':<10} {'────────':<12} {'─────────':<14} {'────'}")
        top_demand = max(ctx.demand_zones, key=lambda z: abs(z.gex_dollars))
        for z in ctx.demand_zones[:n_zones]:
            tag = '← top' if z.strike == top_demand.strike else ''
            print(f"  ${z.strike:<9,.0f} {format_gex_dollars(z.gex_dollars):<12} "
                  f"{z.distance_pct:+.2%}{'':8} {tag}")

    print()


# ---------------------------------------------------------------------------
# Fetch + build for one ticker
# ---------------------------------------------------------------------------

def fetch_and_build(
    ticker: str,
    sources,
    max_dte: int | None,
    n_zones: int,
    zone_max_distance_pct: float,
    risk_free_rate: float,
) -> GexContext | None:
    """Fetch chain + spot and build GexContext.  Returns None on failure."""
    try:
        chain = fetch_with_fallback(sources, 'fetch_option_chain', ticker)
    except DataFetchError as exc:
        print(f"ERROR [{ticker}]: Could not fetch option chain — {exc}",
              file=sys.stderr)
        return None
    except Exception as exc:
        print(f"ERROR [{ticker}]: Unexpected error fetching chain — {exc}",
              file=sys.stderr)
        return None

    try:
        spot = fetch_with_fallback(sources, 'fetch_spot', ticker)
    except DataFetchError:
        # Fall back to spot_price on first contract
        if chain:
            spot = chain[0].spot_price
        else:
            print(f"ERROR [{ticker}]: Could not determine spot price", file=sys.stderr)
            return None

    if not chain:
        print(f"ERROR [{ticker}]: Empty option chain returned", file=sys.stderr)
        return None

    try:
        return build_context(
            ticker=ticker,
            chain=chain,
            spot=spot,
            risk_free_rate=risk_free_rate,
            max_dte=max_dte,
            n_zones=n_zones,
            zone_max_distance_pct=zone_max_distance_pct,
        )
    except Exception as exc:
        print(f"ERROR [{ticker}]: Failed to build GEX context — {exc}",
              file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# One scan pass
# ---------------------------------------------------------------------------

def run_pass(
    tickers: list[str],
    sources,
    max_dte: int | None,
    n_zones: int,
    zone_max_distance_pct: float,
    risk_free_rate: float,
) -> int:
    """Run a single scan pass for all tickers.  Returns exit code (0=success)."""
    any_success = False
    for ticker in tickers:
        ctx = fetch_and_build(
            ticker=ticker,
            sources=sources,
            max_dte=max_dte,
            n_zones=n_zones,
            zone_max_distance_pct=zone_max_distance_pct,
            risk_free_rate=risk_free_rate,
        )
        if ctx is not None:
            render_context(ctx, n_zones=n_zones)
            any_success = True

    return 0 if any_success else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="GEX Levels — dealer gamma exposure scalping tool"
    )
    parser.add_argument('tickers', nargs='+', help='Ticker symbols (e.g. SPY QQQ)')
    parser.add_argument('--watch', type=int, default=None, metavar='SEC',
                        help='Refresh every SEC seconds (Ctrl-C to stop)')
    parser.add_argument('--dte-max', type=int, default=None,
                        help='Maximum DTE to include (e.g. 2 for pin risk, 30 for regime)')
    parser.add_argument('--zones', type=int, default=5,
                        help='Number of supply/demand zones to display (default 5)')
    parser.add_argument('--max-dist', type=float, default=0.05,
                        help='Max distance from spot to include zones, as a fraction '
                             '(e.g. 0.05 = 5%%). Default 0.05.')
    parser.add_argument('--risk-free', type=float, default=0.053,
                        help='Risk-free rate for Black-Scholes gamma (default 0.053)')
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.tickers]
    sources = default_sources()

    if args.watch is None:
        # Single pass
        return run_pass(
            tickers=tickers,
            sources=sources,
            max_dte=args.dte_max,
            n_zones=args.zones,
            zone_max_distance_pct=args.max_dist,
            risk_free_rate=args.risk_free,
        )
    else:
        # Watch mode — loop until Ctrl-C
        try:
            while True:
                print('\033[2J\033[H', end='')  # ANSI clear screen
                run_pass(
                    tickers=tickers,
                    sources=sources,
                    max_dte=args.dte_max,
                    n_zones=args.zones,
                    zone_max_distance_pct=args.max_dist,
                    risk_free_rate=args.risk_free,
                )
                print(f"[watch] Next refresh in {args.watch}s — Ctrl-C to stop")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[watch] Stopped.")
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
