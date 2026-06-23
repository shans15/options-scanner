"""Stock vs Option trade comparator CLI.

Given a ticker + target + stop, projects both a direct share purchase and a
covered option swing forward to the user's price target and stop via
Black-Scholes, then prints a side-by-side comparison and a clear verdict.

Usage:
    python -m scripts.trade_compare KLAC --target 285 --stop 258
    python -m scripts.trade_compare LRCX --target 425 --stop 395 --hold-days 28
    python -m scripts.trade_compare T --direction bearish --target 20 --stop 23
    python -m scripts.trade_compare AMAT  # uses default +/- 5% target/stop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from data.fallback import DataFetchError, fetch_with_fallback
from data.sources.factory import default_sources
from domain.stock_scanner.compare import (
    OptionRoute,
    StockRoute,
    TradeComparison,
    compare_routes,
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP_WIDE = '=' * 78
_SEP_THIN = '─' * 78   # ──── line


def _pct(val: float) -> str:
    """Format a fraction as a ± percentage string."""
    sign = '+' if val >= 0 else ''
    return f'{sign}{val * 100:.1f}%'


def _dollar(val: float, sign: bool = True) -> str:
    """Format a dollar amount with optional sign."""
    if sign:
        prefix = '+' if val >= 0 else '-'
        return f'{prefix}${abs(val):,.2f}'
    return f'${val:,.2f}'


def render_comparison(cmp: TradeComparison) -> None:
    """Print a formatted comparison report to stdout."""
    s = cmp.stock_route
    direction_label = cmp.direction.upper()

    print(_SEP_WIDE)
    print(f'TRADE COMPARISON: {cmp.ticker}  (Direction: {direction_label})')
    print(_SEP_WIDE)
    print(f'Spot:           ${cmp.spot:,.2f}')
    target_pct = (cmp.target - cmp.spot) / cmp.spot
    stop_pct   = (cmp.stop - cmp.spot) / cmp.spot
    print(f'Target:         ${cmp.target:,.2f}  ({_pct(target_pct)})')
    print(f'Stop:           ${cmp.stop:,.2f}  ({_pct(stop_pct)})')
    print(f'Hold:           {cmp.hold_days} days')
    print(f'Account:        ${cmp.account_balance:,.0f}')

    # ---- Stock route -------------------------------------------------------
    print()
    print(_SEP_THIN)
    print('STOCK ROUTE -- buy shares')
    print(_SEP_THIN)
    acct_pct = s.cost / cmp.account_balance * 100
    print(f'  Shares:           {s.shares} @ ${s.entry_price:,.2f}')
    print(f'  Cost:             ${s.cost:,.2f} ({acct_pct:.1f}% of account)')
    print(f'  At target (${cmp.target:,.0f}): {_dollar(s.target_pnl)} ({_pct(s.target_pct)})')
    print(f'  At stop (${cmp.stop:,.0f}):   {_dollar(s.stop_pnl)} ({_pct(s.stop_pct)})')
    print(f'  R:R ratio:        {s.rr_ratio:.2f}:1')

    # ---- Option routes -----------------------------------------------------
    print()
    print(_SEP_THIN)
    if cmp.option_routes:
        print(f'OPTION ROUTES -- top {len(cmp.option_routes)} affordable contracts (ranked by R:R)')
    else:
        print('OPTION ROUTES -- none found within budget / filters')
    print(_SEP_THIN)

    for i, opt in enumerate(cmp.option_routes, start=1):
        acct_pct_opt = opt.cost / cmp.account_balance * 100
        target_val_total = opt.projected_target_value * 100
        stop_val_total = opt.projected_stop_value * 100
        target_pnl_pct = opt.target_pnl / opt.cost * 100 if opt.cost > 0 else 0
        stop_pnl_pct   = opt.stop_pnl / opt.cost * 100   if opt.cost > 0 else 0

        print()
        print(f'#{i}  {opt.ticker} ${opt.strike:,.0f} {opt.option_type.upper()} '
              f'exp {opt.expiration}  (DTE {opt.dte_at_entry})')
        print(f'    Cost:             ${opt.entry_mid:.2f} mid = '
              f'${opt.cost:,.2f} ({acct_pct_opt:.1f}% of account)')
        print(f'    Delta:            {opt.delta:.2f}  '
              f'IV: {opt.iv_at_entry*100:.0f}%  '
              f'Bid ${opt.bid:.2f} Ask ${opt.ask:.2f}')
        sign_t = '+' if opt.target_pnl >= 0 else ''
        sign_s = '+' if opt.stop_pnl >= 0 else ''
        print(f'    At target (${cmp.target:,.0f}): '
              f'est ${opt.projected_target_value:.2f}/sh = '
              f'{sign_t}${abs(opt.target_pnl):,.0f} ({sign_t}{target_pnl_pct:.0f}%)')
        print(f'    At stop (${cmp.stop:,.0f}):   '
              f'est ${opt.projected_stop_value:.2f}/sh = '
              f'{sign_s}${abs(opt.stop_pnl):,.0f} ({sign_s}{stop_pnl_pct:.0f}%)')
        print(f'    Breakeven:        ${opt.breakeven_spot:,.2f}')
        print(f'    R:R:              {opt.rr_ratio:.2f}:1')
        lev_str = f'{opt.leverage:.1f}x' if opt.leverage > 0 else 'n/a'
        print(f'    Leverage vs stock: {lev_str} return per $ deployed')
        if opt.notes:
            print(f'    Note:             {opt.notes}')

    if not cmp.option_routes:
        print()
        print('  (No contracts passed the DTE / budget / strike filters.)')

    # ---- Verdict -----------------------------------------------------------
    print()
    print(_SEP_THIN)
    print(f'VERDICT: {cmp.verdict}')
    print(_SEP_THIN)
    print(f'  Reason: {cmp.verdict_reason}')

    if cmp.option_routes:
        best = cmp.option_routes[0]
        lev_str = f'{best.leverage:.1f}x' if best.leverage > 0 else 'n/a'
        print()
        print(f'  Stock R:R is {s.rr_ratio:.2f} vs best option R:R {best.rr_ratio:.2f}.')
        print(f'  Option leverage of {lev_str} sounds attractive but requires '
              f'breaching breakeven')
        print(f'  at ${best.breakeven_spot:,.2f} — needs '
              f'{abs((best.breakeven_spot - cmp.spot) / cmp.spot)*100:.1f}% spot move '
              f'in {cmp.hold_days} days OR break out fast.')
        print()
        print('  Stock route: cleaner R:R, no theta, no IV crush, holds indefinitely.')
        print('  Best for setups with <10% target moves.')
    print(_SEP_WIDE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Stock vs Option trade comparator — project both routes to target.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('ticker', help='Equity ticker symbol (e.g. KLAC)')
    parser.add_argument('--direction', choices=['bullish', 'bearish'], default='bullish',
                        help='Trade direction (default: bullish)')
    parser.add_argument('--target', type=float, default=None,
                        help='Price target; default: spot * 1.05 (bullish) or spot * 0.95 (bearish)')
    parser.add_argument('--stop', type=float, default=None,
                        help='Stop loss level; default: spot * 0.97 (bullish) or spot * 1.03 (bearish)')
    parser.add_argument('--hold-days', type=int, default=28,
                        help='Expected hold in calendar days (default: 28)')
    parser.add_argument('--account', type=float, default=1000.0,
                        help='Available capital in dollars (default: 1000)')
    parser.add_argument('--risk-free', type=float, default=0.053,
                        help='Risk-free rate for Black-Scholes (default: 0.053)')
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()
    sources = default_sources()

    # Resolve target / stop defaults — need spot first
    if args.target is None or args.stop is None:
        try:
            spot_for_defaults = float(fetch_with_fallback(sources, 'fetch_spot', ticker))
        except DataFetchError as exc:
            print(f'ERROR: Cannot fetch spot for {ticker}: {exc}', file=sys.stderr)
            return 1

        if args.direction == 'bullish':
            target = args.target if args.target is not None else spot_for_defaults * 1.05
            stop   = args.stop   if args.stop   is not None else spot_for_defaults * 0.97
        else:
            target = args.target if args.target is not None else spot_for_defaults * 0.95
            stop   = args.stop   if args.stop   is not None else spot_for_defaults * 1.03
    else:
        target = args.target
        stop   = args.stop

    try:
        cmp = compare_routes(
            ticker=ticker,
            direction=args.direction,
            target=target,
            stop=stop,
            hold_days=args.hold_days,
            account_balance=args.account,
            risk_free_rate=args.risk_free,
            sources=sources,
        )
    except DataFetchError as exc:
        print(f'ERROR [{ticker}]: {exc}', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'ERROR [{ticker}]: Unexpected error — {exc}', file=sys.stderr)
        return 1

    render_comparison(cmp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
