from __future__ import annotations
import argparse
import json
import sys
from datetime import date
from pathlib import Path

from pipeline.run_scan import run_scan, ScanConfig
from pipeline.universe_builder import UniverseFilters, build_universe_cached
from ui.exporter import write_scan


OUTPUT_DIR = Path('output/scans')


def _print_summary(result) -> None:
    n_total = len(result.candidates)
    by_label = {}
    for c in result.candidates:
        by_label[c.label] = by_label.get(c.label, 0) + 1
    print(f"Scanned {n_total} candidates across universe of {len(set(c.contract.ticker for c in result.candidates))} tickers.")
    print(f"  TRADE: {by_label.get('TRADE', 0)}  WATCHLIST: {by_label.get('WATCHLIST', 0)}  NO_TRADE: {by_label.get('NO_TRADE', 0)}")
    print(f"  Skipped: {len(result.skipped)} ({', '.join(f'{t}:{r}' for t,r in list(result.skipped.items())[:3])}{'...' if len(result.skipped) > 3 else ''})")
    if result.candidates:
        top = result.candidates[0]
        print(f"  Top: {top.contract.ticker} {top.strategy.name} ${top.contract.strike} {top.contract.expiration} score={top.composite_score}")


def cmd_scan(args) -> int:
    filters = UniverseFilters(
        min_avg_volume=args.min_volume,
        top_n=args.top,
        require_options_chain=True,
    )
    config = ScanConfig(universe_filters=filters)
    config.use_technical_filter = not args.no_technical_filter
    if args.as_of:
        from datetime import datetime
        try:
            config.as_of = datetime.strptime(args.as_of, '%Y-%m-%d').date()
        except ValueError:
            print(f"Invalid --as-of date: {args.as_of!r} (expected YYYY-MM-DD)", file=sys.stderr)
            return 2
        if config.as_of < date.today():
            print(f"Warning: --as-of {args.as_of} is in the past. Option chain is current; "
                  f"strikes/expirations/IV may not match the historical reality.", file=sys.stderr)
    if args.fast:
        config.n_monte_carlo_paths = 1000  # tradeoff: faster, less precise GARCH-MC PoP

    sources_override = None
    if args.tickers:
        from data.sources.factory import default_sources
        sources_override = default_sources()
        import pipeline.run_scan as rs
        rs.build_universe_cached = lambda *a, **kw: [t.strip().upper() for t in args.tickers.split(',')]

    result = run_scan(config, sources_override=sources_override)
    csv_path, json_path = write_scan(result, OUTPUT_DIR)
    _print_summary(result)
    print(f"Wrote: {csv_path}")
    print(f"       {json_path}")

    if args.then_dashboard:
        return cmd_dashboard(args)
    return 0


def cmd_dashboard(_args) -> int:
    import subprocess
    return subprocess.call([sys.executable, '-m', 'streamlit', 'run', 'ui/dashboard.py'])


def cmd_watchlist(args) -> int:
    from ui.watchlist import render_watchlist
    scan_path = Path(args.scan) if args.scan else Path('output/scans/latest.json')
    if not scan_path.exists():
        print(f"Scan file not found: {scan_path}", file=sys.stderr)
        return 2
    print(render_watchlist(scan_path))
    return 0


def cmd_universe(args) -> int:
    if args.action == 'rebuild':
        from datetime import date as _date
        cache_file = Path('cache/universe') / f"universe_{_date.today().isoformat()}.json"
        if cache_file.exists():
            cache_file.unlink()
        from data.sources.factory import default_sources
        sources = default_sources()
        out = build_universe_cached(UniverseFilters(), sources)
        print(f"Universe rebuilt: {len(out)} tickers")
        print(', '.join(out))
        return 0
    print(f"Unknown universe action: {args.action}", file=sys.stderr)
    return 1


def cmd_aplus_watchlist(args) -> int:
    from scripts.aplus_watchlist import render_watchlist
    scan_path = Path(args.scan) if args.scan else Path('output/scans/latest.json')
    if not scan_path.exists():
        print(f"Scan file not found: {scan_path}", file=sys.stderr)
        return 2
    render_watchlist(scan_path, Path(args.out), account_size=args.account_size)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog='options-scanner')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('scan', help='Run an EOD scan')
    s.add_argument('--top', type=int, default=50)
    s.add_argument('--min-volume', type=int, default=1_000_000)
    s.add_argument('--tickers', type=str, default=None, help='Comma-separated override list')
    s.add_argument('--then-dashboard', action='store_true')
    s.add_argument('--fast', action='store_true', help='Use 1000 GARCH-MC paths instead of 10000 for faster scans')
    s.add_argument('--no-technical-filter', action='store_true',
                   help='Bypass Saty technical filter; use legacy RV/IV regime-only gating')
    s.add_argument('--as-of', type=str, default=None, metavar='YYYY-MM-DD',
                   help='Pin scan to a historical EOD snapshot. Spot becomes the close of that date, '
                        'history is sliced. Option chain remains live (limitation).')
    s.set_defaults(func=cmd_scan)

    d = sub.add_parser('dashboard', help='Launch Streamlit dashboard')
    d.set_defaults(func=cmd_dashboard)

    w = sub.add_parser('watchlist', help='Print a focused long-only report from a scan')
    w.add_argument('--scan', type=str, default=None,
                   help='Path to scan JSON (defaults to output/scans/latest.json)')
    w.set_defaults(func=cmd_watchlist)

    u = sub.add_parser('universe', help='Manage universe cache')
    u.add_argument('action', choices=['rebuild'])
    u.set_defaults(func=cmd_universe)

    a = sub.add_parser('aplus_watchlist', help='Generate A+ confluence-graded watchlist')
    a.add_argument('--scan', type=str, default=None,
                   help='Path to scan JSON (defaults to output/scans/latest.json)')
    a.add_argument('--out', type=str, default='output/aplus',
                   help='Output directory for latest.json')
    a.add_argument('--account-size', type=float, default=1000.0,
                   help='Account size for sizing recommendations (default $1000)')
    a.set_defaults(func=cmd_aplus_watchlist)

    args = p.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
