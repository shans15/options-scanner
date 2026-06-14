from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_watchlist(scan_path: Path) -> str:
    """Load the scan JSON, build the report, return the formatted text."""
    data = _load_scan(scan_path)
    candidates: list[dict] = data.get('candidates', [])
    skipped: dict = data.get('skipped', {})
    timestamp_raw: str = data.get('timestamp', '')

    # Parse timestamp
    try:
        ts = datetime.fromisoformat(timestamp_raw)
        ts_str = ts.strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        ts_str = timestamp_raw

    # Identify all tickers that have direction-aligned long candidates
    long_candidates = [c for c in candidates if c.get('strategy') in ('long_call', 'long_put')]

    # Build per-ticker setup info: pick the dominant setup per ticker from all candidates
    ticker_setup: dict[str, dict] = {}
    for c in candidates:
        t = c['contract']['ticker']
        if t not in ticker_setup:
            ticker_setup[t] = c  # first (highest-scored) candidate sets the context

    all_tickers = sorted(ticker_setup.keys())
    total_scanned = len(all_tickers)

    # Tickers that have at least one direction-aligned long candidate
    tickers_with_setups: list[str] = []
    for ticker in all_tickers:
        setup_dir = ticker_setup[ticker].get('setup_direction') or ''
        if _best_long_candidate(long_candidates, ticker, setup_dir) is not None:
            tickers_with_setups.append(ticker)

    tickers_skipped: list[str] = []
    for ticker in all_tickers:
        if ticker not in tickers_with_setups:
            tickers_skipped.append(ticker)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    lines: list[str] = []
    lines.append(f"=== Watchlist — {ts_str} ===")
    lines.append('')
    lines.append(f"Universe: {len(tickers_with_setups)} tickers with setups out of {total_scanned} scanned")
    lines.append('')

    # -----------------------------------------------------------------------
    # Setup confluence (only groups with 2+ tickers, or counter-trend setups)
    # -----------------------------------------------------------------------
    confluence = _setup_confluence(long_candidates)
    confluence_lines = _format_confluence(confluence, ticker_setup)
    if confluence_lines:
        lines.append('Setup confluence:')
        lines.extend(confluence_lines)
        lines.append('')

    # -----------------------------------------------------------------------
    # Per-ticker blocks
    # -----------------------------------------------------------------------
    sep = '─' * 77
    lines.append(sep)

    for ticker in tickers_with_setups:
        ctx = ticker_setup[ticker]
        setup_dir = ctx.get('setup_direction') or ''
        best = _best_long_candidate(long_candidates, ticker, setup_dir)
        if best is None:
            continue

        block = _format_ticker_block(best, long_candidates)
        lines.append(block)

    lines.append(sep)

    # -----------------------------------------------------------------------
    # Skipped tickers
    # -----------------------------------------------------------------------
    if tickers_skipped:
        lines.append('')
        lines.append('Skipped (no direction-aligned long candidate):')
        for ticker in tickers_skipped:
            ctx = ticker_setup[ticker]
            setup_name = ctx.get('setup_name') or 'unknown'
            setup_dir = ctx.get('setup_direction') or 'unknown'
            lines.append(f'  {ticker} ({setup_name} {setup_dir} — no liquid long_call/long_put within delta range)')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scan(scan_path: Path) -> dict:
    return json.loads(scan_path.read_text())


def _setup_confluence(candidates: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Group tickers by (setup_name, direction). Return mapping."""
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in candidates:
        name = c.get('setup_name') or ''
        direction = c.get('setup_direction') or ''
        ticker = c['contract']['ticker']
        if name and direction:
            seen[(name, direction)].add(ticker)
    return {k: sorted(v) for k, v in seen.items()}


def _best_long_candidate(candidates: list[dict], ticker: str, target_direction: str) -> Optional[dict]:
    """Return the highest-composite-score long_call/long_put candidate aligned with
    target_direction for that ticker. Returns None if no aligned long candidate exists.

    Direction alignment rules:
      bullish  → long_call
      bearish  → long_put
    """
    strategy_for_direction = {
        'bullish': 'long_call',
        'bearish': 'long_put',
    }
    target_strategy = strategy_for_direction.get(target_direction)

    aligned = [
        c for c in candidates
        if c['contract']['ticker'] == ticker
        and c.get('strategy') in ('long_call', 'long_put')
        and (target_strategy is None or c.get('strategy') == target_strategy)
    ]

    if not aligned:
        return None

    return max(aligned, key=lambda c: c.get('composite_score', 0.0))


def _liquidity_grade(contract: dict) -> tuple[str, str]:
    """Return (emoji_flag, label).

    Thresholds:
      OI >= 500 AND volume >= 100   → ✓ liquid
      OI >= 500 but volume < 100    → ⚠ thin volume
      100 <= OI < 500               → ⚠ thin OI
      OI < 100                      → ✗ illiquid
    """
    oi = contract.get('open_interest', 0) or 0
    vol = contract.get('volume', 0) or 0

    if oi >= 500 and vol >= 100:
        return ('✓', 'liquid')
    if oi >= 500 and vol < 100:
        return ('⚠', 'thin volume')
    if oi >= 100:
        return ('⚠', 'thin OI')
    return ('✗', 'illiquid')


def _find_liquid_fallback(
    candidates: list[dict],
    ticker: str,
    target_direction: str,
    exclude_strike: float,
) -> Optional[dict]:
    """Search for an alternate long candidate with the same ticker + direction
    that has better liquidity (OI >= 500 ideally; fall back to highest OI).
    Returns the highest-OI direction-aligned long candidate other than exclude_strike.
    """
    strategy_for_direction = {
        'bullish': 'long_call',
        'bearish': 'long_put',
    }
    target_strategy = strategy_for_direction.get(target_direction)

    alternates = [
        c for c in candidates
        if c['contract']['ticker'] == ticker
        and c.get('strategy') in ('long_call', 'long_put')
        and (target_strategy is None or c.get('strategy') == target_strategy)
        and c['contract'].get('strike') != exclude_strike
    ]

    if not alternates:
        return None

    return max(alternates, key=lambda c: c['contract'].get('open_interest', 0) or 0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_confluence(
    confluence: dict[tuple[str, str], list[str]],
    ticker_setup: dict[str, dict],
) -> list[str]:
    """Return formatted confluence lines. Only show groups with 2+ tickers
    or single-ticker groups that are counter-trend."""
    lines = []
    for (setup_name, direction), tickers in sorted(confluence.items(), key=lambda x: -len(x[1])):
        is_counter_trend = any(
            ticker_setup.get(t, {}).get('weekly_trend', direction[:3]) != direction[:3]
            for t in tickers
        )
        if len(tickers) < 2 and not is_counter_trend:
            continue

        ticker_str = ', '.join(tickers)
        note = ''
        if len(tickers) >= 2:
            note = '  (sector agreement)'
        line = f'  {len(tickers)} ticker{"s" if len(tickers) != 1 else ""} {direction} {setup_name}: {ticker_str}{note}'

        # Check for counter-trend warnings on individual tickers
        for t in tickers:
            ctx = ticker_setup.get(t, {})
            wt = ctx.get('weekly_trend') or ''
            if wt and wt != direction[:3] and not (direction == 'bullish' and wt == 'up') and not (direction == 'bearish' and wt == 'down'):
                pass  # detailed warnings shown per-ticker block

        lines.append(line)
    return lines


def _format_ticker_block(best: dict, all_candidates: list[dict]) -> str:
    """Format the full block for a single ticker."""
    contract = best['contract']
    ticker = contract['ticker']
    spot = contract.get('spot_price', 0.0)
    strategy = best.get('strategy', '')
    setup_name = best.get('setup_name') or ''
    setup_dir = best.get('setup_direction') or ''
    setup_strength = best.get('setup_strength') or 0.0
    weekly_trend = best.get('weekly_trend') or ''
    streak = best.get('consecutive_close_streak')
    p1w = best.get('pct_change_1w')
    p4w = best.get('pct_change_4w')
    agree = best.get('weekly_ribbon_agreement')
    label = best.get('label', '')
    vix_now = best.get('vix_now')
    vix_regime = best.get('vix_regime')
    vix_pct_vs_7d = best.get('vix_pct_vs_7d')

    # Direction arrow
    arrow = '▲' if setup_dir == 'bullish' else '▼'

    # Counter-trend warning
    wt_str = weekly_trend or '?'
    is_counter = (setup_dir == 'bullish' and weekly_trend == 'down') or \
                 (setup_dir == 'bearish' and weekly_trend == 'up')
    counter_warn = '  ⚠ weekly_trend=down' if (setup_dir == 'bullish' and weekly_trend == 'down') else \
                   '  ⚠ weekly_trend=up' if (setup_dir == 'bearish' and weekly_trend == 'up') else ''

    # Title line
    title = f"{ticker}  spot ${spot:.2f}  {arrow} {setup_dir} {setup_name} (strength {setup_strength:.2f}){counter_warn}"

    # Context line
    streak_str = f'{streak:+d}d' if streak is not None else '?d'
    p1w_str = f'{p1w:+.1%}' if p1w is not None else '?'
    p4w_str = f'{p4w:+.1%}' if p4w is not None else '?'
    agree_str = str(agree) if agree is not None else '?'
    ctx_line = f"  ctx: wt={wt_str}  streak={streak_str}  1w={p1w_str}  4w={p4w_str}  agree={agree_str}"

    # Contract details
    strike = contract.get('strike', 0.0)
    expiration = contract.get('expiration', '')
    dte = contract.get('dte', 0)
    mid = contract.get('mid', 0.0)
    delta = contract.get('delta', 0.0)
    pop = best.get('pop_blended', 0.0)
    score = best.get('composite_score', 0.0)
    vol = contract.get('volume', 0) or 0
    oi = contract.get('open_interest', 0) or 0

    contract_type = 'LongCall' if strategy == 'long_call' else 'LongPut'
    best_line = (f"  best {contract_type}:  ${strike} exp {expiration} ({dte}d)  "
                 f"mid ${mid:.2f}  delta {delta:.2f}  PoP {pop:.0%}  score {score:.1f}  [{label}]")

    # Liquidity line
    flag, liq_label = _liquidity_grade(contract)
    reason_against = best.get('reason_against', '') or ''
    liq_detail = ''
    if flag == '✗' and reason_against:
        # Surface which filters failed from reason_against
        liq_detail = f' ({reason_against})'
    elif flag == '⚠' and reason_against:
        liq_detail = f' ({reason_against})'

    liq_line = f"  liquidity:      vol {vol}  OI {oi}   {flag} {liq_label}{liq_detail}"

    lines = [title, ctx_line, best_line, liq_line]

    # VIX-regime line — only render if vix_now is present
    if vix_now is not None and vix_regime is not None:
        regime_upper = vix_regime.upper()
        regime_symbol = '⚠ ' if vix_regime in ('expansion', 'contraction') else ''
        pct_str = f', {vix_pct_vs_7d*100:+.0f}%' if vix_pct_vs_7d is not None else ''
        if vix_regime == 'expansion':
            regime_action = 'equity vol expanding → directional setups have better follow-through'
        elif vix_regime == 'contraction':
            regime_action = 'equity vol compressing → directional setups bleed theta; favor shorter DTE'
        else:
            regime_action = 'no strong vol regime signal'
        vix_7d_avg_val = vix_now / (1 + vix_pct_vs_7d) if vix_pct_vs_7d is not None and vix_pct_vs_7d != -1 else None
        vix_7d_str = f', 7d avg {vix_7d_avg_val:.1f}' if vix_7d_avg_val is not None else ''
        vix_line = f"  {regime_symbol}VIX_REGIME: {regime_upper} (VIX {vix_now:.1f}{vix_7d_str}{pct_str})"
        vix_detail = f"    → {regime_action}"
        lines.append(vix_line)
        lines.append(vix_detail)

    # Fallback — only show if primary is not already liquid (OI < 500)
    if oi < 500:
        fallback = _find_liquid_fallback(all_candidates, ticker, setup_dir, exclude_strike=float(strike))
        if fallback is not None:
            fb_contract = fallback['contract']
            fb_strike = fb_contract.get('strike', 0.0)
            fb_exp = fb_contract.get('expiration', '')
            fb_dte = fb_contract.get('dte', 0)
            fb_mid = fb_contract.get('mid', 0.0)
            fb_delta = fb_contract.get('delta', 0.0)
            fb_pop = fallback.get('pop_blended', 0.0)
            fb_score = fallback.get('composite_score', 0.0)
            fb_vol = fb_contract.get('volume', 0) or 0
            fb_oi = fb_contract.get('open_interest', 0) or 0
            fb_flag, fb_liq = _liquidity_grade(fb_contract)

            fallback_line = (f"  ↳ fallback:     ${fb_strike} exp {fb_exp} ({fb_dte}d)  "
                             f"mid ${fb_mid:.2f}  delta {fb_delta:.2f}  PoP {fb_pop:.0%}  score {fb_score:.1f}  "
                             f"vol {fb_vol}  OI {fb_oi}   {fb_flag} {fb_liq}")
            lines.append(fallback_line)

    lines.append('')  # blank line between ticker blocks
    return '\n'.join(lines)
