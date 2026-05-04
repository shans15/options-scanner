from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
from scanner.clock import get_market_status, is_market_open
import yfinance as yf

LATEST_JSON = Path(__file__).parent / 'scans' / 'latest.json'
REFRESH_INTERVAL_SECONDS = 60

DECISION_COLORS = {
    'TRADE': '#00c853',
    'WATCHLIST': '#ffab00',
    'NO TRADE': '#d50000',
}

DISPLAY_COLUMNS = [
    'ticker', 'strategy', 'expiration', 'strike', 'bid', 'ask', 'mid',
    'delta', 'IV', 'IV_rank', 'PoP_blended', 'expected_value',
    'breakeven', 'composite_score', 'decision',
]


def _load_latest() -> tuple:
    """Load latest.json. Returns (results list, scan_time string)."""
    if not LATEST_JSON.exists():
        return [], 'No scan yet'
    try:
        data = json.loads(LATEST_JSON.read_text())
        return data.get('results', []), data.get('scan_time', 'Unknown')
    except Exception:
        return [], 'Error loading results'


def _get_market_header() -> dict:
    """Fetch VIX and SPY % change for header bar."""
    try:
        vix = float(yf.download('^VIX', period='2d', progress=False)['Close'].iloc[-1])
        spy = yf.download('SPY', period='5d', progress=False)['Close']
        spy_chg = float((spy.iloc[-1] / spy.iloc[-2] - 1) * 100)
        return {'vix': round(vix, 2), 'spy_chg': round(spy_chg, 2)}
    except Exception:
        return {'vix': None, 'spy_chg': None}


def _color_decision(val: str) -> str:
    color = DECISION_COLORS.get(val, '#ffffff')
    return f'background-color: {color}; color: white; font-weight: bold'


def _render_table(df: pd.DataFrame, container: object) -> None:
    """Render filtered dataframe with expandable detail rows."""
    if df.empty:
        container.write('No results for this filter.')
        return

    display_df = df[[c for c in DISPLAY_COLUMNS if c in df.columns]].copy()

    # Format percentages
    for col in ['PoP_blended', 'IV']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f'{x:.1%}' if x is not None and x == x else ''
            )

    styled = display_df.style.map(_color_decision, subset=['decision'])
    container.dataframe(styled, use_container_width=True, height=400)

    # Detail expanders
    container.markdown('**Click a ticker for full detail:**')
    for _, row in df.iterrows():
        label = (
            f"{row.get('ticker')} | {row.get('strategy')} | "
            f"Strike {row.get('strike')} | {row.get('expiration')} | "
            f"{row.get('decision')}"
        )
        with container.expander(label):
            col1, col2 = st.columns(2)

            col1.markdown('**Probability Models**')
            for k in ['PoP_delta', 'PoP_BS', 'PoP_historical', 'PoP_GARCH_MC', 'PoP_blended']:
                v = row.get(k)
                col1.write(f"{k}: {v:.1%}" if v is not None else f"{k}: N/A")

            col1.markdown('**Stress Scenarios**')
            for k in ['stress_1SD', 'stress_2SD', 'stress_expiry']:
                v = row.get(k)
                col1.write(f"{k}: ${v:.2f}" if v is not None else f"{k}: N/A")

            col2.markdown('**Risk & Margin**')
            col2.write(f"Margin estimate: ${row.get('margin_estimate', 'N/A')}")
            col2.write(f"Breakeven: ${row.get('breakeven', 'N/A')}")
            col2.write(f"Hard filter passed: {row.get('hard_filter_passed')}")
            if not row.get('hard_filter_passed'):
                col2.warning(f"Filter reason: {row.get('hard_filter_reason')}")

            col2.markdown('**Trade Narrative**')
            col2.success(f"FOR: {row.get('reason_for', '')}")
            col2.error(f"AGAINST: {row.get('reason_against', '')}")
            col2.info(f"STOP: {row.get('stop_trigger', '')}")


def render_dashboard() -> None:
    """Main Streamlit dashboard render function."""
    st.set_page_config(
        page_title='Options Scanner',
        page_icon='📊',
        layout='wide',
    )

    # Auto-refresh every 60 seconds
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0.0
    if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL_SECONDS:
        st.session_state.last_refresh = time.time()
        st.rerun()

    results, scan_time = _load_latest()
    market_status = get_market_status()
    header = _get_market_header()

    # Header bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Market', market_status)
    scan_display = scan_time[:16] if len(scan_time) > 16 else scan_time
    col2.metric('Last Scan', scan_display)
    if header['vix']:
        col3.metric('VIX', header['vix'])
        direction = '▲' if header['spy_chg'] >= 0 else '▼'
        col4.metric('SPY', f"{direction} {abs(header['spy_chg']):.2f}%")

    st.divider()

    if not results:
        if is_market_open():
            st.info(
                'Scan in progress or no qualifying contracts found. '
                'Results update at 09:45, 11:00, 13:00, 15:00 ET.'
            )
        else:
            st.warning('Market is closed. Showing last available scan results.')
        st.divider()
        st.caption(
            'This is quantitative research tooling only — not financial advice. '
            'Naked options carry theoretically unlimited risk.'
        )
        return

    df = pd.DataFrame(results)

    # Filter tabs
    tab_all, tab_puts, tab_calls, tab_trade, tab_watch = st.tabs(
        ['All', 'Naked Puts', 'Naked Calls', 'TRADE Only', 'WATCHLIST']
    )

    with tab_all:
        _render_table(df, tab_all)
    with tab_puts:
        _render_table(df[df['strategy'] == 'naked_put'], tab_puts)
    with tab_calls:
        _render_table(df[df['strategy'] == 'naked_call'], tab_calls)
    with tab_trade:
        _render_table(df[df['decision'] == 'TRADE'], tab_trade)
    with tab_watch:
        _render_table(df[df['decision'] == 'WATCHLIST'], tab_watch)

    st.divider()
    st.caption(
        'This is quantitative research tooling only — not financial advice. '
        'Naked options carry theoretically unlimited risk. '
        'Always verify independently before acting.'
    )


if __name__ == '__main__':
    render_dashboard()
