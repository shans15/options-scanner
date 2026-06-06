from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st


LATEST = Path('output/scans/latest.json')


def _load_latest() -> dict | None:
    if not LATEST.exists():
        return None
    return json.loads(LATEST.read_text())


def _candidates_to_df(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        con = c['contract']
        rows.append({
            'ticker': con['ticker'],
            'strategy': c['strategy'],
            'direction': 'sell' if c['strategy'] in ('naked_put', 'naked_call') else 'buy',
            'option_type': con['option_type'],
            'strike': con['strike'],
            'expiration': con['expiration'],
            'dte': con['dte'],
            'mid': con['mid'],
            'pop_blended': c['pop_blended'],
            'ev': c['ev'],
            'score': c['composite_score'],
            'label': c['label'],
            'setup_name': c.get('setup_name', ''),
            'setup_direction': c.get('setup_direction', ''),
            'setup_strength': c.get('setup_strength'),
        })
    return pd.DataFrame(rows)


def render_dashboard() -> None:
    st.set_page_config(page_title='Options Scanner', layout='wide')
    st.title('Options Scanner — EOD Candidates')

    data = _load_latest()
    if data is None:
        st.warning('No scan results yet. Run `python -m main scan` first.')
        return

    st.caption(f"Last scan: {data['timestamp']}")
    candidates_data = data['candidates']

    setup_options = sorted({c.get('setup_name', '') for c in candidates_data if c.get('setup_name')})
    selected_setup = st.sidebar.selectbox('Filter by setup', ['(any)'] + setup_options)
    if selected_setup != '(any)':
        candidates_data = [c for c in candidates_data if c.get('setup_name') == selected_setup]

    df = _candidates_to_df(candidates_data)

    if df.empty:
        st.info('No candidates in latest scan.')
        return

    counts = df['label'].value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    c1.metric('TRADE', counts.get('TRADE', 0))
    c2.metric('WATCHLIST', counts.get('WATCHLIST', 0))
    c3.metric('NO_TRADE', counts.get('NO_TRADE', 0))

    tab_all, tab_sell, tab_buy, tab_trade, tab_watch = st.tabs(
        ['All', 'Sell', 'Buy', 'TRADE only', 'WATCHLIST only']
    )

    def _show(view: pd.DataFrame):
        st.dataframe(view.sort_values('score', ascending=False), use_container_width=True)

    with tab_all: _show(df)
    with tab_sell: _show(df[df['direction'] == 'sell'])
    with tab_buy: _show(df[df['direction'] == 'buy'])
    with tab_trade: _show(df[df['label'] == 'TRADE'])
    with tab_watch: _show(df[df['label'] == 'WATCHLIST'])

    st.divider()
    sel = st.selectbox('Inspect candidate', options=list(range(len(candidates_data))),
                       format_func=lambda i: f"{candidates_data[i]['contract']['ticker']} "
                                             f"{candidates_data[i]['strategy']} "
                                             f"{candidates_data[i]['contract']['strike']} "
                                             f"{candidates_data[i]['contract']['expiration']}")
    if sel is not None:
        c = candidates_data[sel]
        st.write(c)

    if st.button('Refresh from latest.json'):
        st.rerun()
