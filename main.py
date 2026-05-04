from __future__ import annotations
"""
Options Scanner — Entry Point

Run with:
    streamlit run main.py

The APScheduler starts in a background thread and triggers scans at:
    09:45, 11:00, 13:00, 15:00 ET (Monday-Friday)

The Streamlit dashboard reads from output/scans/latest.json and auto-refreshes every 60s.

This is quantitative research tooling only. Not financial advice.
"""
import streamlit as st

# Start scheduler once per process (singleton — safe across Streamlit reruns)
if 'scheduler_initialized' not in st.session_state:
    from scanner.scheduler import get_or_create_scheduler
    get_or_create_scheduler()
    st.session_state.scheduler_initialized = True

# Render the dashboard
from output.dashboard import render_dashboard
render_dashboard()
