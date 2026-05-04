# scanner/clock.py
from __future__ import annotations
from datetime import datetime
import pytz
import exchange_calendars as xcal

ET = pytz.timezone('America/New_York')
SCAN_TIMES_ET = [(9, 45), (11, 0), (13, 0), (15, 0)]
_nyse = xcal.get_calendar('XNYS')


def _now_et() -> datetime:
    return datetime.now(ET)


def is_market_open() -> bool:
    """Return True if NYSE is currently open."""
    import pandas as pd

    now = _now_et()
    today_ts = pd.Timestamp(now.date())

    # Check if today is a trading session
    if today_ts not in _nyse.sessions:
        return False

    # Get market open and close times for today
    market_open = _nyse.session_open(today_ts)
    market_close = _nyse.session_close(today_ts)

    return market_open <= now <= market_close


def get_next_scan_times() -> list[str]:
    """Return list of today's remaining scan times as 'HH:MM ET' strings."""
    now = _now_et()
    remaining = []
    for hour, minute in SCAN_TIMES_ET:
        scan_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if scan_dt > now:
            remaining.append(f"{hour:02d}:{minute:02d} ET")
    return remaining


def get_market_status() -> str:
    """Return human-readable market status string."""
    if is_market_open():
        next_scans = get_next_scan_times()
        next_str = next_scans[0] if next_scans else 'None today'
        return f"OPEN | Next scan: {next_str}"
    return "CLOSED"
