# data/earnings_calendar.py
from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
import yfinance as yf


def _get_earnings_date(ticker: str) -> Optional[date]:
    """Return the next earnings date for a ticker, or None if unknown."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or (hasattr(cal, 'empty') and cal.empty):
            return None
        # calendar is a DataFrame; columns are dates
        if hasattr(cal, 'columns') and len(cal.columns) > 0:
            earnings_dt = cal.columns[0]
            if hasattr(earnings_dt, 'date'):
                return earnings_dt.date()
        return None
    except Exception:
        return None


def has_earnings_soon(ticker: str, days: int = 5) -> bool:
    """Return True if ticker has earnings within `days` calendar days."""
    earnings_date = _get_earnings_date(ticker)
    if earnings_date is None:
        return False
    today = date.today()
    return today <= earnings_date <= today + timedelta(days=days)
