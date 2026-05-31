from __future__ import annotations
from datetime import date, timedelta
import pandas as pd
from yahooquery import Ticker


def has_earnings_within(ticker: str, days: int) -> bool:
    try:
        events = Ticker(ticker).calendar_events
    except Exception:
        return False
    if not isinstance(events, dict):
        return False
    info = events.get(ticker, {})
    if not isinstance(info, dict):
        return False
    earnings_info = info.get('earnings', {}) or {}
    dates = earnings_info.get('earningsDate', []) or []
    cutoff = date.today() + timedelta(days=days)
    for d in dates:
        try:
            d_norm = pd.Timestamp(d).date()
        except Exception:
            continue
        if date.today() <= d_norm <= cutoff:
            return True
    return False
