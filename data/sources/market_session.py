"""US equity market session detection — pre/regular/post/closed.

All times in US/Eastern. Doesn't account for half-days (e.g., day after
Thanksgiving) — those will return 'regular' incorrectly for the partial-day
close hour. Acceptable trade-off for now; revisit if it becomes a real bug.
"""
from __future__ import annotations
from datetime import datetime, time
from typing import Literal

try:
    from zoneinfo import ZoneInfo  # py 3.9+
    _ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    _ET = pytz.timezone("America/New_York")


Session = Literal["closed", "pre_market", "regular", "post_market"]


def current_session(now: datetime | None = None) -> Session:
    """Return the current US equity market session.

    Pre-market:  Mon-Fri 04:00 - 09:30 ET
    Regular:     Mon-Fri 09:30 - 16:00 ET
    Post-market: Mon-Fri 16:00 - 20:00 ET
    Closed:      Sat/Sun all day, plus Mon-Fri 20:00 - 04:00
    """
    now = now or datetime.now(tz=_ET)
    if now.tzinfo is None:
        # Assume naive datetime is already in ET
        now = now.replace(tzinfo=_ET)
    else:
        now = now.astimezone(_ET)

    # Weekend
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return "closed"

    t = now.time()
    if t < time(4, 0):
        return "closed"
    if t < time(9, 30):
        return "pre_market"
    if t < time(16, 0):
        return "regular"
    if t < time(20, 0):
        return "post_market"
    return "closed"


def is_extended_hours(session: Session) -> bool:
    """True if pre_market or post_market session."""
    return session in ("pre_market", "post_market")
