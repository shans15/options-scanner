"""Hardcoded macro-event calendar (FOMC / CPI / PPI / NFP).

Update this file quarterly with announced future dates from
fomc.gov / bls.gov / atlanta fed economic calendar. Versioning in
git history; no live API dependency.
"""
from __future__ import annotations
from datetime import date


# Format: (date, event_kind). Listed chronologically.
# Source: official Federal Reserve, BLS, and BEA calendars.
MACRO_EVENTS: list[tuple[date, str]] = [
    (date(2026, 1, 2), 'NFP'),
    (date(2026, 1, 14), 'CPI'),
    (date(2026, 1, 15), 'PPI'),
    (date(2026, 1, 28), 'FOMC'),
    (date(2026, 2, 6), 'NFP'),
    (date(2026, 2, 11), 'CPI'),
    (date(2026, 2, 12), 'PPI'),
    (date(2026, 3, 6), 'NFP'),
    (date(2026, 3, 11), 'CPI'),
    (date(2026, 3, 12), 'PPI'),
    (date(2026, 3, 18), 'FOMC'),
    (date(2026, 4, 3), 'NFP'),
    (date(2026, 4, 14), 'CPI'),
    (date(2026, 4, 15), 'PPI'),
    (date(2026, 4, 29), 'FOMC'),
    (date(2026, 5, 1), 'NFP'),
    (date(2026, 5, 13), 'CPI'),
    (date(2026, 5, 14), 'PPI'),
    (date(2026, 6, 5), 'NFP'),
    (date(2026, 6, 10), 'CPI'),
    (date(2026, 6, 11), 'PPI'),
    (date(2026, 6, 17), 'FOMC'),
    (date(2026, 7, 2), 'NFP'),
    (date(2026, 7, 14), 'CPI'),
    (date(2026, 7, 15), 'PPI'),
    (date(2026, 7, 29), 'FOMC'),
    (date(2026, 8, 7), 'NFP'),
    (date(2026, 8, 12), 'CPI'),
    (date(2026, 8, 13), 'PPI'),
    (date(2026, 9, 4), 'NFP'),
    (date(2026, 9, 9), 'CPI'),
    (date(2026, 9, 10), 'PPI'),
    (date(2026, 9, 16), 'FOMC'),
    (date(2026, 10, 2), 'NFP'),
    (date(2026, 10, 14), 'CPI'),
    (date(2026, 10, 15), 'PPI'),
    (date(2026, 11, 4), 'FOMC'),
    (date(2026, 11, 6), 'NFP'),
    (date(2026, 11, 12), 'CPI'),
    (date(2026, 11, 13), 'PPI'),
    (date(2026, 12, 4), 'NFP'),
    (date(2026, 12, 9), 'CPI'),
    (date(2026, 12, 10), 'PPI'),
    (date(2026, 12, 16), 'FOMC'),
]


def days_to_next_event(today: date) -> int | None:
    """Return whole calendar days until next blocking macro event (>=0).
    Returns None if `today` is after all known events."""
    upcoming = [d for d, _ in MACRO_EVENTS if d >= today]
    if not upcoming:
        return None
    return (upcoming[0] - today).days
