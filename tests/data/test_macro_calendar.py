from datetime import date
from data.sources.macro_calendar import days_to_next_event, MACRO_EVENTS


def test_macro_events_list_not_empty():
    assert len(MACRO_EVENTS) > 0
    for d, kind in MACRO_EVENTS:
        assert isinstance(d, date)
        assert kind in ('FOMC', 'CPI', 'PPI', 'NFP')


def test_days_to_next_event_returns_zero_on_event_day():
    if not MACRO_EVENTS:
        return
    first_event_date, _ = MACRO_EVENTS[0]
    assert days_to_next_event(first_event_date) == 0


def test_days_to_next_event_returns_positive_before_event():
    if not MACRO_EVENTS:
        return
    first_event_date, _ = MACRO_EVENTS[0]
    day_before = date.fromordinal(first_event_date.toordinal() - 1)
    assert days_to_next_event(day_before) == 1


def test_days_to_next_event_returns_none_after_last_event():
    if not MACRO_EVENTS:
        return
    last_event_date, _ = MACRO_EVENTS[-1]
    day_after = date.fromordinal(last_event_date.toordinal() + 1)
    assert days_to_next_event(day_after) is None
