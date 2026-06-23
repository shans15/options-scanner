from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo
from data.sources.market_session import current_session, is_extended_hours


_ET = ZoneInfo("America/New_York")


def test_weekday_3am_is_closed():
    dt = datetime(2026, 6, 23, 3, 0, tzinfo=_ET)
    assert current_session(dt) == "closed"


def test_weekday_4am_is_pre_market():
    dt = datetime(2026, 6, 23, 4, 0, tzinfo=_ET)
    assert current_session(dt) == "pre_market"


def test_weekday_830am_is_pre_market():
    dt = datetime(2026, 6, 23, 8, 30, tzinfo=_ET)
    assert current_session(dt) == "pre_market"


def test_weekday_930am_is_regular():
    dt = datetime(2026, 6, 23, 9, 30, tzinfo=_ET)
    assert current_session(dt) == "regular"


def test_weekday_noon_is_regular():
    dt = datetime(2026, 6, 23, 12, 0, tzinfo=_ET)
    assert current_session(dt) == "regular"


def test_weekday_4pm_is_post_market():
    dt = datetime(2026, 6, 23, 16, 0, tzinfo=_ET)
    assert current_session(dt) == "post_market"


def test_weekday_7pm_is_post_market():
    dt = datetime(2026, 6, 23, 19, 0, tzinfo=_ET)
    assert current_session(dt) == "post_market"


def test_weekday_8pm_is_closed():
    dt = datetime(2026, 6, 23, 20, 0, tzinfo=_ET)
    assert current_session(dt) == "closed"


def test_saturday_is_closed():
    dt = datetime(2026, 6, 27, 12, 0, tzinfo=_ET)  # Saturday
    assert current_session(dt) == "closed"


def test_sunday_is_closed():
    dt = datetime(2026, 6, 28, 12, 0, tzinfo=_ET)  # Sunday
    assert current_session(dt) == "closed"


def test_is_extended_hours_true_for_pre():
    assert is_extended_hours("pre_market") is True


def test_is_extended_hours_true_for_post():
    assert is_extended_hours("post_market") is True


def test_is_extended_hours_false_for_regular():
    assert is_extended_hours("regular") is False


def test_is_extended_hours_false_for_closed():
    assert is_extended_hours("closed") is False


def test_default_now_uses_current_time():
    # Just verify it doesn't crash with no args
    s = current_session()
    assert s in ("closed", "pre_market", "regular", "post_market")


# -- additional boundary tests --

def test_weekday_just_before_4am_is_closed():
    dt = datetime(2026, 6, 23, 3, 59, tzinfo=_ET)
    assert current_session(dt) == "closed"


def test_weekday_just_before_930am_is_pre_market():
    dt = datetime(2026, 6, 23, 9, 29, tzinfo=_ET)
    assert current_session(dt) == "pre_market"


def test_weekday_just_before_4pm_is_regular():
    dt = datetime(2026, 6, 23, 15, 59, tzinfo=_ET)
    assert current_session(dt) == "regular"


def test_weekday_just_before_8pm_is_post_market():
    dt = datetime(2026, 6, 23, 19, 59, tzinfo=_ET)
    assert current_session(dt) == "post_market"


def test_utc_datetime_is_converted_to_et():
    # 14:00 UTC on a weekday = 10:00 ET (EDT) → regular session
    dt = datetime(2026, 6, 23, 14, 0, tzinfo=timezone.utc)
    assert current_session(dt) == "regular"


def test_monday_morning_pre_market():
    # Monday at 5:00 AM ET
    dt = datetime(2026, 6, 22, 5, 0, tzinfo=_ET)
    assert current_session(dt) == "pre_market"


def test_friday_post_market():
    # Friday at 17:00 ET
    dt = datetime(2026, 6, 26, 17, 0, tzinfo=_ET)
    assert current_session(dt) == "post_market"
