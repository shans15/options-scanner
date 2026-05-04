# tests/test_clock.py
from __future__ import annotations
from unittest.mock import patch
from datetime import datetime
import pytz
from scanner.clock import is_market_open, get_next_scan_times, SCAN_TIMES_ET

ET = pytz.timezone('America/New_York')

def test_scan_times_defined():
    assert len(SCAN_TIMES_ET) == 4
    assert (9, 45) in SCAN_TIMES_ET
    assert (11, 0) in SCAN_TIMES_ET
    assert (13, 0) in SCAN_TIMES_ET
    assert (15, 0) in SCAN_TIMES_ET

def test_market_closed_on_weekend():
    saturday = ET.localize(datetime(2026, 5, 2, 10, 0, 0))
    with patch('scanner.clock._now_et', return_value=saturday):
        assert is_market_open() is False

def test_market_open_on_weekday_during_hours():
    monday_10am = ET.localize(datetime(2026, 5, 4, 10, 0, 0))
    with patch('scanner.clock._now_et', return_value=monday_10am):
        assert is_market_open() is True

def test_market_closed_before_open():
    monday_8am = ET.localize(datetime(2026, 5, 4, 8, 0, 0))
    with patch('scanner.clock._now_et', return_value=monday_8am):
        assert is_market_open() is False

def test_get_next_scan_times_returns_list():
    times = get_next_scan_times()
    assert isinstance(times, list)
    assert all(isinstance(t, str) for t in times)
