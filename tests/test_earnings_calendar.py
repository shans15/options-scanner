# tests/test_earnings_calendar.py
from __future__ import annotations
from unittest.mock import patch
from datetime import date, timedelta
from data.earnings_calendar import has_earnings_soon


def test_has_earnings_soon_returns_bool():
    with patch('data.earnings_calendar._get_earnings_date', return_value=None):
        result = has_earnings_soon('AAPL', days=5)
    assert isinstance(result, bool)


def test_earnings_within_window_returns_true():
    tomorrow = date.today() + timedelta(days=2)
    with patch('data.earnings_calendar._get_earnings_date', return_value=tomorrow):
        assert has_earnings_soon('AAPL', days=5) is True


def test_earnings_outside_window_returns_false():
    far_future = date.today() + timedelta(days=30)
    with patch('data.earnings_calendar._get_earnings_date', return_value=far_future):
        assert has_earnings_soon('AAPL', days=5) is False


def test_no_earnings_date_returns_false():
    with patch('data.earnings_calendar._get_earnings_date', return_value=None):
        assert has_earnings_soon('AAPL', days=5) is False
