from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from pipeline.earnings import has_earnings_within


def test_returns_true_when_earnings_within_window():
    soon = date.today() + timedelta(days=3)
    fake_events = {'SPY': {'earnings': {'earningsDate': [pd.Timestamp(soon)]}}}
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.return_value.calendar_events = fake_events
        assert has_earnings_within('SPY', days=5) is True


def test_returns_false_when_no_earnings_in_window():
    far = date.today() + timedelta(days=30)
    fake_events = {'SPY': {'earnings': {'earningsDate': [pd.Timestamp(far)]}}}
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.return_value.calendar_events = fake_events
        assert has_earnings_within('SPY', days=5) is False


def test_returns_false_when_no_earnings_data():
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.return_value.calendar_events = {'SPY': {}}
        assert has_earnings_within('SPY', days=5) is False


def test_returns_false_when_yahooquery_throws():
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.side_effect = RuntimeError('network')
        assert has_earnings_within('SPY', days=5) is False
