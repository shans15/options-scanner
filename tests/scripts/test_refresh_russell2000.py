"""Test refresh_russell2000 — behavior only, mocked network."""
from unittest.mock import patch
import io

import pandas as pd

from scripts.refresh_russell2000 import fetch_constituents


def test_fetch_constituents_normalises_dot_to_dash():
    csv = 'Ticker,Name\nAAPL,Apple\nBRK.B,Berkshire\nGOOG,Google\n'
    fake = pd.read_csv(io.StringIO(csv))
    with patch('scripts.refresh_russell2000.pd.read_csv', return_value=fake):
        result = fetch_constituents(url='https://fake')
    assert 'AAPL' in result
    assert 'BRK-B' in result
    assert 'BRK.B' not in result


def test_fetch_constituents_deduplicates():
    csv = 'Ticker,Name\nAAPL,Apple\nAAPL,Apple\nMSFT,Microsoft\n'
    fake = pd.read_csv(io.StringIO(csv))
    with patch('scripts.refresh_russell2000.pd.read_csv', return_value=fake):
        result = fetch_constituents(url='https://fake')
    assert result.count('AAPL') == 1
    assert len(result) == 2


def test_fetch_constituents_raises_when_no_ticker_column():
    csv = 'X,Y\n1,2\n'
    fake = pd.read_csv(io.StringIO(csv))
    with patch('scripts.refresh_russell2000.pd.read_csv', return_value=fake):
        try:
            fetch_constituents(url='https://fake')
        except RuntimeError:
            return
        raise AssertionError('expected RuntimeError')
