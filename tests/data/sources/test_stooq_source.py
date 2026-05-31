from io import StringIO
from unittest.mock import patch
import pandas as pd
import pytest
from data.sources.stooq_source import StooqSource


def test_fetch_price_history_parses_stooq_csv():
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-05-27,630,634,629,631,1000\n"
        "2026-05-28,631,633,630,632,1100\n"
    )
    with patch('data.sources.stooq_source.pd.read_csv', return_value=pd.read_csv(StringIO(csv))):
        out = StooqSource().fetch_price_history('SPY', 30)
        assert out.tolist() == [631.0, 632.0]


def test_fetch_spot_returns_last_close():
    csv = "Date,Open,High,Low,Close,Volume\n2026-05-28,631,633,630,632,1100\n"
    with patch('data.sources.stooq_source.pd.read_csv', return_value=pd.read_csv(StringIO(csv))):
        assert StooqSource().fetch_spot('SPY') == 632.0


def test_fetch_option_chain_returns_empty_list():
    assert StooqSource().fetch_option_chain('SPY') == []
