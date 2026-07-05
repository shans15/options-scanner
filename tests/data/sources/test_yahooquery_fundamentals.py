"""yahooquery_fundamentals — mocked network tests only."""
from unittest.mock import MagicMock, patch

from data.sources.yahooquery_fundamentals import (
    extract_ticker_row,
    fetch_fundamentals_batch,
)


def test_extract_ticker_row_maps_all_fields():
    """Given a mocked yahooquery Ticker payload for one ticker, extract
    the fields we need into a plain dict."""
    yq = MagicMock()
    yq.asset_profile = {'INTC': {'sector': 'Technology'}}
    yq.key_stats = {'INTC': {
        'forwardEps': 2.87, 'bookValue': 21.9, 'enterpriseValue': 100_000_000_000,
        'sharesShort': 180_000_000, 'floatShares': 1_040_000_000,
    }}
    yq.income_statement = {'INTC': [
        {'periodType': 'TTM', 'TotalRevenue': 54_000_000_000, 'DilutedEPS': 1.4, 'EBITDA': 15_400_000_000},
        {'periodType': '12M', 'TotalRevenue': 48_000_000_000, 'DilutedEPS': 1.29, 'EBITDA': 14_000_000_000},
    ]}
    yq.summary_detail = {'INTC': {
        'regularMarketPrice': 24.11, 'averageVolume': 29_000_000,
    }}

    row = extract_ticker_row(yq, 'INTC', volume_5d_avg=45e6, volume_20d_avg=38e6,
                              price_1y_ago=29.55)
    assert row['ticker'] == 'INTC'
    assert row['sector'] == 'Technology'
    assert row['forward_eps'] == 2.87
    assert row['revenue_ttm'] == 54_000_000_000
    assert row['revenue_ttm_1y_ago'] == 48_000_000_000
    assert row['shares_short'] == 180_000_000
    assert row['price'] == 24.11
    assert row['volume_5d_avg'] == 45e6
    assert row['price_1y_ago'] == 29.55


def test_extract_ticker_row_returns_none_fields_on_missing():
    """Missing sub-dicts → all fields present but None."""
    yq = MagicMock()
    yq.asset_profile = {'X': None}
    yq.key_stats = {'X': None}
    yq.income_statement = {'X': None}
    yq.summary_detail = {'X': None}

    row = extract_ticker_row(yq, 'X', volume_5d_avg=None, volume_20d_avg=None,
                              price_1y_ago=None)
    assert row['ticker'] == 'X'
    assert row['sector'] is None
    assert row['forward_eps'] is None
    assert row['revenue_ttm'] is None


def test_fetch_fundamentals_batch_calls_yahooquery():
    """Wire-level test that fetch_fundamentals_batch splits into batches
    and calls yahooquery once per batch."""
    with patch('data.sources.yahooquery_fundamentals.Ticker') as MockTicker:
        yq_instance = MagicMock()
        MockTicker.return_value = yq_instance
        yq_instance.asset_profile = {}
        yq_instance.key_stats = {}
        yq_instance.income_statement = {}
        yq_instance.summary_detail = {}
        yq_instance.history = MagicMock(return_value=MagicMock(empty=True))

        result = fetch_fundamentals_batch(['A', 'B', 'C'], batch_size=2, sleep_seconds=0.0)

    # 2 batches expected: ['A','B'] and ['C']
    assert MockTicker.call_count == 2
    assert isinstance(result, dict)
