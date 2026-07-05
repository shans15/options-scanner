"""Normalize raw yahooquery payload → ValuationInputs."""
from domain.value.fundamentals import normalize_fundamentals


def test_normalize_full_payload_populates_all_fields():
    raw = {
        'ticker': 'INTC',
        'sector': 'Technology',
        'price': 24.11,
        'forward_eps': 2.87,
        'revenue_ttm': 54_000_000_000,
        'revenue_ttm_1y_ago': 48_000_000_000,
        'eps_ttm': 1.4,
        'eps_ttm_1y_ago': 1.29,
        'book_value_per_share': 21.9,
        'enterprise_value': 100_000_000_000,
        'ebitda_ttm': 15_400_000_000,
        'shares_short': 180_000_000,
        'float_shares': 1_040_000_000,
        'avg_daily_volume_30d': 29_000_000,
        'volume_5d_avg': 45_000_000,
        'volume_20d_avg': 38_000_000,
        'price_1y_ago': 29.55,
    }
    v = normalize_fundamentals(raw)
    assert v.ticker == 'INTC'
    assert v.sector == 'Technology'
    assert v.price == 24.11
    assert v.forward_eps == 2.87


def test_normalize_missing_optional_fields_returns_none():
    raw = {'ticker': 'X', 'sector': 'Technology'}
    v = normalize_fundamentals(raw)
    assert v.ticker == 'X'
    assert v.price is None
    assert v.forward_eps is None
    assert v.eps_ttm is None


def test_normalize_nan_treated_as_none():
    import math
    raw = {'ticker': 'X', 'sector': None, 'price': math.nan}
    v = normalize_fundamentals(raw)
    assert v.price is None
    assert v.sector is None


def test_normalize_missing_ticker_raises():
    import pytest
    with pytest.raises((KeyError, ValueError)):
        normalize_fundamentals({'sector': 'Technology'})


def test_normalize_zero_prices_kept_as_zero_not_none():
    """Zero is a valid value distinct from missing; only NaN/None get nulled."""
    raw = {'ticker': 'X', 'price': 0.0, 'revenue_ttm': 0}
    v = normalize_fundamentals(raw)
    assert v.price == 0.0
    assert v.revenue_ttm == 0
