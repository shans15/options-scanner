# tests/test_chain_fetcher.py
from __future__ import annotations
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from scanner.chain_fetcher import (
    parse_contract, filter_by_delta, _safe_float, _safe_int
)

# Use a dynamic expiration 20 days from today so it always falls within DTE_MIN=3 and DTE_MAX=45
_EXP_DATE = (date.today() + timedelta(days=20)).strftime('%Y-%m-%d')

RAW_PUT = {
    'strike_price': '185.0',
    'bid_price': '1.10',
    'ask_price': '1.30',
    'volume': '450',
    'open_interest': '1200',
    'implied_volatility': '0.2800',
    'delta': '-0.1800',
    'gamma': '0.0500',
    'theta': '-0.0400',
    'vega': '0.1200',
    'expiration_date': _EXP_DATE,  # 20 days out — within DTE_MIN=3, DTE_MAX=45
}


def test_parse_contract_put():
    c = parse_contract(RAW_PUT, ticker='AAPL', spot_price=195.0, option_type='put')
    assert c is not None
    assert c['ticker'] == 'AAPL'
    assert c['strategy'] == 'naked_put'
    assert c['strike'] == 185.0
    assert c['bid'] == 1.10
    assert c['delta'] == -0.18
    assert c['dte'] > 0


def test_parse_contract_call():
    raw = {**RAW_PUT, 'delta': '0.1800'}
    c = parse_contract(raw, ticker='AAPL', spot_price=195.0, option_type='call')
    assert c is not None
    assert c['strategy'] == 'naked_call'


def test_parse_contract_mid_calculation():
    c = parse_contract(RAW_PUT, ticker='SPY', spot_price=530.0, option_type='put')
    assert c is not None
    assert abs(c['mid'] - 1.20) < 0.01


def test_filter_by_delta_puts():
    contracts = [
        {'delta': -0.04, 'strategy': 'naked_put'},   # too low — filter out
        {'delta': -0.20, 'strategy': 'naked_put'},   # keep
        {'delta': -0.35, 'strategy': 'naked_put'},   # keep
        {'delta': -0.50, 'strategy': 'naked_put'},   # too high — filter out
    ]
    filtered = filter_by_delta(contracts)
    assert len(filtered) == 2
    assert all(0.05 <= abs(c['delta']) <= 0.35 for c in filtered)


def test_parse_contract_missing_fields_returns_none():
    c = parse_contract({}, ticker='AAPL', spot_price=195.0, option_type='put')
    assert c is None


def test_safe_float_handles_none():
    assert _safe_float(None) == 0.0
    assert _safe_float('1.23') == 1.23
    assert _safe_float('bad') == 0.0


def test_safe_int_handles_none():
    assert _safe_int(None) == 0
    assert _safe_int('450') == 450
    assert _safe_int('bad') == 0
