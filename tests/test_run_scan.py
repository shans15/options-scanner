from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from scanner.run_scan import run_full_scan


def _make_ticker_score(ticker: str = 'SPY', price: float = 530.0) -> object:
    class TS:
        pass
    ts = TS()
    ts.ticker = ticker
    ts.current_price = price
    ts.current_iv = 0.18
    ts.score = 75.0
    return ts


def _make_contract(ticker: str = 'SPY') -> dict:
    return {
        'ticker': ticker, 'strategy': 'naked_put', 'expiration': '2026-05-15',
        'strike': 515.0, 'bid': 1.20, 'ask': 1.40, 'mid': 1.30,
        'volume': 800, 'open_interest': 2000, 'implied_volatility': 0.18,
        'delta': -0.15, 'gamma': 0.03, 'theta': -0.02, 'vega': 0.10,
        'dte': 11, 'spot_price': 530.0,
    }


def _make_hist() -> pd.Series:
    np.random.seed(42)
    prices = 500 + np.cumsum(np.random.normal(0, 5, 252))
    return pd.Series(prices, name='Close')


def test_run_full_scan_returns_list(tmp_path: Path) -> None:
    mock_hist_df = pd.DataFrame({'Close': _make_hist()})

    with patch('scanner.run_scan.screen_universe', return_value=[_make_ticker_score()]), \
         patch('scanner.run_scan.fetch_contracts', return_value=[_make_contract()]), \
         patch('scanner.run_scan.yf.download', return_value=mock_hist_df), \
         patch('scanner.run_scan.export_scan') as mock_export:
        results = run_full_scan(output_dir=tmp_path)

    assert isinstance(results, list)
    mock_export.assert_called_once()


def test_run_full_scan_sorts_trade_first(tmp_path: Path) -> None:
    """TRADE decisions should appear before NO TRADE."""
    mock_hist_df = pd.DataFrame({'Close': _make_hist()})

    # Two contracts — we'll patch build_scan_result to control decisions
    contracts = [_make_contract('SPY'), _make_contract('AAPL')]

    trade_result = {'decision': 'TRADE', 'composite_score': 80.0, 'ticker': 'SPY'}
    no_trade_result = {'decision': 'NO TRADE', 'composite_score': 30.0, 'ticker': 'AAPL'}

    call_count = {'n': 0}
    def fake_build(contract, pop_result, hard_filter_passed, filter_reason):
        result = trade_result if call_count['n'] == 0 else no_trade_result
        call_count['n'] += 1
        return result

    with patch('scanner.run_scan.screen_universe', return_value=[_make_ticker_score()]), \
         patch('scanner.run_scan.fetch_contracts', return_value=contracts), \
         patch('scanner.run_scan.yf.download', return_value=mock_hist_df), \
         patch('scanner.run_scan.build_scan_result', side_effect=fake_build), \
         patch('scanner.run_scan.export_scan'):
        results = run_full_scan(output_dir=tmp_path)

    assert results[0]['decision'] == 'TRADE'
    assert results[-1]['decision'] == 'NO TRADE'


def test_run_full_scan_handles_empty_candidates(tmp_path: Path) -> None:
    with patch('scanner.run_scan.screen_universe', return_value=[]), \
         patch('scanner.run_scan.export_scan') as mock_export:
        results = run_full_scan(output_dir=tmp_path)

    assert results == []
    mock_export.assert_called_once_with([], output_dir=tmp_path)
