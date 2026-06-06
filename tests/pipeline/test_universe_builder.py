import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
from data.sources.base import DataSource
from pipeline.universe_builder import (
    UniverseFilters, fetch_sp500_constituents, build_universe, build_universe_cached
)


class _FakeSource(DataSource):
    def __init__(self, volumes: dict[str, int], chains: dict[str, list]):
        self._volumes = volumes
        self._chains = chains
    def fetch_spot(self, ticker): return 100.0
    def fetch_price_history(self, ticker, lookback_days):
        vol = self._volumes.get(ticker, 0)
        return pd.Series([vol] * lookback_days)
    def fetch_option_chain(self, ticker):
        return self._chains.get(ticker, [])
    def fetch_price_history_ohlcv(self, ticker, lookback_days): return pd.DataFrame()


def test_fetch_sp500_constituents_returns_list_of_strings():
    fake_df = pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'BRK.B']})
    with patch('pipeline.universe_builder.pd.read_html', return_value=[fake_df]):
        out = fetch_sp500_constituents()
        assert 'AAPL' in out and 'MSFT' in out
        assert 'BRK-B' in out


def test_build_universe_filters_by_volume():
    src = _FakeSource(
        volumes={'AAA': 2_000_000, 'BBB': 500_000, 'CCC': 5_000_000},
        chains={'AAA': ['x'], 'BBB': ['x'], 'CCC': ['x']},
    )
    with patch('pipeline.universe_builder.fetch_sp500_constituents', return_value=['AAA', 'BBB', 'CCC']):
        out = build_universe(UniverseFilters(min_avg_volume=1_000_000, top_n=10), [src])
        assert 'BBB' not in out
        assert out == ['CCC', 'AAA']


def test_build_universe_drops_no_options():
    src = _FakeSource(
        volumes={'AAA': 2_000_000, 'BBB': 5_000_000},
        chains={'AAA': ['x'], 'BBB': []},
    )
    with patch('pipeline.universe_builder.fetch_sp500_constituents', return_value=['AAA', 'BBB']):
        out = build_universe(UniverseFilters(require_options_chain=True), [src])
        assert out == ['AAA']


def test_build_universe_top_n_caps_list():
    src = _FakeSource(
        volumes={f'T{i:02}': 1_000_000 + i*1000 for i in range(20)},
        chains={f'T{i:02}': ['x'] for i in range(20)},
    )
    tickers = [f'T{i:02}' for i in range(20)]
    with patch('pipeline.universe_builder.fetch_sp500_constituents', return_value=tickers):
        out = build_universe(UniverseFilters(top_n=5), [src])
        assert len(out) == 5
        assert out[0] == 'T19'


def test_build_universe_cached_uses_cache_if_exists(tmp_path):
    cache_file = tmp_path / 'universe_2099-01-01.json'
    cache_file.write_text(json.dumps(['SPY', 'QQQ']))
    with patch('pipeline.universe_builder.date') as mock_date:
        mock_date.today.return_value = type('D', (), {'isoformat': lambda self: '2099-01-01'})()
        out = build_universe_cached(UniverseFilters(), sources=[], cache_dir=tmp_path)
        assert out == ['SPY', 'QQQ']


def test_build_universe_falls_back_to_known_good_when_scrape_fails():
    src = _FakeSource(volumes={}, chains={})
    with patch('pipeline.universe_builder.fetch_sp500_constituents', side_effect=RuntimeError('wiki down')):
        out = build_universe(UniverseFilters(top_n=3), [src])
        assert out == ['SPY', 'QQQ', 'IWM']
