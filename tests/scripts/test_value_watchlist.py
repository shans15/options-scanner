"""Integration test — end-to-end value_watchlist pipeline."""
import json
from pathlib import Path
from unittest.mock import patch

from scripts.value_watchlist import run_watchlist


def _mini_rows() -> dict[str, dict]:
    """Build a small in-memory universe of 11 tickers: 1 target + 10 peers."""
    fixture = json.loads(
        Path('tests/scripts/fixtures/value_mini_universe.json').read_text()
    )
    fake_rows: dict[str, dict] = {r['ticker']: r for r in fixture}
    for i in range(10):
        tk = f'PEER{i}'
        fake_rows[tk] = {
            'ticker': tk, 'sector': 'Technology', 'price': 20.0,
            'forward_eps': 2.0, 'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 1e9,
            'eps_ttm': 2.0, 'eps_ttm_1y_ago': 2.0,
            'book_value_per_share': 10.0, 'enterprise_value': 2e10,
            'ebitda_ttm': 2e9, 'shares_short': 5e7, 'float_shares': 1e9,
            'avg_daily_volume_30d': 1e6, 'volume_5d_avg': 1e6, 'volume_20d_avg': 1e6,
            'price_1y_ago': 20.0,
        }
    return fake_rows


def test_run_watchlist_with_mocked_fundamentals(tmp_path):
    fake_rows = _mini_rows()

    with patch('scripts.value_watchlist.fetch_fundamentals_batch',
                return_value=fake_rows):
        report = run_watchlist(
            constituents=list(fake_rows.keys()),
            out_dir=tmp_path, cache_dir=tmp_path / 'cache',
            universe_name='mini_universe', top_n=5, use_cache=False,
        )

    assert report.universe == 'mini_universe'
    assert report.graded_size >= 1
    assert (tmp_path / 'latest.json').exists()

    out_data = json.loads((tmp_path / 'latest.json').read_text())
    assert 'role1_ranked_longs' in out_data
    assert 'role2_ranked_longs' in out_data


def test_cache_hit_skips_fetch(tmp_path):
    """When a monthly cache Parquet exists, do NOT call fetch_fundamentals_batch."""
    import pandas as pd
    from datetime import datetime
    cache = tmp_path / 'cache'
    cache.mkdir()
    ym = datetime.utcnow().strftime('%Y-%m')
    fixture = json.loads(
        Path('tests/scripts/fixtures/value_mini_universe.json').read_text()
    )
    df = pd.DataFrame(fixture)
    df.to_parquet(cache / f'fundamentals_{ym}.parquet')

    with patch('scripts.value_watchlist.fetch_fundamentals_batch') as spy:
        run_watchlist(
            constituents=['A'], out_dir=tmp_path, cache_dir=cache,
            universe_name='mini', top_n=5, use_cache=True,
        )
        assert spy.call_count == 0
