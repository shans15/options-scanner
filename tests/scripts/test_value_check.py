"""value_check — single-ticker deep dive against monthly cache."""
from pathlib import Path

import pandas as pd

from scripts.value_check import check_ticker


def _write_cache(tmp_path: Path, ym: str, rows: list[dict]):
    cache = tmp_path / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(cache / f'fundamentals_{ym}.parquet')
    return cache


def test_check_ticker_prints_full_breakdown(tmp_path):
    """A cached ticker returns a dict with the breakdown fields."""
    from datetime import datetime
    ym = datetime.utcnow().strftime('%Y-%m')
    rows = [{
        'ticker': f'T{i}', 'sector': 'Technology', 'price': 20.0,
        'forward_eps': 2.0, 'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 1e9,
        'eps_ttm': 2.0, 'eps_ttm_1y_ago': 2.0,
        'book_value_per_share': 10.0, 'enterprise_value': 2e10,
        'ebitda_ttm': 2e9,
        'shares_short': 5e7, 'float_shares': 1e9, 'avg_daily_volume_30d': 1e6,
        'volume_5d_avg': 1e6, 'volume_20d_avg': 1e6, 'price_1y_ago': 20.0,
    } for i in range(10)]
    rows.append({
        'ticker': 'X', 'sector': 'Technology', 'price': 10.0,
        'forward_eps': 2.5, 'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 0.8e9,
        'eps_ttm': 2.0, 'eps_ttm_1y_ago': 1.5,
        'book_value_per_share': 5.0, 'enterprise_value': 2e9,
        'ebitda_ttm': 2e8,
        'shares_short': 5e7, 'float_shares': 5e8, 'avg_daily_volume_30d': 1e6,
        'volume_5d_avg': 1.2e6, 'volume_20d_avg': 1e6, 'price_1y_ago': 8.0,
    })
    cache = _write_cache(tmp_path, ym, rows)

    out = check_ticker('X', cache_dir=cache)
    assert out is not None
    assert out['ticker'] == 'X'
    assert 'sector_relative_score' in out
    assert 'fundamental_divergence_score' in out
    assert 'composite_score' in out


def test_check_ticker_missing_returns_none(tmp_path):
    from datetime import datetime
    ym = datetime.utcnow().strftime('%Y-%m')
    _write_cache(tmp_path, ym, [{
        'ticker': 'A', 'sector': 'Technology',
        'price': 20.0, 'forward_eps': 2.0,
        'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 1e9,
        'eps_ttm': 2.0, 'eps_ttm_1y_ago': 2.0,
        'book_value_per_share': 10.0, 'enterprise_value': 2e10,
        'ebitda_ttm': 2e9,
        'shares_short': 5e7, 'float_shares': 1e9,
        'avg_daily_volume_30d': 1e6,
        'volume_5d_avg': 1e6, 'volume_20d_avg': 1e6,
        'price_1y_ago': 20.0,
    }])
    cache = tmp_path / 'cache'
    out = check_ticker('DOES_NOT_EXIST', cache_dir=cache)
    assert out is None
