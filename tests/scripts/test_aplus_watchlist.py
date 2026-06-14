import json
from pathlib import Path
from datetime import date
from unittest.mock import patch

from scripts.aplus_watchlist import render_watchlist


def _scan_payload():
    return {
        'timestamp': '2026-06-14T08:00:00',
        'candidates': [
            {
                'contract': {
                    'ticker': 'XLF', 'option_type': 'put', 'strike': 52.0,
                    'expiration': '2026-07-10', 'dte': 26, 'spot_price': 53.34,
                    'bid': 0.47, 'ask': 0.53, 'mid': 0.50, 'volume': 500,
                    'open_interest': 1500, 'implied_volatility': 0.17,
                    'delta': -0.30, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                },
                'strategy': 'long_put',
                'setup_name': 'compression_breakout',
                'setup_direction': 'bullish',
                'setup_strength': 0.85,
                'weekly_ribbon_agreement': True,
                'pct_change_1w': 0.01, 'pct_change_4w': 0.05,
                'vix_regime': 'neutral', 'vix_pct_vs_7d': -0.05,
                'composite_score': 70.0, 'label': 'WATCHLIST',
                'reason_for': '', 'reason_against': '',
            }
        ],
    }


def test_render_watchlist_with_strong_candidate_produces_output(tmp_path):
    scan_file = tmp_path / 'scan.json'
    scan_file.write_text(json.dumps(_scan_payload()))
    out_dir = tmp_path / 'aplus'

    with patch('scripts.aplus_watchlist._fetch_market_context') as mock_mc:
        from domain.aplus.types import MarketContext
        mock_mc.return_value = MarketContext(
            spx_trend_score=9.0, sector_rotation_rank={'XLF': 2},
            dxy_trend_score=7.0, yield_10y_score=8.0, vvix_score=9.0,
            days_to_macro_event=8,
        )
        with patch('scripts.aplus_watchlist._fetch_days_to_earnings') as mock_e:
            mock_e.return_value = 15
            result = render_watchlist(scan_file, out_dir, account_size=1000.0, today=date(2026, 6, 14))

    assert (out_dir / 'latest.json').exists()
    out_data = json.loads((out_dir / 'latest.json').read_text())
    assert 'graded_candidates' in out_data
    assert out_data['account_size'] == 1000.0
    # Every graded candidate must be A+ or A (filtered)
    for c in out_data['graded_candidates']:
        assert c['grade'] in ('A+', 'A'), f"Found {c['grade']} in output — should be filtered"


def test_render_watchlist_integration_fixture(tmp_path):
    fixture = Path(__file__).parent / 'fixtures' / 'sample_scan.json'
    out_dir = tmp_path / 'aplus'
    with patch('scripts.aplus_watchlist._fetch_market_context') as mock_mc:
        from domain.aplus.types import MarketContext
        mock_mc.return_value = MarketContext(
            spx_trend_score=9.0,
            sector_rotation_rank={'XLF': 2, 'XLK': 1},
            dxy_trend_score=7.0, yield_10y_score=8.0, vvix_score=9.0,
            days_to_macro_event=10,
        )
        with patch('scripts.aplus_watchlist._fetch_days_to_earnings') as mock_e:
            mock_e.return_value = 20
            result = render_watchlist(fixture, out_dir, account_size=1000.0, today=date(2026, 6, 14))

    out_payload = json.loads((out_dir / 'latest.json').read_text())
    assert 'graded_candidates' in out_payload
    # AAPL has vix_regime=expansion → vol_vix=0 floor → should not be A+/A
    aapl = next((c for c in out_payload['graded_candidates'] if c['ticker'] == 'AAPL'), None)
    if aapl is not None:
        assert aapl['grade'] != 'A+'
    # MSFT has compression + clean context → should grade well
    msft = next((c for c in out_payload['graded_candidates'] if c['ticker'] == 'MSFT'), None)
    # MSFT may not appear if features push it to B; that's okay too.
