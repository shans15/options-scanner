from __future__ import annotations
import json
from pathlib import Path
import pytest

from ui.watchlist import (
    render_watchlist, _liquidity_grade, _best_long_candidate,
    _find_liquid_fallback, _setup_confluence,
)


def _make_candidate(ticker='X', strategy='long_call', strike=100.0, dte=10,
                    mid=1.0, score=60.0, label='WATCHLIST',
                    setup_name='compression_breakout', setup_direction='bullish',
                    setup_strength=0.90, volume=500, open_interest=1000,
                    delta=0.55, pop=0.50, ev=0.4,
                    reason_against='', weekly_trend='up', streak=1,
                    p1w=0.01, p2w=0.02, p4w=0.04, agree=True) -> dict:
    return {
        'contract': {
            'ticker': ticker, 'strategy': strategy, 'strike': strike,
            'expiration': '2026-06-15', 'dte': dte, 'mid': mid,
            'option_type': 'call' if strategy.endswith('call') else 'put',
            'volume': volume, 'open_interest': open_interest,
            'delta': delta, 'implied_volatility': 0.22,
            'spot_price': 100.0, 'bid': mid - 0.05, 'ask': mid + 0.05,
            'gamma': 0.05, 'theta': -0.01, 'vega': 0.10,
        },
        'strategy': strategy,
        'composite_score': score, 'label': label,
        'pop_blended': pop, 'ev': ev,
        'setup_name': setup_name, 'setup_direction': setup_direction, 'setup_strength': setup_strength,
        'weekly_trend': weekly_trend, 'consecutive_close_streak': streak,
        'pct_change_1w': p1w, 'pct_change_2w': p2w, 'pct_change_4w': p4w,
        'weekly_ribbon_agreement': agree,
        'reason_for': '', 'reason_against': reason_against,
    }


def test_liquidity_grade_liquid_when_oi_and_volume_high():
    flag, label = _liquidity_grade({'volume': 1000, 'open_interest': 5000})
    assert flag == '✓' and 'liquid' in label.lower()


def test_liquidity_grade_thin_oi_when_oi_low():
    flag, label = _liquidity_grade({'volume': 100, 'open_interest': 200})
    assert flag == '⚠' and 'oi' in label.lower()


def test_liquidity_grade_illiquid_when_oi_very_low():
    flag, label = _liquidity_grade({'volume': 0, 'open_interest': 50})
    assert flag == '✗'


def test_liquidity_grade_thin_volume_when_oi_high_but_volume_low():
    flag, label = _liquidity_grade({'volume': 50, 'open_interest': 600})
    assert flag == '⚠' and 'volume' in label.lower()


def test_best_long_candidate_picks_highest_score():
    cs = [
        _make_candidate(ticker='X', score=50),
        _make_candidate(ticker='X', score=70, strike=105),
        _make_candidate(ticker='X', score=60, strike=110),
    ]
    out = _best_long_candidate(cs, 'X', 'bullish')
    assert out['composite_score'] == 70
    assert out['contract']['strike'] == 105


def test_best_long_candidate_filters_by_direction_alignment():
    cs = [
        _make_candidate(ticker='X', strategy='long_put', score=80, setup_direction='bullish'),  # wrong direction
        _make_candidate(ticker='X', strategy='long_call', score=60, setup_direction='bullish'),
    ]
    out = _best_long_candidate(cs, 'X', 'bullish')
    assert out['strategy'] == 'long_call'


def test_best_long_candidate_returns_none_when_no_match():
    cs = [
        _make_candidate(ticker='X', strategy='long_put', score=80, setup_direction='bullish'),
    ]
    out = _best_long_candidate(cs, 'X', 'bullish')
    assert out is None


def test_find_liquid_fallback_returns_higher_oi_alternate():
    cs = [
        _make_candidate(ticker='X', strike=100, open_interest=50),    # primary, illiquid
        _make_candidate(ticker='X', strike=105, open_interest=2000),  # better
        _make_candidate(ticker='X', strike=110, open_interest=500),
    ]
    out = _find_liquid_fallback(cs, 'X', 'bullish', exclude_strike=100.0)
    assert out['contract']['strike'] == 105
    assert out['contract']['open_interest'] == 2000


def test_find_liquid_fallback_returns_none_when_no_alternate():
    cs = [_make_candidate(ticker='X', strike=100)]
    out = _find_liquid_fallback(cs, 'X', 'bullish', exclude_strike=100.0)
    assert out is None


def test_setup_confluence_groups_setups_across_tickers():
    cs = [
        _make_candidate(ticker='SPY', setup_name='pullback_in_trend'),
        _make_candidate(ticker='QQQ', setup_name='pullback_in_trend'),
        _make_candidate(ticker='AAPL', setup_name='compression_breakout'),
    ]
    out = _setup_confluence(cs)
    pullback_tickers = out[('pullback_in_trend', 'bullish')]
    assert 'SPY' in pullback_tickers and 'QQQ' in pullback_tickers


def test_render_watchlist_smoke(tmp_path):
    """End-to-end: build a minimal scan JSON, render, ensure key sections appear."""
    payload = {
        'timestamp': '2026-06-06T04:51:00',
        'candidates': [
            _make_candidate(ticker='QQQ', strike=735, score=66.5, label='TRADE',
                            volume=1234, open_interest=5678, setup_strength=0.90),
            _make_candidate(ticker='TSLA', strike=417.5, score=63.2, label='WATCHLIST',
                            volume=89, open_interest=156, setup_strength=1.00, weekly_trend='down'),
            _make_candidate(ticker='TSLA', strike=420, score=52, label='NO_TRADE',
                            volume=1432, open_interest=4221),
        ],
        'skipped': {},
    }
    scan_path = tmp_path / 'scan.json'
    scan_path.write_text(json.dumps(payload))
    out = render_watchlist(scan_path)
    assert 'QQQ' in out
    assert 'TSLA' in out
    assert 'liquid' in out.lower()
    assert 'fallback' in out.lower()
