from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd

from domain.aplus.market_context import (
    score_spx_trend, score_dxy_trend, score_yield_10y, score_vvix,
    rank_sectors, build_market_context,
)


def _series(values, start='2026-01-01'):
    idx = pd.date_range(start, periods=len(values), freq='D')
    return pd.Series(values, index=idx)


def test_score_spx_trend_bullish_setup_strong_uptrend_high_score():
    closes = _series(list(range(80, 130)))  # rising
    score = score_spx_trend(closes, setup_direction='bullish')
    assert score >= 8.0


def test_score_spx_trend_bullish_setup_strong_downtrend_low_score():
    closes = _series(list(range(130, 80, -1)))
    score = score_spx_trend(closes, setup_direction='bullish')
    assert score <= 4.0


def test_score_dxy_trend_neutral_when_mixed():
    closes = _series([100.0] * 60)
    score = score_dxy_trend(closes, setup_direction='bullish')
    assert 4.0 <= score <= 6.0


def test_score_vvix_low_high_score():
    score = score_vvix(85.0)
    assert score >= 8.0


def test_score_vvix_high_low_score():
    score = score_vvix(125.0)
    assert score <= 4.0


def test_rank_sectors_assigns_rank_1_to_best_performer():
    returns = {'XLK': 0.05, 'XLF': 0.01, 'XLE': -0.02}
    ranks = rank_sectors(returns)
    assert ranks['XLK'] == 1


def test_build_market_context_returns_market_context():
    with patch('domain.aplus.market_context._fetch_macro_series') as m:
        m.return_value = _series([100.0] * 60)
        with patch('domain.aplus.market_context._fetch_vvix') as m_vvix:
            m_vvix.return_value = 95.0
            with patch('domain.aplus.market_context._fetch_sector_returns') as m_sec:
                m_sec.return_value = {'XLK': 0.05, 'XLF': 0.01}
                from domain.aplus.types import MarketContext
                ctx = build_market_context(today=date(2026, 6, 14), setup_direction='bullish')
                assert isinstance(ctx, MarketContext)
                assert 0 <= ctx.spx_trend_score <= 10
                assert 0 <= ctx.dxy_trend_score <= 10
                assert 0 <= ctx.vvix_score <= 10
                assert ctx.days_to_macro_event is None or ctx.days_to_macro_event >= 0
