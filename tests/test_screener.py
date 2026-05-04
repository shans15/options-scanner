from __future__ import annotations
from unittest.mock import patch
import pandas as pd
import numpy as np
from scanner.screener import compute_iv_rank, compute_rsi, score_ticker, TickerScore


def _make_price_series(n: int = 252, trend: str = 'up') -> pd.Series:
    prices = np.linspace(100, 130 if trend == 'up' else 70, n)
    return pd.Series(prices)


def test_compute_iv_rank_in_range():
    prices = _make_price_series()
    rank = compute_iv_rank(0.28, prices)
    assert 0.0 <= rank <= 100.0


def test_compute_rsi_in_range():
    prices = _make_price_series()
    rsi = compute_rsi(prices)
    assert 0.0 <= rsi <= 100.0


def test_score_ticker_returns_ticker_score():
    prices = _make_price_series(trend='up')
    result = score_ticker(
        ticker='AAPL',
        current_price=150.0,
        avg_volume=5_000_000,
        current_iv=0.28,
        price_history=prices,
        vix_level=18.0,
        spy_trend='up',
    )
    assert isinstance(result, TickerScore)
    assert 0.0 <= result.score <= 100.0
    assert result.ticker == 'AAPL'


def test_low_volume_ticker_scores_low():
    prices = _make_price_series()
    result = score_ticker(
        ticker='TINY',
        current_price=5.0,
        avg_volume=100_000,
        current_iv=0.20,
        price_history=prices,
        vix_level=18.0,
        spy_trend='up',
    )
    assert result.score < 40.0


def test_high_vix_penalizes_regime_score():
    prices = _make_price_series(trend='up')
    high_vix = score_ticker('SPY', 500.0, 10_000_000, 0.30, prices, vix_level=40.0, spy_trend='up')
    normal_vix = score_ticker('SPY', 500.0, 10_000_000, 0.30, prices, vix_level=20.0, spy_trend='up')
    assert high_vix.score < normal_vix.score
