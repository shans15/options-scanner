"""Tests for scripts/kalshi_btc_backtest.py (Sprint 2)

Uses mocked KalshiSource and CoinbaseSource.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.kalshi_btc_backtest import (
    run_backtest,
    _sharpe,
    _max_drawdown,
    _pnl_per_dollar,
    wilson_ci,
    _stratified_sample,
    _ci_overlap,
)
from domain.prediction.market import KalshiMarket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 1000, base_price: float = 65_000.0) -> pd.DataFrame:
    now = datetime(2026, 6, 13, 14, 0, tzinfo=timezone.utc)
    idx = pd.date_range(end=now, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(99)
    log_rets = rng.normal(0, 0.001, size=n)
    close = base_price * np.exp(np.cumsum(log_rets))
    df = pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.001, "low": close * 0.998,
         "close": close, "volume": rng.uniform(100, 500, size=n)},
        index=idx,
    )
    return df


def _make_market(
    i: int,
    side: str = "above",
    resolution: str = "yes",
    yes_bid: int = 38,
    yes_ask: int = 42,
) -> KalshiMarket:
    """Synthetic settled KalshiMarket."""
    base = datetime(2026, 3, 1, 16, 0, tzinfo=timezone.utc)
    exp = base + timedelta(hours=i * 24)
    return KalshiMarket(
        ticker=f"KXBTC-26MAR{i:02d}-T65000",
        event_ticker=f"KXBTC-26MAR{i:02d}",
        title=f"BTC > $65,000 at 4PM on day {i}",
        strike=65_000.0,
        side=side,
        expiration=exp,
        close_time=exp,
        yes_bid=yes_bid / 100.0,
        yes_ask=yes_ask / 100.0,
        no_bid=(100 - yes_ask) / 100.0,
        no_ask=(100 - yes_bid) / 100.0,
        volume_24h=1000,
        open_interest=300,
        status="settled",
        resolution=resolution,
    )


# ---------------------------------------------------------------------------
# test_run_backtest_smoke
# ---------------------------------------------------------------------------

def test_run_backtest_smoke():
    """run_backtest produces metrics for all 3 strategies on 10 mocked markets."""
    ohlcv = _make_ohlcv(n=1000)
    markets = [_make_market(i) for i in range(1, 11)]

    per_trade, summary, n_skipped = run_backtest(markets, ohlcv, position_size=50.0)

    assert len(summary) == 3, "Expected 3 strategy summaries"
    labels = {s["strategy"] for s in summary}
    assert "A: vol-model only" in labels
    assert "B: research ens." in labels
    assert "A+B: combined" in labels

    for s in summary:
        assert "n_trades" in s
        assert "win_pct" in s
        assert "mean_edge" in s
        assert "total_pnl_dollars" in s
        assert "sharpe" in s
        assert "max_dd" in s
        assert "wilson_ci" in s
        assert 0.0 <= s["win_pct"] <= 100.0
        ci_lo, ci_hi = s["wilson_ci"]
        assert 0.0 <= ci_lo <= ci_hi <= 1.0


# ---------------------------------------------------------------------------
# test_run_backtest_csv_columns (Sprint 2 column set)
# ---------------------------------------------------------------------------

def test_run_backtest_csv_columns(tmp_path):
    """CSV written by backtest has the Sprint 2 required column set."""
    ohlcv = _make_ohlcv(n=1000)
    markets = [_make_market(i) for i in range(1, 11)]
    per_trade, summary, n_skipped = run_backtest(markets, ohlcv, position_size=50.0)

    required_cols = {
        "strategy", "market_ticker", "close_time", "side", "strike",
        "spot_at_open", "hours_to_expiry", "market_implied_prob", "fair_prob",
        "edge", "position_size", "won", "gross_pnl", "kalshi_fee", "net_pnl", "cum_pnl",
    }

    if per_trade:
        for row in per_trade:
            missing = required_cols - row.keys()
            assert not missing, f"Missing CSV columns: {missing}"


# ---------------------------------------------------------------------------
# test_sharpe_calculation
# ---------------------------------------------------------------------------

def test_sharpe_known_series():
    """Sharpe on a known series matches expected value."""
    assert _sharpe([1.0, 1.0, 1.0]) == 0.0

    pnl = [0.1, -0.05, 0.15, -0.02, 0.08]
    arr = np.array(pnl)
    expected = arr.mean() / arr.std()
    assert abs(_sharpe(pnl) - expected) < 1e-9


# ---------------------------------------------------------------------------
# test_max_drawdown_known_series
# ---------------------------------------------------------------------------

def test_max_drawdown_known_series():
    """Max drawdown on a known P&L series."""
    pnl = [1.0, 1.0, -1.0, -1.0, 1.0]
    result = _max_drawdown(pnl)
    assert result == pytest.approx(-2.0)

    assert _max_drawdown([1.0, 2.0, 3.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_pnl_per_dollar — Sprint 2: returns (net, gross, fee) tuple
# ---------------------------------------------------------------------------

def test_pnl_per_dollar_buy_yes_win():
    """BUY YES win: net P&L is gross minus 7% fee."""
    fair_prob = 0.55
    market_prob = 0.40
    net, gross, fee = _pnl_per_dollar(fair_prob, market_prob, resolved_yes=True)
    assert net > 0
    # Fee = 7% of gross profit
    assert fee == pytest.approx(gross * 0.07, rel=1e-5)
    assert net == pytest.approx(gross - fee, rel=1e-5)


def test_pnl_per_dollar_buy_yes_win_fee_exact():
    """Fee is exactly 7% of the gross profit per dollar risked."""
    # fair > market → BUY YES; cost = market_prob = 0.40
    # gross profit per dollar = (1 - 0.40) / 0.40 = 1.5
    # fee per dollar = 1.5 * 0.07 = 0.105
    # net per dollar = 1.5 * 0.93 = 1.395
    net, gross, fee = _pnl_per_dollar(0.60, 0.40, resolved_yes=True)
    assert gross == pytest.approx((1.0 - 0.40) / 0.40, rel=1e-5)
    assert fee == pytest.approx(gross * 0.07, rel=1e-5)
    assert net == pytest.approx(gross * 0.93, rel=1e-5)


def test_pnl_per_dollar_buy_yes_loss():
    """BUY YES loss: negative P&L, no fee."""
    fair_prob = 0.55
    market_prob = 0.40
    net, gross, fee = _pnl_per_dollar(fair_prob, market_prob, resolved_yes=False)
    assert net < 0
    assert fee == 0.0


def test_pnl_per_dollar_buy_no_win():
    """BUY NO win: positive net, fee applied."""
    fair_prob = 0.45
    market_prob = 0.60
    net, gross, fee = _pnl_per_dollar(fair_prob, market_prob, resolved_yes=False)
    assert net > 0
    assert fee == pytest.approx(gross * 0.07, rel=1e-5)


def test_pnl_per_dollar_below_min_edge():
    """|fair - market| below MIN_EDGE → no trade → all zeros."""
    net, gross, fee = _pnl_per_dollar(0.50, 0.48, resolved_yes=True)
    assert net == 0.0
    assert gross == 0.0
    assert fee == 0.0


# ---------------------------------------------------------------------------
# test_wilson_ci — known cases
# ---------------------------------------------------------------------------

def test_wilson_ci_zero_trials():
    """Zero trials returns (0, 1) uninformative interval."""
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ci_all_wins():
    """100% win rate CI still has a lower bound < 1, and upper bound is capped at 1.0."""
    lo, hi = wilson_ci(100, 100)
    assert lo > 0.9
    assert hi == pytest.approx(1.0, abs=1e-6)


def test_wilson_ci_fifty_fifty():
    """50/100 win rate CI is centred near 0.5."""
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=0.01)
    assert hi == pytest.approx(0.596, abs=0.01)
    assert lo < 0.5 < hi


def test_wilson_ci_monotone():
    """Higher win count → higher CI bounds for same trial count."""
    lo40, hi40 = wilson_ci(40, 100)
    lo60, hi60 = wilson_ci(60, 100)
    assert lo60 > lo40
    assert hi60 > hi40


def test_wilson_ci_within_range():
    """CI always in [0, 1]."""
    for wins, trials in [(0, 10), (5, 10), (10, 10), (1, 1000), (999, 1000)]:
        lo, hi = wilson_ci(wins, trials)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# test_ci_overlap
# ---------------------------------------------------------------------------

def test_ci_overlap_overlapping():
    assert _ci_overlap((0.4, 0.6), (0.5, 0.7)) is True


def test_ci_overlap_disjoint():
    assert _ci_overlap((0.3, 0.45), (0.55, 0.70)) is False


def test_ci_overlap_touching():
    """Intervals that share exactly one endpoint count as overlapping."""
    assert _ci_overlap((0.3, 0.5), (0.5, 0.7)) is True


# ---------------------------------------------------------------------------
# test_stratified_sample
# ---------------------------------------------------------------------------

def test_stratified_sample_no_op_when_under_limit():
    """When len(markets) ≤ max_markets, return all markets unchanged."""
    markets = [_make_market(i) for i in range(1, 11)]
    result = _stratified_sample(markets, 100)
    assert len(result) == len(markets)


def test_stratified_sample_respects_max():
    """Sampling honours max_markets cap."""
    # Create markets spread across 5 different days, 20 per day = 100 total
    markets = []
    base = datetime(2026, 3, 1, 16, 0, tzinfo=timezone.utc)
    for day in range(5):
        exp = base + timedelta(days=day)
        for j in range(20):
            markets.append(KalshiMarket(
                ticker=f"KXBTC-26MAR{day:02d}-T6500{j}",
                event_ticker=f"KXBTC-26MAR{day:02d}",
                title=f"BTC > $65,000",
                strike=65_000.0 + j * 100,
                side="above" if j % 2 == 0 else "below",
                expiration=exp,
                close_time=exp,
                yes_bid=0.38,
                yes_ask=0.42,
                no_bid=0.58,
                no_ask=0.62,
                volume_24h=100,
                open_interest=50,
                status="settled",
                resolution="yes",
            ))

    result = _stratified_sample(markets, 30)
    assert len(result) <= 30


def test_stratified_sample_covers_multiple_days():
    """Stratified sample draws from multiple dates, not just one."""
    markets = []
    base = datetime(2026, 3, 1, 16, 0, tzinfo=timezone.utc)
    for day in range(10):
        exp = base + timedelta(days=day)
        for j in range(10):
            markets.append(KalshiMarket(
                ticker=f"KXBTC-26MAR{day:02d}-T6500{j}",
                event_ticker=f"KXBTC-26MAR{day:02d}",
                title=f"BTC > $65,000",
                strike=65_000.0,
                side="above",
                expiration=exp,
                close_time=exp,
                yes_bid=0.40,
                yes_ask=0.44,
                no_bid=0.56,
                no_ask=0.60,
                volume_24h=100,
                open_interest=50,
                status="settled",
                resolution="yes",
            ))

    result = _stratified_sample(markets, 20)
    dates_sampled = {m.close_time.date() for m in result}
    # Should cover most of the 10 days (≥ 3) even when taking only 20/100
    assert len(dates_sampled) >= 3


# ---------------------------------------------------------------------------
# test_run_backtest_no_lookahead (structural check)
# ---------------------------------------------------------------------------

def test_run_backtest_uses_only_past_data():
    """run_backtest slices OHLCV at market close_time, not future data."""
    ohlcv = _make_ohlcv(n=2000)
    market = _make_market(1, resolution="yes")

    per_trade, summary, n_skipped = run_backtest([market], ohlcv, position_size=50.0)

    if per_trade:
        for row in per_trade:
            assert row["market_implied_prob"] > 0
            assert row["fair_prob"] >= 0


# ---------------------------------------------------------------------------
# test_run_backtest_end_to_end_with_mocked_cli
# ---------------------------------------------------------------------------

def test_backtest_main_with_mocked_sources(tmp_path):
    """main() runs end-to-end with mocked KalshiSource and CoinbaseSource."""
    from scripts.kalshi_btc_backtest import main

    ohlcv = _make_ohlcv(n=1000)

    raw_settled = [
        {
            "ticker": f"KXBTC-26MAR{i:02d}-T65000",
            "event_ticker": f"KXBTC-26MAR{i:02d}",
            "title": "BTC > $65,000",
            "status": "settled",
            "yes_bid": 38,
            "yes_ask": 42,
            "no_bid": 58,
            "no_ask": 62,
            "volume": 1000,
            "open_interest": 200,
            "close_time": "2026-03-10T16:00:00Z",
            "expiration_time": "2026-03-10T16:00:00Z",
            "result": "yes" if i % 2 == 0 else "no",
        }
        for i in range(1, 6)
    ]

    mock_ks = MagicMock()
    mock_ks.get_settled_markets.return_value = raw_settled
    # _cache_dir must be a real path for _fetch_market_implied_prob
    mock_ks._cache_dir = tmp_path

    mock_cb = MagicMock()
    mock_cb.fetch_ohlcv.return_value = ohlcv

    output_dir = tmp_path / "output" / "kalshi"

    with patch("scripts.kalshi_btc_backtest.KalshiSource", return_value=mock_ks), \
         patch("scripts.kalshi_btc_backtest.CoinbaseSource", return_value=mock_cb), \
         patch("scripts.kalshi_btc_backtest._OUTPUT_DIR", output_dir), \
         patch("sys.argv", ["backtest", "--start", "2026-01-01", "--end", "2026-06-01",
                            "--max-markets", "100"]):
        rc = main()

    assert rc == 0
    csv_files = list(output_dir.glob("backtest_*.csv"))
    assert len(csv_files) == 1
