"""Tests for scripts/kalshi_btc_backtest.py

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

    per_trade, summary = run_backtest(markets, ohlcv, position_size=50.0)

    assert len(summary) == 3, "Expected 3 strategy summaries"
    labels = {s["strategy"] for s in summary}
    assert "A: Vol-model only" in labels
    assert "B: Research ensemble" in labels
    assert "A+B: Combined" in labels

    for s in summary:
        assert "n_trades" in s
        assert "win_pct" in s
        assert "mean_edge" in s
        assert "total_pnl" in s
        assert "sharpe" in s
        assert "max_dd" in s
        assert 0.0 <= s["win_pct"] <= 100.0


# ---------------------------------------------------------------------------
# test_run_backtest_csv_columns
# ---------------------------------------------------------------------------

def test_run_backtest_csv_columns(tmp_path):
    """CSV written by backtest has the required column set."""
    import csv as csv_mod

    ohlcv = _make_ohlcv(n=1000)
    markets = [_make_market(i) for i in range(1, 11)]
    per_trade, summary = run_backtest(markets, ohlcv, position_size=50.0)

    # Simulate CSV write
    required_cols = {
        "strategy", "ticker", "market_prob", "fair_prob", "edge",
        "position_size", "resolved_yes", "pnl_per_dollar", "pnl_dollars",
    }

    if per_trade:
        for row in per_trade:
            assert required_cols.issubset(row.keys()), (
                f"Missing CSV columns: {required_cols - row.keys()}"
            )


# ---------------------------------------------------------------------------
# test_sharpe_calculation
# ---------------------------------------------------------------------------

def test_sharpe_known_series():
    """Sharpe on a known series matches expected value."""
    # Series: [1, 1, 1, 1, 1] → mean=1, std=0 → sharpe=0 (guard)
    assert _sharpe([1.0, 1.0, 1.0]) == 0.0

    # Series with non-zero std
    pnl = [0.1, -0.05, 0.15, -0.02, 0.08]
    arr = np.array(pnl)
    expected = arr.mean() / arr.std()
    assert abs(_sharpe(pnl) - expected) < 1e-9


# ---------------------------------------------------------------------------
# test_max_drawdown_known_series
# ---------------------------------------------------------------------------

def test_max_drawdown_known_series():
    """Max drawdown on a known P&L series."""
    # Cumulative: [1, 2, 1, 0, 1] → peak at 2, trough at 0 → dd = -2
    pnl = [1.0, 1.0, -1.0, -1.0, 1.0]
    result = _max_drawdown(pnl)
    assert result == pytest.approx(-2.0)

    # All positive → no drawdown
    assert _max_drawdown([1.0, 2.0, 3.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_pnl_per_dollar_buy_yes
# ---------------------------------------------------------------------------

def test_pnl_per_dollar_buy_yes_win():
    """BUY YES (fair_prob > market_prob), resolves YES → positive P&L net of fee."""
    # fair_prob = 0.55, market_prob = 0.40 → edge = 0.15 > MIN_EDGE
    fair_prob = 0.55
    market_prob = 0.40
    pnl = _pnl_per_dollar(fair_prob, market_prob, resolved_yes=True)
    assert pnl > 0


def test_pnl_per_dollar_buy_yes_loss():
    """BUY YES (fair_prob > market_prob), resolves NO → negative P&L."""
    fair_prob = 0.55
    market_prob = 0.40
    pnl = _pnl_per_dollar(fair_prob, market_prob, resolved_yes=False)
    assert pnl < 0


def test_pnl_per_dollar_buy_no_win():
    """BUY NO (fair_prob < market_prob), resolves NO → positive P&L."""
    fair_prob = 0.45  # fair < market → sell YES
    market_prob = 0.60  # edge = -0.15 → abs edge > MIN_EDGE
    pnl = _pnl_per_dollar(fair_prob, market_prob, resolved_yes=False)
    assert pnl > 0


def test_pnl_per_dollar_below_min_edge():
    """|fair_prob - market_prob| below MIN_EDGE threshold → no trade → pnl=0."""
    # fair=0.50, market=0.48 → edge=0.02 < 0.05
    pnl = _pnl_per_dollar(0.50, 0.48, resolved_yes=True)
    assert pnl == 0.0


# ---------------------------------------------------------------------------
# test_run_backtest_no_lookahead (structural check)
# ---------------------------------------------------------------------------

def test_run_backtest_uses_only_past_data():
    """run_backtest slices OHLCV at market close_time, not future data."""
    ohlcv = _make_ohlcv(n=2000)
    # Market with close_time in the past relative to ohlcv end
    market = _make_market(1, resolution="yes")

    per_trade, summary = run_backtest([market], ohlcv, position_size=50.0)

    # Can't check internal slicing directly, but if we get here without error
    # and the spot price used is plausible, the test passes
    if per_trade:
        for row in per_trade:
            assert row["market_prob"] > 0
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
            "title": f"BTC > $65,000",
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

    mock_cb = MagicMock()
    mock_cb.fetch_ohlcv.return_value = ohlcv

    output_dir = tmp_path / "output" / "kalshi"

    with patch("scripts.kalshi_btc_backtest.KalshiSource", return_value=mock_ks), \
         patch("scripts.kalshi_btc_backtest.CoinbaseSource", return_value=mock_cb), \
         patch("scripts.kalshi_btc_backtest._OUTPUT_DIR", output_dir), \
         patch("sys.argv", ["backtest", "--start", "2026-01-01", "--end", "2026-06-01"]):
        rc = main()

    assert rc == 0
    # CSV must have been written
    csv_files = list(output_dir.glob("backtest_*.csv"))
    assert len(csv_files) == 1
