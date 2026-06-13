from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.prediction.market import (
    KalshiMarket,
    PredictionMarketEdge,
    parse_kalshi_market,
    _parse_strike_and_side,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_market(
    ticker: str = "KXBTC-26JUN1316-T62500",
    yes_bid: int = 8,
    yes_ask: int = 10,
    no_bid: int = 90,
    no_ask: int = 92,
    status: str = "open",
    resolution: str | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": "KXBTC-26JUN1316",
        "title": "BTC > $62,500 at 4:00 PM ET",
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "volume": 1240,
        "open_interest": 500,
        "close_time": "2026-06-13T20:00:00Z",
        "expiration_time": "2026-06-13T20:00:00Z",
        "status": status,
        "result": resolution,
    }


def _make_market(**kwargs) -> KalshiMarket:
    defaults = dict(
        ticker="KXBTC-26JUN1316-T62500",
        event_ticker="KXBTC-26JUN1316",
        title="BTC > $62,500 at 4:00 PM ET",
        strike=62500.0,
        side="above",
        expiration=datetime(2026, 6, 13, 20, 0, tzinfo=timezone.utc),
        close_time=datetime(2026, 6, 13, 20, 0, tzinfo=timezone.utc),
        yes_bid=0.08,
        yes_ask=0.10,
        no_bid=0.90,
        no_ask=0.92,
        volume_24h=1240,
        open_interest=500,
        status="open",
        resolution=None,
    )
    defaults.update(kwargs)
    return KalshiMarket(**defaults)


# ---------------------------------------------------------------------------
# test_parse_kalshi_market_extracts_strike_from_ticker
# ---------------------------------------------------------------------------

def test_parse_kalshi_market_extracts_strike_from_ticker():
    """'KXBTC-26JUN1316-T62500' must yield strike=62500.0."""
    raw = _raw_market(ticker="KXBTC-26JUN1316-T62500")
    market = parse_kalshi_market(raw)
    assert market.strike == 62500.0


def test_parse_kalshi_market_extracts_large_strike():
    """Handles six-digit strikes like T100000."""
    raw = _raw_market(ticker="KXBTC-26DEC2400-T100000")
    raw["close_time"] = "2026-12-24T00:00:00Z"
    raw["expiration_time"] = "2026-12-24T00:00:00Z"
    market = parse_kalshi_market(raw)
    assert market.strike == 100000.0


# ---------------------------------------------------------------------------
# test_parse_kalshi_market_above_side_default
# ---------------------------------------------------------------------------

def test_parse_kalshi_market_above_side_default():
    """T prefix in ticker must yield side='above'."""
    raw = _raw_market(ticker="KXBTC-26JUN1316-T62500")
    market = parse_kalshi_market(raw)
    assert market.side == "above"


def test_parse_kalshi_market_below_side_from_b_prefix():
    """B prefix in ticker must yield side='below'."""
    raw = _raw_market(ticker="KXBTC-26JUN1316-B62500")
    market = parse_kalshi_market(raw)
    assert market.side == "below"
    assert market.strike == 62500.0


# ---------------------------------------------------------------------------
# test_market_implied_prob_mid_of_bid_ask
# ---------------------------------------------------------------------------

def test_market_implied_prob_mid_of_bid_ask():
    """market_implied_prob must be (yes_bid + yes_ask) / 2."""
    market = _make_market(yes_bid=0.08, yes_ask=0.10)
    assert market.market_implied_prob == pytest.approx(0.09)


def test_market_implied_prob_at_extremes():
    """Works correctly at probability extremes."""
    market = _make_market(yes_bid=0.0, yes_ask=0.02)
    assert market.market_implied_prob == pytest.approx(0.01)

    market2 = _make_market(yes_bid=0.97, yes_ask=0.99)
    assert market2.market_implied_prob == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# test_market_spread
# ---------------------------------------------------------------------------

def test_market_spread():
    """spread must be yes_ask - yes_bid."""
    market = _make_market(yes_bid=0.08, yes_ask=0.10)
    assert market.spread == pytest.approx(0.02)


def test_market_spread_zero_for_tight_book():
    """spread is zero when bid == ask."""
    market = _make_market(yes_bid=0.50, yes_ask=0.50)
    assert market.spread == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_kalshi_market_is_frozen
# ---------------------------------------------------------------------------

def test_kalshi_market_is_frozen():
    """Attempting mutation on a frozen KalshiMarket must raise FrozenInstanceError."""
    market = _make_market()
    with pytest.raises(Exception):
        market.strike = 99999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# test_prediction_market_edge_dataclass_fields
# ---------------------------------------------------------------------------

def test_prediction_market_edge_dataclass_fields():
    """PredictionMarketEdge must accept all required fields and expose them."""
    market = _make_market()
    edge = PredictionMarketEdge(
        market=market,
        fair_prob_A=0.14,
        fair_prob_B=0.15,
        fair_prob_AB=0.145,
        canonical_fair_prob=0.145,
        edge=0.055,
        expected_value_per_dollar=0.06,
        estimator_agreement=(3, 4),
        suggested_position_dollars=80.0,
        adjustments_applied=["favorite-longshot:-2%", "TOD:+1%"],
    )

    assert edge.market is market
    assert edge.fair_prob_A == pytest.approx(0.14)
    assert edge.fair_prob_B == pytest.approx(0.15)
    assert edge.fair_prob_AB == pytest.approx(0.145)
    assert edge.canonical_fair_prob == pytest.approx(0.145)
    assert edge.edge == pytest.approx(0.055)
    assert edge.expected_value_per_dollar == pytest.approx(0.06)
    assert edge.estimator_agreement == (3, 4)
    assert edge.suggested_position_dollars == pytest.approx(80.0)
    assert "favorite-longshot:-2%" in edge.adjustments_applied


def test_prediction_market_edge_is_frozen():
    """PredictionMarketEdge is a frozen dataclass."""
    market = _make_market()
    edge = PredictionMarketEdge(
        market=market,
        fair_prob_A=0.14,
        fair_prob_B=0.15,
        fair_prob_AB=0.145,
        canonical_fair_prob=0.145,
        edge=0.055,
        expected_value_per_dollar=0.06,
        estimator_agreement=(3, 4),
        suggested_position_dollars=80.0,
        adjustments_applied=[],
    )
    with pytest.raises(Exception):
        edge.edge = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# test_parse_kalshi_market_normalises_cent_prices
# ---------------------------------------------------------------------------

def test_parse_kalshi_market_normalises_cent_prices():
    """Raw cent prices (0-99) must be normalised to 0-1 probabilities."""
    raw = _raw_market(yes_bid=8, yes_ask=10, no_bid=90, no_ask=92)
    market = parse_kalshi_market(raw)
    assert market.yes_bid == pytest.approx(0.08)
    assert market.yes_ask == pytest.approx(0.10)
    assert market.no_bid == pytest.approx(0.90)
    assert market.no_ask == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# test_parse_kalshi_market_resolution
# ---------------------------------------------------------------------------

def test_parse_kalshi_market_resolution_yes():
    """Settled markets with result='yes' must have resolution='yes'."""
    raw = _raw_market(status="settled", resolution="yes")
    market = parse_kalshi_market(raw)
    assert market.status == "settled"
    assert market.resolution == "yes"


def test_parse_kalshi_market_resolution_no():
    """Settled markets with result='no' must have resolution='no'."""
    raw = _raw_market(status="settled", resolution="no")
    market = parse_kalshi_market(raw)
    assert market.resolution == "no"


def test_parse_kalshi_market_open_has_no_resolution():
    """Open markets must have resolution=None."""
    raw = _raw_market(status="open", resolution=None)
    market = parse_kalshi_market(raw)
    assert market.resolution is None
