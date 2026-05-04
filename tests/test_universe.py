from __future__ import annotations

from data.universe import get_universe, FOCUSED_UNIVERSE


def test_get_universe_returns_list_of_strings():
    universe = get_universe()
    assert isinstance(universe, list)
    assert len(universe) > 10
    assert all(isinstance(t, str) for t in universe)


def test_liquid_etfs_included():
    universe = get_universe()
    for etf in ['SPY', 'QQQ', 'IWM']:
        assert etf in universe


def test_mag7_included():
    universe = get_universe()
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']:
        assert ticker in universe


def test_no_duplicates():
    universe = get_universe()
    assert len(universe) == len(set(universe))


def test_universe_is_sorted():
    universe = get_universe()
    assert universe == sorted(universe)
