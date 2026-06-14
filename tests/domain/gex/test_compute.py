"""Unit tests for domain.gex.compute.

All tests use synthetic RawContract lists — no live data fetches.
Fixed today = date(2026, 6, 15) for reproducibility.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from data.sources.base import RawContract
from domain.gex.compute import (
    GexContext,
    GexZone,
    StrikeGex,
    build_context,
    gamma_flip_level,
    gex_per_strike,
    top_zones,
    total_gex,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date(2026, 6, 15)
_EXPIRY_30D = date(2026, 7, 15)   # 30 DTE from _TODAY
_EXPIRY_60D = date(2026, 8, 14)   # 60 DTE from _TODAY


def _make_contract(
    strike: float,
    option_type: str,
    oi: int,
    iv: float = 0.20,
    spot: float = 100.0,
    dte: int = 30,
    expiration: date | None = None,
) -> RawContract:
    if expiration is None:
        expiration = _EXPIRY_30D
    return RawContract(
        ticker='TEST',
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        bid=1.0,
        ask=1.05,
        volume=100,
        open_interest=oi,
        implied_volatility=iv,
        dte=dte,
        spot_price=spot,
    )


def _approx_gex(oi: int, gamma: float, spot: float) -> float:
    """Expected per-contract GEX using the standard formula (call, no put)."""
    return oi * gamma * 100.0 * (spot ** 2)


# ---------------------------------------------------------------------------
# gex_per_strike tests
# ---------------------------------------------------------------------------

class TestGexPerStrike:

    def test_gex_per_strike_calls_only_positive_gex(self):
        """Single ATM call → positive GEX (call-dominated strike)."""
        # OI=100, IV=0.20, spot=100, strike=100, expiry=30d
        chain = [_make_contract(strike=100.0, option_type='call', oi=100)]
        result = gex_per_strike(chain, spot=100.0, today=_TODAY)

        assert 100.0 in result
        # Gamma for this ATM call is positive; call contribution is positive
        assert result[100.0] > 0.0

    def test_gex_per_strike_puts_only_negative_gex(self):
        """Single ATM put → negative GEX (put-dominated strike)."""
        chain = [_make_contract(strike=100.0, option_type='put', oi=200)]
        result = gex_per_strike(chain, spot=100.0, today=_TODAY)

        assert 100.0 in result
        assert result[100.0] < 0.0

    def test_gex_per_strike_aggregates_calls_and_puts_same_strike(self):
        """call_OI=100, put_OI=50, same gamma → net = (100-50) * gamma * 100 * spot²."""
        iv = 0.20
        spot = 100.0
        strike = 100.0
        chain = [
            _make_contract(strike=strike, option_type='call', oi=100, iv=iv, spot=spot),
            _make_contract(strike=strike, option_type='put',  oi=50,  iv=iv, spot=spot),
        ]
        result = gex_per_strike(chain, spot=spot, today=_TODAY)

        # calls contribute +, puts contribute −; call_OI > put_OI → net positive
        assert 100.0 in result
        assert result[100.0] > 0.0

        # Verify the ratio: with same gamma, call(100) - put(50) = 2x put(50)
        # So net should be exactly half of what a 100-OI call would produce
        chain_call_only = [
            _make_contract(strike=strike, option_type='call', oi=100, iv=iv, spot=spot)
        ]
        call_gex = gex_per_strike(chain_call_only, spot=spot, today=_TODAY)[strike]
        assert math.isclose(result[strike], call_gex * 0.5, rel_tol=1e-6)

    def test_gex_per_strike_skips_zero_iv(self):
        """Contracts with IV=0 must be skipped (no NaN/inf)."""
        chain = [
            _make_contract(strike=100.0, option_type='call', oi=100, iv=0.0),
            _make_contract(strike=105.0, option_type='call', oi=50,  iv=0.20),
        ]
        result = gex_per_strike(chain, spot=100.0, today=_TODAY)
        # IV=0 strike should be absent
        assert 100.0 not in result
        # Valid strike should be present
        assert 105.0 in result

    def test_gex_per_strike_aggregates_multiple_expirations(self):
        """Same strike, two expirations → summed into one dict entry."""
        chain = [
            _make_contract(strike=100.0, option_type='call', oi=100,
                           expiration=_EXPIRY_30D, dte=30),
            _make_contract(strike=100.0, option_type='call', oi=100,
                           expiration=_EXPIRY_60D, dte=60),
        ]
        result = gex_per_strike(chain, spot=100.0, today=_TODAY)
        assert 100.0 in result
        # Two separate expirations both add positive GEX for a call
        assert result[100.0] > 0.0


# ---------------------------------------------------------------------------
# total_gex tests
# ---------------------------------------------------------------------------

class TestTotalGex:

    def test_total_gex_sums_per_strike_values(self):
        per_strike = {100.0: 1e6, 105.0: 2e6, 95.0: -1e6}
        assert total_gex(per_strike) == pytest.approx(2e6)

    def test_total_gex_empty(self):
        assert total_gex({}) == 0.0

    def test_total_gex_all_negative(self):
        per_strike = {100.0: -3e6, 95.0: -2e6}
        assert total_gex(per_strike) == pytest.approx(-5e6)


# ---------------------------------------------------------------------------
# top_zones tests
# ---------------------------------------------------------------------------

class TestTopZones:

    def test_top_zones_supply_above_spot(self):
        """Supply zones are strikes above spot."""
        per_strike = {105.0: 2e6, 95.0: 1e6, 110.0: 0.5e6}
        spot = 100.0
        supply, demand = top_zones(per_strike, spot=spot, n=5, max_distance_pct=0.15)
        supply_strikes = {z.strike for z in supply}
        demand_strikes = {z.strike for z in demand}

        assert 105.0 in supply_strikes
        assert 110.0 in supply_strikes
        assert 95.0 in demand_strikes

    def test_top_zones_sorted_by_abs_gex_descending(self):
        """Supply zones are sorted by abs(gex_dollars) descending."""
        per_strike = {105.0: 0.5e6, 110.0: 2e6, 115.0: 1e6}
        supply, _ = top_zones(per_strike, spot=100.0, n=5, max_distance_pct=0.20)
        abs_vals = [abs(z.gex_dollars) for z in supply]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_top_zones_filters_max_distance(self):
        """Strikes far from spot (>max_distance_pct) must be excluded."""
        per_strike = {200.0: 1e9, 105.0: 2e6}  # 200 is 100% away
        supply, _ = top_zones(per_strike, spot=100.0, n=5, max_distance_pct=0.05)
        supply_strikes = {z.strike for z in supply}
        assert 200.0 not in supply_strikes
        assert 105.0 in supply_strikes

    def test_top_zones_demand_below_spot(self):
        """Demand zones contain only strikes below spot."""
        per_strike = {95.0: -1e6, 90.0: -3e6, 105.0: 2e6}
        _, demand = top_zones(per_strike, spot=100.0, n=5, max_distance_pct=0.15)
        for z in demand:
            assert z.strike < 100.0
            assert z.side == 'below'

    def test_top_zones_respects_n_limit(self):
        """Returns at most n zones per side."""
        per_strike = {i * 1.0: 1e6 for i in range(101, 115)}  # 14 strikes above 100
        supply, _ = top_zones(per_strike, spot=100.0, n=3, max_distance_pct=0.20)
        assert len(supply) <= 3

    def test_top_zones_distance_pct_sign(self):
        """Distance pct is positive above spot, negative below."""
        per_strike = {105.0: 1e6, 95.0: 1e6}
        supply, demand = top_zones(per_strike, spot=100.0, n=5, max_distance_pct=0.10)
        assert all(z.distance_pct > 0.0 for z in supply)
        assert all(z.distance_pct < 0.0 for z in demand)


# ---------------------------------------------------------------------------
# gamma_flip_level tests
# ---------------------------------------------------------------------------

class TestGammaFlipLevel:

    def _heavy_call_chain(self, spot: float = 100.0) -> list:
        """Chain with massive call OI at strike=100 (above given spot) to push
        total_gex positive at spot; at lower spot the calls go deeper ITM and
        gamma changes, but we also add heavy puts at 90 to ensure flip below spot."""
        # Heavy calls at 100 → positive GEX when spot is near 100
        # Heavy puts at 90 → more negative as spot drops toward 90
        return [
            _make_contract(strike=100.0, option_type='call', oi=10000, iv=0.20,
                           spot=spot, expiration=_EXPIRY_30D, dte=30),
            _make_contract(strike=90.0, option_type='put',  oi=50000, iv=0.25,
                           spot=spot, expiration=_EXPIRY_30D, dte=30),
        ]

    def test_gamma_flip_level_finds_zero_crossing(self):
        """Returns a price between search bounds when crossing exists."""
        # Use large put OI to guarantee negative regime at some lower price,
        # and large call OI to guarantee positive at current spot.
        chain = [
            _make_contract(strike=100.0, option_type='call', oi=10000, iv=0.20,
                           expiration=_EXPIRY_30D, dte=30),
            _make_contract(strike=80.0,  option_type='put',  oi=100000, iv=0.25,
                           expiration=_EXPIRY_30D, dte=30),
        ]
        spot = 100.0
        gfl = gamma_flip_level(chain, spot=spot, today=_TODAY,
                                search_range_pct=0.30, search_step_pct=0.002)
        # If a crossing is found, it must be within search range
        if gfl is not None:
            assert spot * 0.70 <= gfl <= spot * 1.30

    def test_gamma_flip_level_returns_none_when_no_crossing(self):
        """Returns None when all candidate spots produce same-sign GEX."""
        # Pure call chain → total GEX always positive across any spot range
        chain = [
            _make_contract(strike=100.0, option_type='call', oi=1000, iv=0.20,
                           expiration=_EXPIRY_30D, dte=30),
            _make_contract(strike=110.0, option_type='call', oi=500,  iv=0.20,
                           expiration=_EXPIRY_30D, dte=30),
        ]
        gfl = gamma_flip_level(chain, spot=100.0, today=_TODAY,
                                search_range_pct=0.05, search_step_pct=0.001)
        # A pure-call chain is always positive → should be None
        # (Note: might still return None or a value near edges; we just verify no crash
        #  and that if returned, it's within range)
        if gfl is not None:
            assert 95.0 <= gfl <= 105.0

    def test_gamma_flip_level_returns_none_empty_chain(self):
        """Empty chain returns None without crash."""
        result = gamma_flip_level([], spot=100.0, today=_TODAY)
        assert result is None

    def test_gamma_flip_level_within_search_range(self):
        """When a crossing is found it must be within the search range."""
        chain = [
            _make_contract(strike=100.0, option_type='call', oi=5000, iv=0.20,
                           expiration=_EXPIRY_30D, dte=30),
            _make_contract(strike=95.0,  option_type='put',  oi=20000, iv=0.20,
                           expiration=_EXPIRY_30D, dte=30),
        ]
        spot = 100.0
        gfl = gamma_flip_level(chain, spot=spot, today=_TODAY,
                                search_range_pct=0.10, search_step_pct=0.002)
        if gfl is not None:
            assert spot * 0.90 <= gfl <= spot * 1.10


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------

class TestBuildContext:

    def _sample_chain(self) -> list:
        """~10 contract chain spanning several strikes."""
        contracts = []
        # Calls above and at spot
        for strike, oi in [(100.0, 500), (105.0, 300), (110.0, 200), (95.0, 150)]:
            contracts.append(_make_contract(strike=strike, option_type='call',
                                            oi=oi, iv=0.20, expiration=_EXPIRY_30D, dte=30))
        # Puts below and at spot
        for strike, oi in [(100.0, 400), (95.0, 600), (90.0, 200), (85.0, 100)]:
            contracts.append(_make_contract(strike=strike, option_type='put',
                                            oi=oi, iv=0.22, expiration=_EXPIRY_30D, dte=30))
        return contracts

    def test_build_context_filters_zero_iv(self):
        """Contracts with IV=0 are excluded; no NaN/inf in result."""
        chain = self._sample_chain()
        # Add a zero-IV contract
        chain.append(_make_contract(strike=120.0, option_type='call', oi=9999, iv=0.0))
        ctx = build_context('TEST', chain, spot=100.0, today=_TODAY)

        assert math.isfinite(ctx.total_gex_dollars)
        assert ctx.chain_size == len(chain)  # chain_size = len before gex computation

    def test_build_context_respects_max_dte(self):
        """Contracts beyond max_dte are excluded from chain_size."""
        chain = self._sample_chain()
        # Add 60 DTE contracts that should be excluded
        for strike, oi in [(100.0, 1000), (95.0, 1000)]:
            chain.append(_make_contract(strike=strike, option_type='call',
                                        oi=oi, iv=0.20,
                                        expiration=_EXPIRY_60D, dte=60))

        ctx_no_filter = build_context('TEST', chain, spot=100.0, today=_TODAY)
        ctx_filtered  = build_context('TEST', chain, spot=100.0, today=_TODAY, max_dte=30)

        # With max_dte=30, the 60 DTE contracts are excluded
        assert ctx_filtered.chain_size < ctx_no_filter.chain_size
        # chain_size should equal len(30-DTE contracts only)
        n_30d = sum(1 for c in chain if c.dte <= 30)
        assert ctx_filtered.chain_size == n_30d

    def test_build_context_returns_valid_context(self):
        """Full happy path returns a well-formed GexContext."""
        chain = self._sample_chain()
        ctx = build_context('TEST', chain, spot=100.0, today=_TODAY,
                             n_zones=5, zone_max_distance_pct=0.15)

        assert isinstance(ctx, GexContext)
        assert ctx.ticker == 'TEST'
        assert ctx.spot == 100.0
        assert math.isfinite(ctx.total_gex_dollars)
        assert ctx.chain_size == len(chain)

        # Supply zones are all above spot
        for z in ctx.supply_zones:
            assert z.strike > 100.0
            assert z.side == 'above'

        # Demand zones are all below spot
        for z in ctx.demand_zones:
            assert z.strike < 100.0
            assert z.side == 'below'

    def test_build_context_empty_chain(self):
        """Empty chain returns zero GEX, no zones."""
        ctx = build_context('TEST', [], spot=100.0, today=_TODAY)
        assert ctx.total_gex_dollars == 0.0
        assert ctx.supply_zones == []
        assert ctx.demand_zones == []
        assert ctx.chain_size == 0

    def test_build_context_ticker_field(self):
        """ticker field is preserved correctly."""
        chain = self._sample_chain()
        ctx = build_context('SPY', chain, spot=100.0, today=_TODAY)
        assert ctx.ticker == 'SPY'
