"""Unit tests for domain.stock_scanner.compare.

All tests are self-contained — no live data sources.  Network calls are
patched where compare_routes() is exercised.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from data.sources.base import RawContract
from domain.stock_scanner.compare import (
    OptionRoute,
    StockRoute,
    TradeComparison,
    _build_option_route,
    _build_stock_route,
    _decide_verdict,
    _project_option_value,
    _select_candidate_contracts,
    compare_routes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPIRY_30 = date(2026, 7, 14)   # 30 DTE from notional "today"
_EXPIRY_45 = date(2026, 7, 29)   # 45 DTE


def _make_contract(
    strike: float = 100.0,
    option_type: str = 'call',
    bid: float = 2.80,
    ask: float = 3.20,
    iv: float = 0.28,
    dte: int = 35,
    spot_price: float = 100.0,
    expiration: date = _EXPIRY_30,
    ticker: str = 'TEST',
) -> RawContract:
    return RawContract(
        ticker=ticker,
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        volume=500,
        open_interest=1000,
        implied_volatility=iv,
        dte=dte,
        spot_price=spot_price,
    )


def _make_stock_route(
    target_pnl: float = 300.0,
    stop_pnl: float = -150.0,
    cost: float = 3000.0,
    target_pct: float = 0.10,
    rr_ratio: float = 2.0,
) -> StockRoute:
    return StockRoute(
        ticker='TEST',
        shares=30,
        entry_price=100.0,
        cost=cost,
        target_pnl=target_pnl,
        stop_pnl=stop_pnl,
        target_pct=target_pct,
        stop_pct=-0.05,
        rr_ratio=rr_ratio,
    )


def _make_option_route(
    target_pnl: float = 400.0,
    stop_pnl: float = -200.0,
    cost: float = 300.0,
    iv: float = 0.28,
    rr_ratio: float = 2.0,
    leverage: float = 5.0,
) -> OptionRoute:
    return OptionRoute(
        ticker='TEST',
        option_type='call',
        strike=100.0,
        expiration='2026-07-14',
        dte_at_entry=35,
        entry_mid=3.0,
        cost=cost,
        iv_at_entry=iv,
        delta=0.52,
        bid=2.80,
        ask=3.20,
        projected_target_value=7.0,
        projected_stop_value=1.0,
        target_pnl=target_pnl,
        stop_pnl=stop_pnl,
        breakeven_spot=103.0,
        rr_ratio=rr_ratio,
        leverage=leverage,
        notes='',
    )


# ---------------------------------------------------------------------------
# _project_option_value
# ---------------------------------------------------------------------------

class TestProjectOptionValue:

    def test_atm_call_increases_with_higher_spot(self):
        """A call should be worth more when the future spot is higher."""
        low_val  = _project_option_value(100.0, 100.0, 30, 0.30, 'call')
        high_val = _project_option_value(100.0, 110.0, 30, 0.30, 'call')
        assert high_val > low_val, (
            f"Expected call to increase with spot: {low_val:.4f} -> {high_val:.4f}"
        )

    def test_atm_put_increases_with_lower_spot(self):
        """A put should be worth more when the future spot is lower."""
        high_val = _project_option_value(100.0, 100.0, 30, 0.30, 'put')
        low_val  = _project_option_value(100.0,  90.0, 30, 0.30, 'put')
        assert low_val > high_val, (
            f"Expected put to increase when spot falls: {high_val:.4f} -> {low_val:.4f}"
        )

    def test_otm_call_decays_with_less_dte(self):
        """Same OTM spot, less DTE => lower option value (theta decay)."""
        val_30dte = _project_option_value(105.0, 100.0, 30, 0.30, 'call')
        val_5dte  = _project_option_value(105.0, 100.0,  5, 0.30, 'call')
        assert val_5dte < val_30dte, (
            f"Expected OTM call to decay: 30dte={val_30dte:.4f}, 5dte={val_5dte:.4f}"
        )

    def test_zero_dte_returns_intrinsic_call(self):
        """At expiration, call value = max(spot - strike, 0)."""
        val = _project_option_value(100.0, 110.0, 0, 0.28, 'call')
        assert abs(val - 10.0) < 0.01, f"Expected ~10.0, got {val}"

    def test_zero_dte_otm_call_is_zero(self):
        """OTM call at expiration is worth nothing."""
        val = _project_option_value(110.0, 100.0, 0, 0.28, 'call')
        assert val == 0.0

    def test_zero_dte_returns_intrinsic_put(self):
        """At expiration, put value = max(strike - spot, 0)."""
        val = _project_option_value(100.0, 90.0, 0, 0.28, 'put')
        assert abs(val - 10.0) < 0.01, f"Expected ~10.0, got {val}"

    def test_returns_nonnegative(self):
        """Option value is always >= 0."""
        for spot in [80.0, 100.0, 120.0]:
            val = _project_option_value(100.0, spot, 30, 0.28, 'call')
            assert val >= 0.0


# ---------------------------------------------------------------------------
# _select_candidate_contracts
# ---------------------------------------------------------------------------

class TestSelectCandidateContracts:

    def _chain(self) -> list[RawContract]:
        """Build a small synthetic chain with calls and puts at various strikes."""
        spot = 100.0
        contracts = []
        # Calls at ATM, +5%, +10%, +15%
        for strike, dte in [(100.0, 35), (105.0, 35), (110.0, 35), (115.0, 35)]:
            contracts.append(_make_contract(
                strike=strike, option_type='call', bid=3.0, ask=3.40,
                iv=0.28, dte=dte, spot_price=spot,
            ))
        # Puts at ATM, -5%, -10%
        for strike in [100.0, 95.0, 90.0]:
            contracts.append(_make_contract(
                strike=strike, option_type='put', bid=2.80, ask=3.20,
                iv=0.28, dte=35, spot_price=spot,
            ))
        return contracts

    def test_bullish_returns_only_calls(self):
        chain = self._chain()
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bullish',
            target=110.0, hold_days=28, account_balance=10000.0,
        )
        assert all(c.option_type == 'call' for c in result), \
            "Bullish filter should return only calls"

    def test_bearish_returns_only_puts(self):
        chain = self._chain()
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bearish',
            target=90.0, hold_days=28, account_balance=10000.0,
        )
        assert all(c.option_type == 'put' for c in result), \
            "Bearish filter should return only puts"

    def test_excludes_contracts_above_budget_cap(self):
        """Contracts where mid > 30% of account_balance/100 are excluded."""
        # Contract mid = 3.10, account = 500 → max_mid = 500*0.30/100 = 1.50
        # 3.10 > 1.50, so this should be excluded
        chain = [_make_contract(strike=100.0, option_type='call',
                                bid=3.0, ask=3.20, dte=35, spot_price=100.0)]
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bullish',
            target=110.0, hold_days=28, account_balance=500.0,
        )
        assert result == [], \
            "Expensive contract should be excluded when account is small"

    def test_includes_contracts_within_budget(self):
        """Contracts within budget cap are included."""
        # mid=3.10, account=5000 → max_mid = 5000*0.30/100 = 15.0 → ok
        chain = [_make_contract(strike=100.0, option_type='call',
                                bid=3.0, ask=3.20, dte=35, spot_price=100.0)]
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bullish',
            target=110.0, hold_days=28, account_balance=5000.0,
        )
        assert len(result) == 1

    def test_returns_atm_through_otm_for_bullish(self):
        """Bullish: only strikes from ATM up to +10% are included."""
        chain = self._chain()  # includes +15% strike (115) which should be excluded
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bullish',
            target=110.0, hold_days=28, account_balance=10000.0,
        )
        strikes = [c.strike for c in result]
        assert 100.0 in strikes, "ATM strike should be included"
        assert 110.0 in strikes, "10% OTM strike should be included"
        assert 115.0 not in strikes, "+15% strike should be excluded"

    def test_excludes_low_mid_lottery_tickets(self):
        """Contracts with mid < 0.20 are excluded as worthless lotto tickets."""
        chain = [_make_contract(strike=100.0, option_type='call',
                                bid=0.05, ask=0.10, dte=35, spot_price=100.0)]
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bullish',
            target=110.0, hold_days=28, account_balance=10000.0,
        )
        assert result == [], "Sub-$0.20 mid should be excluded"

    def test_sorted_closest_atm_first(self):
        """Results sorted by distance from ATM, closest first."""
        chain = self._chain()
        result = _select_candidate_contracts(
            chain, spot=100.0, direction='bullish',
            target=110.0, hold_days=28, account_balance=10000.0,
        )
        if len(result) >= 2:
            for a, b in zip(result, result[1:]):
                dist_a = abs(a.strike - 100.0)
                dist_b = abs(b.strike - 100.0)
                assert dist_a <= dist_b


# ---------------------------------------------------------------------------
# _build_option_route
# ---------------------------------------------------------------------------

class TestBuildOptionRoute:

    def test_breakeven_call_equals_strike_plus_mid(self):
        """For a call, breakeven at expiry = strike + entry_mid."""
        contract = _make_contract(strike=100.0, option_type='call',
                                  bid=2.80, ask=3.20, dte=35, spot_price=100.0)
        route = _build_option_route(
            contract=contract, spot=100.0, target=110.0, stop=95.0,
            hold_days=28, risk_free_rate=0.053, stock_target_pct=0.10,
        )
        expected = 100.0 + (2.80 + 3.20) / 2  # strike + mid
        assert abs(route.breakeven_spot - expected) < 0.01, \
            f"Expected breakeven {expected:.2f}, got {route.breakeven_spot:.2f}"

    def test_breakeven_put_equals_strike_minus_mid(self):
        """For a put, breakeven at expiry = strike - entry_mid."""
        contract = _make_contract(strike=100.0, option_type='put',
                                  bid=2.80, ask=3.20, dte=35, spot_price=100.0)
        route = _build_option_route(
            contract=contract, spot=100.0, target=90.0, stop=105.0,
            hold_days=28, risk_free_rate=0.053, stock_target_pct=-0.10,
        )
        expected = 100.0 - (2.80 + 3.20) / 2
        assert abs(route.breakeven_spot - expected) < 0.01

    def test_call_target_pnl_positive_when_spot_rises(self):
        """A call bought ATM should have positive target_pnl when spot reaches target."""
        contract = _make_contract(strike=100.0, option_type='call',
                                  bid=2.80, ask=3.20, iv=0.28, dte=35, spot_price=100.0)
        route = _build_option_route(
            contract=contract, spot=100.0, target=115.0, stop=93.0,
            hold_days=14, risk_free_rate=0.053, stock_target_pct=0.15,
        )
        assert route.target_pnl > 0, \
            f"ATM call should profit when spot rises 15%: pnl={route.target_pnl}"

    def test_stop_pnl_negative_for_long_call(self):
        """A long call loses value when spot drops to stop."""
        contract = _make_contract(strike=100.0, option_type='call',
                                  bid=2.80, ask=3.20, iv=0.28, dte=35, spot_price=100.0)
        route = _build_option_route(
            contract=contract, spot=100.0, target=115.0, stop=90.0,
            hold_days=14, risk_free_rate=0.053, stock_target_pct=0.15,
        )
        assert route.stop_pnl < 0, \
            f"ATM call should lose when spot drops to stop: pnl={route.stop_pnl}"

    def test_cost_equals_mid_times_100(self):
        """Contract cost = entry_mid * 100."""
        contract = _make_contract(bid=3.0, ask=3.40, dte=35, spot_price=100.0)
        route = _build_option_route(
            contract=contract, spot=100.0, target=110.0, stop=95.0,
            hold_days=28, risk_free_rate=0.053, stock_target_pct=0.10,
        )
        assert abs(route.cost - route.entry_mid * 100) < 0.01


# ---------------------------------------------------------------------------
# _decide_verdict
# ---------------------------------------------------------------------------

class TestDecideVerdict:

    def test_stock_preferred_when_no_options(self):
        stock = _make_stock_route()
        verdict, reason = _decide_verdict(stock, [])
        assert verdict == 'STOCK_PREFERRED'
        assert 'No affordable' in reason

    def test_stock_preferred_when_iv_high(self):
        """IV > 50% → STOCK_PREFERRED (vol crush risk)."""
        stock = _make_stock_route(target_pnl=300, stop_pnl=-150, rr_ratio=2.0)
        opt = _make_option_route(iv=0.65, rr_ratio=2.5, target_pnl=400, stop_pnl=-150)
        verdict, reason = _decide_verdict(stock, [opt])
        assert verdict == 'STOCK_PREFERRED'
        assert 'elevated' in reason.lower() or 'IV' in reason

    def test_stock_preferred_when_option_rr_worse(self):
        """Option R:R < stock R:R → STOCK_PREFERRED."""
        stock = _make_stock_route(rr_ratio=2.0)
        opt = _make_option_route(iv=0.28, rr_ratio=1.0, target_pnl=200, stop_pnl=-200)
        verdict, reason = _decide_verdict(stock, [opt])
        assert verdict == 'STOCK_PREFERRED'

    def test_option_preferred_when_high_leverage_low_cost(self):
        """Option delivers >2.5x stock pnl at <50% of stock cost → OPTION_PREFERRED."""
        stock = _make_stock_route(target_pnl=200.0, stop_pnl=-100.0, cost=2000.0)
        # target_pnl > 200*2.5=500 AND cost <= 2000*0.5=1000
        opt = _make_option_route(target_pnl=600.0, stop_pnl=-200.0, cost=900.0,
                                 iv=0.28, rr_ratio=3.0, leverage=10.0)
        verdict, reason = _decide_verdict(stock, [opt])
        assert verdict == 'OPTION_PREFERRED'
        assert 'leverage' in reason.lower() or 'x' in reason

    def test_option_preferred_when_rr_materially_better(self):
        """Option R:R >= 1.5x stock R:R → OPTION_PREFERRED."""
        stock = _make_stock_route(rr_ratio=2.0, cost=3000.0)
        # rr_ratio >= 2.0*1.5=3.0, iv low, not captured by leverage rule
        opt = _make_option_route(iv=0.28, rr_ratio=4.0, target_pnl=300,
                                 stop_pnl=-100, cost=500.0, leverage=2.0)
        # For the leverage rule: 300 < 200*2.5=500, so leverage rule doesn't fire
        # IV rule doesn't fire (0.28 < 0.50)
        # Target% rule: stock.target_pct=0.10 is not <0.05, so doesn't fire
        # RR worse rule: 4.0 is not < 2.0, so doesn't fire
        # RR materially better: 4.0 >= 2.0*1.5=3.0 → OPTION_PREFERRED
        verdict, reason = _decide_verdict(stock, [opt])
        assert verdict == 'OPTION_PREFERRED'

    def test_close_call_when_similar_rr(self):
        """Similar R:R for both routes → CLOSE_CALL."""
        # stock R:R = 2.0; option R:R = 2.4 (< 2.0*1.5=3.0, >= 2.0)
        stock = _make_stock_route(rr_ratio=2.0, cost=3000.0, target_pnl=200, target_pct=0.10)
        opt = _make_option_route(iv=0.28, rr_ratio=2.4, target_pnl=250,
                                 stop_pnl=-100, cost=500.0, leverage=2.0)
        # leverage: 250 < 200*2.5=500 → no
        # iv: 0.28 < 0.50 → no
        # target_pct: 0.10 >= 0.05 → no
        # rr worse: 2.4 >= 2.0 → no
        # rr materially: 2.4 < 3.0 → no
        # → CLOSE_CALL
        verdict, reason = _decide_verdict(stock, [opt])
        assert verdict == 'CLOSE_CALL'
        assert 'CLOSE_CALL' == verdict or 'viable' in reason.lower()

    def test_stock_preferred_small_target_high_cost(self):
        """<5% target move and option costs >40% of stock cost → STOCK_PREFERRED."""
        # stock target_pct = 0.04 (<0.05), stock cost = 1000
        stock = _make_stock_route(target_pct=0.04, cost=1000.0,
                                  target_pnl=40, stop_pnl=-30, rr_ratio=1.33)
        # option cost = 450 > 1000*0.40=400
        opt = _make_option_route(iv=0.28, rr_ratio=1.4, target_pnl=50,
                                 stop_pnl=-30, cost=450.0, leverage=2.0)
        verdict, reason = _decide_verdict(stock, [opt])
        assert verdict == 'STOCK_PREFERRED'


# ---------------------------------------------------------------------------
# compare_routes — full integration (mocked sources)
# ---------------------------------------------------------------------------

class TestCompareRoutesFull:

    def _make_chain(self, spot: float = 100.0) -> list[RawContract]:
        """Synthetic chain: 3 ATM-to-OTM calls with 35 DTE."""
        return [
            _make_contract(strike=100.0, option_type='call', bid=3.0, ask=3.40,
                           iv=0.28, dte=35, spot_price=spot),
            _make_contract(strike=105.0, option_type='call', bid=1.80, ask=2.10,
                           iv=0.29, dte=35, spot_price=spot),
            _make_contract(strike=110.0, option_type='call', bid=0.90, ask=1.10,
                           iv=0.30, dte=35, spot_price=spot),
        ]

    def _fallback(self, chain, spot):
        """Return a side_effect function for fetch_with_fallback."""
        def _fn(sources, method, ticker, *args, **kwargs):
            if method == 'fetch_spot':
                return spot
            if method == 'fetch_option_chain':
                return chain
            raise ValueError(f"Unexpected method: {method}")
        return _fn

    def test_returns_trade_comparison_object(self):
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert isinstance(result, TradeComparison)

    def test_stock_route_populated(self):
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert isinstance(result.stock_route, StockRoute)
        assert result.stock_route.shares >= 1
        assert result.stock_route.cost > 0

    def test_option_routes_capped_at_3(self):
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert len(result.option_routes) <= 3

    def test_option_routes_only_calls_for_bullish(self):
        chain = self._make_chain(100.0) + [
            _make_contract(strike=100.0, option_type='put', bid=2.80, ask=3.20,
                           iv=0.28, dte=35, spot_price=100.0)
        ]
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        for opt in result.option_routes:
            assert opt.option_type == 'call'

    def test_option_routes_only_puts_for_bearish(self):
        chain = [
            _make_contract(strike=100.0, option_type='put', bid=3.0, ask=3.40,
                           iv=0.28, dte=35, spot_price=100.0),
            _make_contract(strike=95.0, option_type='put', bid=1.5, ask=1.80,
                           iv=0.29, dte=35, spot_price=100.0),
            _make_contract(strike=100.0, option_type='call', bid=3.0, ask=3.40,
                           iv=0.28, dte=35, spot_price=100.0),
        ]
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bearish', target=90.0, stop=105.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        for opt in result.option_routes:
            assert opt.option_type == 'put'

    def test_empty_chain_gives_stock_preferred(self):
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback([], 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert result.verdict == 'STOCK_PREFERRED'
        assert result.option_routes == []

    def test_verdict_is_valid_literal(self):
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert result.verdict in ('STOCK_PREFERRED', 'OPTION_PREFERRED', 'CLOSE_CALL')

    def test_verdict_reason_is_nonempty_string(self):
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert isinstance(result.verdict_reason, str)
        assert len(result.verdict_reason) > 0

    def test_spot_falls_back_to_chain_when_fetch_spot_fails(self):
        """If fetch_spot raises DataFetchError, spot is read from chain[0].spot_price."""
        from data.fallback import DataFetchError

        def _fn(sources, method, ticker, *args, **kwargs):
            if method == 'fetch_spot':
                raise DataFetchError('no spot')
            return self._make_chain(100.0)

        with patch('domain.stock_scanner.compare.fetch_with_fallback', side_effect=_fn):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert result.spot == 100.0

    def test_option_routes_ranked_by_rr_descending(self):
        """option_routes should be sorted highest R:R first."""
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='TEST', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        rr_values = [o.rr_ratio for o in result.option_routes]
        assert rr_values == sorted(rr_values, reverse=True)

    def test_ticker_preserved_in_output(self):
        chain = self._make_chain(100.0)
        with patch('domain.stock_scanner.compare.fetch_with_fallback',
                   side_effect=self._fallback(chain, 100.0)):
            result = compare_routes(
                ticker='KLAC', direction='bullish', target=110.0, stop=95.0,
                hold_days=28, account_balance=5000.0, sources=[MagicMock()],
            )
        assert result.ticker == 'KLAC'
        assert result.stock_route.ticker == 'KLAC'
