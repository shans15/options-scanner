"""Smoke tests for scripts.trade_compare.

All tests mock the data layer — no live network calls.

Patching strategy:
  Both compare.py and trade_compare.py import fetch_with_fallback at module
  level. We patch each module's local binding so both code paths are covered:
    - 'domain.stock_scanner.compare.fetch_with_fallback'
    - 'scripts.trade_compare.fetch_with_fallback'
"""
from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from data.sources.base import RawContract
from scripts.trade_compare import main, render_comparison
from domain.stock_scanner.compare import (
    TradeComparison,
    StockRoute,
    OptionRoute,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPIRY = date(2026, 7, 14)


def _make_contract(
    strike: float = 100.0,
    option_type: str = 'call',
    bid: float = 2.80,
    ask: float = 3.20,
    iv: float = 0.28,
    dte: int = 35,
    spot_price: float = 100.0,
) -> RawContract:
    return RawContract(
        ticker='TEST',
        expiration=_EXPIRY,
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


def _make_chain(spot: float = 100.0) -> list[RawContract]:
    return [
        _make_contract(strike=100.0, option_type='call', bid=3.0, ask=3.40,
                       iv=0.28, dte=35, spot_price=spot),
        _make_contract(strike=105.0, option_type='call', bid=1.80, ask=2.10,
                       iv=0.29, dte=35, spot_price=spot),
    ]


def _make_put_chain(spot: float = 100.0) -> list[RawContract]:
    return [
        _make_contract(strike=100.0, option_type='put', bid=3.0, ask=3.40,
                       iv=0.28, dte=35, spot_price=spot),
        _make_contract(strike=95.0, option_type='put', bid=1.80, ask=2.10,
                       iv=0.29, dte=35, spot_price=spot),
    ]


def _fallback_factory(chain, spot=100.0):
    """Build a side_effect for fetch_with_fallback."""
    def _fn(sources, method, ticker, *args, **kwargs):
        if method == 'fetch_spot':
            return spot
        if method == 'fetch_option_chain':
            return chain
        raise ValueError(f"Unexpected method: {method}")
    return _fn


@contextmanager
def _mock_all(chain, spot=100.0):
    """Patch both modules' fetch_with_fallback bindings + default_sources."""
    side_effect = _fallback_factory(chain, spot)
    with patch('domain.stock_scanner.compare.fetch_with_fallback', side_effect=side_effect):
        with patch('scripts.trade_compare.fetch_with_fallback', side_effect=side_effect):
            with patch('scripts.trade_compare.default_sources', return_value=[MagicMock()]):
                with patch('domain.stock_scanner.compare.default_sources', return_value=[MagicMock()]):
                    yield


def _capture_stdout(fn, *args, **kwargs):
    """Run fn and return (return_value, captured_stdout_text)."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


# ---------------------------------------------------------------------------
# CLI smoke tests via main()
# ---------------------------------------------------------------------------

class TestCliSmoke:

    def test_cli_runs_with_defaults_for_bullish(self):
        """main(['TEST']) with mocked sources should return exit code 0."""
        chain = _make_chain(100.0)
        with _mock_all(chain, 100.0):
            code, output = _capture_stdout(
                main, ['TEST', '--account', '5000', '--target', '105', '--stop', '97']
            )
        assert code == 0
        assert 'TRADE COMPARISON' in output

    def test_cli_runs_with_bearish_flag(self):
        """main(['TEST', '--direction', 'bearish']) should succeed and show bearish output."""
        chain = _make_put_chain(100.0)
        with _mock_all(chain, 100.0):
            code, output = _capture_stdout(
                main,
                ['TEST', '--direction', 'bearish',
                 '--target', '90', '--stop', '105', '--account', '5000'],
            )
        assert code == 0
        assert 'BEARISH' in output

    def test_cli_writes_no_option_routes_when_chain_empty(self):
        """Empty chain: output should still show stock route and note no contracts."""
        with _mock_all([], 100.0):
            code, output = _capture_stdout(
                main,
                ['TEST', '--target', '110', '--stop', '95', '--account', '5000'],
            )
        assert code == 0
        assert 'STOCK ROUTE' in output
        assert 'none found' in output.lower() or 'STOCK_PREFERRED' in output

    def test_cli_uses_account_balance_for_stock_sizing(self):
        """Larger account → more shares purchased in stock route."""
        chain = _make_chain(100.0)
        results = {}
        for acct in ['1000', '10000']:
            with _mock_all(chain, 100.0):
                _, output = _capture_stdout(
                    main,
                    ['TEST', '--target', '110', '--stop', '95', '--account', acct],
                )
            results[acct] = output

        # With $10k account at $100/share: ~100 shares
        # With $1k account at $100/share:  ~10 shares
        assert '10 @' in results['1000'] or '10 @ ' in results['1000']
        assert '100 @' in results['10000'] or '99 @' in results['10000']

    def test_cli_returns_nonzero_on_data_fetch_failure(self):
        """If both spot and chain fail, main should return exit code 1."""
        from data.fallback import DataFetchError

        def _fail(*args, **kwargs):
            raise DataFetchError('all sources down')

        with patch('domain.stock_scanner.compare.fetch_with_fallback', side_effect=_fail):
            with patch('scripts.trade_compare.fetch_with_fallback', side_effect=_fail):
                with patch('scripts.trade_compare.default_sources', return_value=[MagicMock()]):
                    with patch('domain.stock_scanner.compare.default_sources', return_value=[MagicMock()]):
                        code = main(['BADTICKER', '--target', '110', '--stop', '95'])
        assert code == 1

    def test_cli_explicit_target_stop_used(self):
        """Explicit --target and --stop appear in the output."""
        chain = _make_chain(100.0)
        with _mock_all(chain, 100.0):
            _, output = _capture_stdout(
                main,
                ['TEST', '--target', '115', '--stop', '93', '--account', '5000'],
            )
        assert '115' in output
        assert '93' in output

    def test_cli_verdict_appears_in_output(self):
        """Output must contain one of the three verdict strings."""
        chain = _make_chain(100.0)
        with _mock_all(chain, 100.0):
            _, output = _capture_stdout(
                main,
                ['TEST', '--target', '110', '--stop', '95', '--account', '5000'],
            )
        verdict_found = any(
            v in output for v in ('STOCK_PREFERRED', 'OPTION_PREFERRED', 'CLOSE_CALL')
        )
        assert verdict_found, f"No verdict found in output:\n{output[:500]}"


# ---------------------------------------------------------------------------
# render_comparison — unit tests
# ---------------------------------------------------------------------------

class TestRenderComparison:

    def _make_stock_route(self) -> StockRoute:
        return StockRoute(
            ticker='TEST',
            shares=10,
            entry_price=100.0,
            cost=1000.0,
            target_pnl=100.0,
            stop_pnl=-30.0,
            target_pct=0.10,
            stop_pct=-0.03,
            rr_ratio=3.33,
        )

    def _make_option_route(self) -> OptionRoute:
        return OptionRoute(
            ticker='TEST',
            option_type='call',
            strike=100.0,
            expiration='2026-07-14',
            dte_at_entry=35,
            entry_mid=3.0,
            cost=300.0,
            iv_at_entry=0.28,
            delta=0.52,
            bid=2.80,
            ask=3.20,
            projected_target_value=7.0,
            projected_stop_value=1.0,
            target_pnl=400.0,
            stop_pnl=-200.0,
            breakeven_spot=103.0,
            rr_ratio=2.0,
            leverage=5.0,
            notes='',
        )

    def _make_comparison(self, option_routes=None, verdict='STOCK_PREFERRED') -> TradeComparison:
        return TradeComparison(
            ticker='TEST',
            direction='bullish',
            spot=100.0,
            target=110.0,
            stop=97.0,
            hold_days=28,
            account_balance=5000.0,
            stock_route=self._make_stock_route(),
            option_routes=option_routes if option_routes is not None else [self._make_option_route()],
            verdict=verdict,
            verdict_reason='Stock R:R is better.',
        )

    def test_render_shows_ticker(self):
        cmp = self._make_comparison()
        _, out = _capture_stdout(render_comparison, cmp)
        assert 'TEST' in out

    def test_render_shows_verdict(self):
        cmp = self._make_comparison(verdict='STOCK_PREFERRED')
        _, out = _capture_stdout(render_comparison, cmp)
        assert 'STOCK_PREFERRED' in out

    def test_render_shows_stock_section(self):
        cmp = self._make_comparison()
        _, out = _capture_stdout(render_comparison, cmp)
        assert 'STOCK ROUTE' in out

    def test_render_shows_option_section(self):
        cmp = self._make_comparison()
        _, out = _capture_stdout(render_comparison, cmp)
        assert 'OPTION ROUTES' in out

    def test_render_no_option_routes_graceful(self):
        cmp = self._make_comparison(option_routes=[])
        _, out = _capture_stdout(render_comparison, cmp)
        assert 'STOCK ROUTE' in out  # still renders stock section
