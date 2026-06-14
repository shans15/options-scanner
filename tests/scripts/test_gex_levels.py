"""Smoke tests for scripts.gex_levels.

All tests use mocks — no live data fetches.
"""
from __future__ import annotations

import io
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from data.sources.base import RawContract
from domain.gex.compute import GexContext, GexZone
from scripts.gex_levels import format_gex_dollars, render_context, fetch_and_build, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date(2026, 6, 15)
_EXPIRY = date(2026, 7, 15)


def _make_context(
    ticker: str = 'SPY',
    spot: float = 100.0,
    total_gex: float = 1.5e9,
    gfl: float | None = 98.0,
    supply_zones: list | None = None,
    demand_zones: list | None = None,
    chain_size: int = 500,
) -> GexContext:
    supply = supply_zones if supply_zones is not None else [
        GexZone(strike=105.0, gex_dollars=8.4e8, side='above', distance_pct=0.05),
        GexZone(strike=110.0, gex_dollars=6.2e8, side='above', distance_pct=0.10),
    ]
    demand = demand_zones if demand_zones is not None else [
        GexZone(strike=95.0, gex_dollars=7.1e8, side='below', distance_pct=-0.05),
        GexZone(strike=90.0, gex_dollars=4.8e8, side='below', distance_pct=-0.10),
    ]
    return GexContext(
        ticker=ticker,
        spot=spot,
        total_gex_dollars=total_gex,
        gamma_flip_level=gfl,
        supply_zones=supply,
        demand_zones=demand,
        chain_size=chain_size,
    )


def _make_raw_contract(
    strike: float = 100.0,
    option_type: str = 'call',
    oi: int = 100,
    iv: float = 0.20,
) -> RawContract:
    return RawContract(
        ticker='SPY',
        expiration=_EXPIRY,
        strike=strike,
        option_type=option_type,
        bid=1.0,
        ask=1.05,
        volume=100,
        open_interest=oi,
        implied_volatility=iv,
        dte=30,
        spot_price=100.0,
    )


# ---------------------------------------------------------------------------
# format_gex_dollars
# ---------------------------------------------------------------------------

class TestFormatGexDollars:

    def test_billion(self):
        assert format_gex_dollars(1.5e9) == '+$1.50B'

    def test_million(self):
        assert format_gex_dollars(2.3e6) == '+$2.30M'

    def test_thousand(self):
        assert format_gex_dollars(5e3) == '+$5.00K'

    def test_negative_billion(self):
        assert format_gex_dollars(-1.2e9) == '-$1.20B'

    def test_negative_million(self):
        assert format_gex_dollars(-2.5e6) == '-$2.50M'

    def test_negative_thousand(self):
        assert format_gex_dollars(-3e3) == '-$3.00K'

    def test_small_value(self):
        result = format_gex_dollars(500.0)
        assert result.startswith('+$')
        assert '500' in result

    def test_negative_small_value(self):
        result = format_gex_dollars(-500.0)
        assert result.startswith('-$')

    def test_zero(self):
        result = format_gex_dollars(0.0)
        assert result.startswith('+')


# ---------------------------------------------------------------------------
# render_context
# ---------------------------------------------------------------------------

class TestRenderContext:

    def _capture(self, ctx: GexContext, n_zones: int = 5) -> str:
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            render_context(ctx, n_zones=n_zones)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_render_context_handles_empty_supply_zones(self):
        """Context with no supply zones prints 'No supply zones within range'."""
        ctx = _make_context(supply_zones=[])
        output = self._capture(ctx)
        assert 'No supply zones within range' in output

    def test_render_context_handles_empty_demand_zones(self):
        """Context with no demand zones prints 'No demand zones within range'."""
        ctx = _make_context(demand_zones=[])
        output = self._capture(ctx)
        assert 'No demand zones within range' in output

    def test_render_context_shows_ticker(self):
        ctx = _make_context(ticker='QQQ')
        output = self._capture(ctx)
        assert 'QQQ' in output

    def test_render_context_shows_spot(self):
        ctx = _make_context(spot=611.45)
        output = self._capture(ctx)
        assert '611.45' in output

    def test_render_context_positive_regime_label(self):
        ctx = _make_context(total_gex=2.34e9)
        output = self._capture(ctx)
        assert 'vol-suppressed' in output

    def test_render_context_negative_regime_label(self):
        ctx = _make_context(total_gex=-1.2e9)
        output = self._capture(ctx)
        assert 'vol-amplified' in output

    def test_render_context_gfl_none(self):
        ctx = _make_context(gfl=None)
        output = self._capture(ctx)
        assert 'n/a' in output

    def test_render_context_top_annotation(self):
        """The strike with largest abs(gex) should be annotated with '← top'."""
        ctx = _make_context()
        output = self._capture(ctx)
        assert '← top' in output

    def test_render_context_no_crash_with_all_defaults(self):
        ctx = _make_context()
        # Should not raise
        self._capture(ctx)

    def test_render_context_shows_chain_size(self):
        ctx = _make_context(chain_size=3247)
        output = self._capture(ctx)
        assert '3,247' in output


# ---------------------------------------------------------------------------
# fetch_and_build — mocked data layer
# ---------------------------------------------------------------------------

class TestFetchAndBuild:

    def test_fetch_and_build_uses_fetch_with_fallback(self):
        """fetch_with_fallback is called with the right args for chain and spot."""
        mock_chain = [_make_raw_contract()]
        mock_spot = 100.0
        mock_sources = [MagicMock()]

        def _fallback(sources, method, ticker):
            if method == 'fetch_option_chain':
                return mock_chain
            if method == 'fetch_spot':
                return mock_spot
            raise ValueError(f"Unexpected method: {method}")

        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=_fallback) as mock_fb:
            ctx = fetch_and_build(
                ticker='SPY',
                sources=mock_sources,
                max_dte=None,
                n_zones=5,
                zone_max_distance_pct=0.05,
                risk_free_rate=0.053,
            )

        # fetch_with_fallback should have been called at least twice
        assert mock_fb.call_count >= 2
        # First call: fetch_option_chain
        first_call = mock_fb.call_args_list[0]
        assert first_call[0][1] == 'fetch_option_chain'
        assert first_call[0][2] == 'SPY'

    def test_fetch_and_build_returns_none_on_chain_error(self):
        """DataFetchError during chain fetch returns None without crashing."""
        from data.fallback import DataFetchError

        def _fallback(sources, method, ticker):
            raise DataFetchError('no sources available')

        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=_fallback):
            ctx = fetch_and_build(
                ticker='SPY',
                sources=[],
                max_dte=None,
                n_zones=5,
                zone_max_distance_pct=0.05,
                risk_free_rate=0.053,
            )
        assert ctx is None

    def test_fetch_and_build_returns_none_on_empty_chain(self):
        """Empty chain returns None."""
        def _fallback(sources, method, ticker):
            if method == 'fetch_option_chain':
                return []
            return 100.0

        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=_fallback):
            ctx = fetch_and_build(
                ticker='SPY',
                sources=[],
                max_dte=None,
                n_zones=5,
                zone_max_distance_pct=0.05,
                risk_free_rate=0.053,
            )
        assert ctx is None


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

class TestMain:

    def _mock_fetch(self, sources, method, ticker):
        if method == 'fetch_option_chain':
            return [_make_raw_contract()]
        if method == 'fetch_spot':
            return 100.0
        raise ValueError(method)

    def test_cli_main_exits_zero_on_success(self):
        """main(['SPY']) returns 0 when fetch succeeds."""
        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=self._mock_fetch):
            with patch('scripts.gex_levels.default_sources', return_value=[MagicMock()]):
                result = main(['SPY'])
        assert result == 0

    def test_cli_main_exits_nonzero_on_all_failures(self):
        """main returns 1 when all tickers fail."""
        from data.fallback import DataFetchError

        def _fail(sources, method, ticker):
            raise DataFetchError('down')

        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=_fail):
            with patch('scripts.gex_levels.default_sources', return_value=[MagicMock()]):
                result = main(['SPY'])
        assert result == 1

    def test_cli_main_multi_ticker(self):
        """Multiple tickers run without crash; returns 0 if at least one succeeds."""
        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=self._mock_fetch):
            with patch('scripts.gex_levels.default_sources', return_value=[MagicMock()]):
                result = main(['SPY', 'QQQ'])
        assert result == 0

    def test_cli_main_continues_after_one_ticker_failure(self):
        """When first ticker fails, other tickers are still attempted."""
        from data.fallback import DataFetchError
        call_count = {'n': 0}

        def _mixed(sources, method, ticker):
            if ticker == 'SPY' and method == 'fetch_option_chain':
                raise DataFetchError('SPY down')
            if method == 'fetch_option_chain':
                return [_make_raw_contract()]
            return 100.0

        with patch('scripts.gex_levels.fetch_with_fallback', side_effect=_mixed):
            with patch('scripts.gex_levels.default_sources', return_value=[MagicMock()]):
                result = main(['SPY', 'QQQ'])
        # QQQ succeeded → exit 0
        assert result == 0
