"""Methodology A — sector-relative multiples."""
from domain.value.types import ValuationInputs
from domain.value.sector_relative import (
    compute_sector_medians,
    compute_sector_relative,
)


def _make(ticker: str, sector: str, pe: float, ps: float, pb: float, ev_ebitda: float):
    """Constructs a ValuationInputs whose derived ratios will be exactly
    pe, ps, pb, ev_ebitda when the sector_relative module reads them.

    Prices are all 100 for simplicity; back-fields are set to yield the
    target ratios via the module's computation formula.
    """
    return ValuationInputs(
        ticker=ticker, sector=sector, price=100.0,
        forward_eps=100.0 / pe,               # price/eps = pe
        revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None,
        book_value_per_share=100.0 / pb,      # price/book = pb
        enterprise_value=ev_ebitda * 1_000_000_000,
        ebitda_ttm=1_000_000_000,             # EV/EBITDA = ev_ebitda
        shares_short=None, float_shares=None,
        avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=None,
    )


def test_sector_medians_computed_across_sector():
    """Given 3 tickers in a sector, medians should equal the middle value."""
    inputs = [
        _make('A', 'Technology', pe=10, ps=1, pb=1, ev_ebitda=5),
        _make('B', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10),
        _make('C', 'Technology', pe=30, ps=3, pb=3, ev_ebitda=15),
    ]
    medians = compute_sector_medians(inputs, min_peers=3)
    assert 'Technology' in medians
    assert medians['Technology']['pe_fwd'] == 20.0
    assert medians['Technology']['pb'] == 2.0


def test_sector_medians_require_min_10_peers_by_default():
    """Sectors with fewer than 10 members should not appear in medians."""
    inputs = [_make(f'T{i}', 'Utilities', pe=15, ps=2, pb=1, ev_ebitda=8)
              for i in range(5)]
    medians = compute_sector_medians(inputs, min_peers=10)
    assert 'Utilities' not in medians


def test_sector_relative_score_undervalued():
    """A ticker at 50% discount to peer median PE should score high."""
    inputs = [
        _make('A', 'Technology', pe=10, ps=1, pb=1, ev_ebitda=5),  # our ticker
        *[_make(f'B{i}', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10)
          for i in range(10)],
    ]
    medians = compute_sector_medians(inputs)
    result = compute_sector_relative(inputs[0], medians)
    assert result is not None
    assert result.score >= 8.0
    assert result.pe_discount_pct is not None and result.pe_discount_pct > 40.0


def test_sector_relative_score_overvalued():
    inputs = [
        _make('A', 'Technology', pe=40, ps=4, pb=4, ev_ebitda=20),
        *[_make(f'B{i}', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10)
          for i in range(10)],
    ]
    medians = compute_sector_medians(inputs)
    result = compute_sector_relative(inputs[0], medians)
    assert result is not None
    assert result.score <= 3.0


def test_sector_relative_none_when_sector_missing_from_medians():
    """Ticker in a sector with too few peers → None."""
    solo = _make('Q', 'Utilities', pe=15, ps=2, pb=1, ev_ebitda=8)
    medians = compute_sector_medians([solo], min_peers=10)
    assert compute_sector_relative(solo, medians) is None


def test_sector_relative_requires_at_least_two_ratios():
    """A ticker with only one computable ratio should be skipped."""
    v = ValuationInputs(
        ticker='X', sector='Technology', price=100.0,
        forward_eps=5.0,                       # gives PE only
        revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None,
        book_value_per_share=None,             # no PB
        enterprise_value=None,                  # no EV/EBITDA
        ebitda_ttm=None,
        shares_short=None, float_shares=None,
        avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=None,
    )
    peers = [_make(f'B{i}', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10)
             for i in range(10)]
    medians = compute_sector_medians([v] + peers)
    result = compute_sector_relative(v, medians)
    assert result is None
