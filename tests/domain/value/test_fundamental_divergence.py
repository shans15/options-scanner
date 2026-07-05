"""Methodology B — fundamental momentum divergence."""
from domain.value.types import ValuationInputs
from domain.value.fundamental_divergence import compute_fundamental_divergence


def _v(revenue: float, revenue_prior: float, eps: float, eps_prior: float,
       price: float, price_prior: float, ticker: str = 'X'):
    return ValuationInputs(
        ticker=ticker, sector='Technology', price=price,
        forward_eps=None,
        revenue_ttm=revenue, revenue_ttm_1y_ago=revenue_prior,
        eps_ttm=eps, eps_ttm_1y_ago=eps_prior,
        book_value_per_share=None, enterprise_value=None, ebitda_ttm=None,
        shares_short=None, float_shares=None, avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=price_prior,
    )


def test_undervalued_when_fundamentals_up_price_down():
    v = _v(revenue=112_000_000, revenue_prior=100_000_000,
           eps=1.087, eps_prior=1.0,
           price=81.6, price_prior=100.0)
    r = compute_fundamental_divergence(v)
    assert r is not None
    assert r.divergence_pp > 25
    assert r.score >= 8.0


def test_overvalued_when_price_ran_ahead():
    v = _v(revenue=101_000_000, revenue_prior=100_000_000,
           eps=1.01, eps_prior=1.0,
           price=150, price_prior=100)
    r = compute_fundamental_divergence(v)
    assert r is not None
    assert r.divergence_pp < -25
    assert r.score <= 2.0


def test_aligned_when_growth_and_price_match():
    v = _v(revenue=110_000_000, revenue_prior=100_000_000,
           eps=1.10, eps_prior=1.0,
           price=110, price_prior=100)
    r = compute_fundamental_divergence(v)
    assert r is not None
    assert -10 <= r.divergence_pp <= 10
    assert 4.0 <= r.score <= 6.0


def test_skip_when_revenue_below_50m():
    v = _v(revenue=10_000_000, revenue_prior=8_000_000,
           eps=0.10, eps_prior=0.08,
           price=10, price_prior=8)
    assert compute_fundamental_divergence(v) is None


def test_skip_when_eps_crosses_zero():
    v = _v(revenue=100_000_000, revenue_prior=90_000_000,
           eps=0.5, eps_prior=-0.2,
           price=10, price_prior=8)
    assert compute_fundamental_divergence(v) is None


def test_revenue_only_when_eps_negative_both_periods():
    """If both EPS values are negative, use revenue-only weighting."""
    v = _v(revenue=100_000_000, revenue_prior=100_000_000,
           eps=-0.5, eps_prior=-1.0,
           price=90, price_prior=100)
    r = compute_fundamental_divergence(v)
    assert r is not None
    # revenue_growth = 0, price_growth = -10% → divergence = +10 pp
    assert 5 <= r.divergence_pp <= 15


def test_skip_when_missing_price_history():
    v = _v(revenue=100_000_000, revenue_prior=100_000_000,
           eps=1.0, eps_prior=1.0,
           price=100, price_prior=100)
    v_missing = ValuationInputs(**{**v.__dict__, 'price_1y_ago': None})
    assert compute_fundamental_divergence(v_missing) is None
