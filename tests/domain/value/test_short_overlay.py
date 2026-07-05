"""Short interest overlay — direction-aware scoring."""
from domain.value.types import ValuationInputs
from domain.value.short_overlay import compute_short_overlay


def _v(shares_short, float_shares, avg_vol=1_000_000, prior_pct=None):
    return ValuationInputs(
        ticker='X', sector='Technology', price=100.0,
        forward_eps=None, revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None, book_value_per_share=None,
        enterprise_value=None, ebitda_ttm=None,
        shares_short=shares_short, float_shares=float_shares,
        avg_daily_volume_30d=avg_vol,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=None,
    )


def test_long_direction_high_si_scores_ten():
    """base_rank >= 5 (long lean), SI 20% → squeeze validation → 10."""
    v = _v(shares_short=200_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=None)
    assert r.short_interest_pct == 20.0
    assert r.score == 10.0


def test_short_direction_high_si_scores_zero():
    """base_rank < 5 (short lean), SI 20% → crowded short → 0."""
    v = _v(shares_short=200_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=2.0, prior_short_interest_pct=None)
    assert r.score == 0.0


def test_short_direction_low_si_scores_ten():
    """base_rank < 5 (short lean), SI 2% → fresh short → 10."""
    v = _v(shares_short=20_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=2.0, prior_short_interest_pct=None)
    assert r.score == 10.0


def test_missing_data_returns_neutral_five():
    v = _v(shares_short=None, float_shares=None)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=None)
    assert r.score == 5.0
    assert r.short_interest_pct is None


def test_days_to_cover_computed():
    v = _v(shares_short=10_000_000, float_shares=100_000_000, avg_vol=2_000_000)
    r = compute_short_overlay(v, base_rank=5.0, prior_short_interest_pct=None)
    assert r.days_to_cover == 5.0


def test_short_interest_delta_positive():
    v = _v(shares_short=180_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=15.0)
    assert r.short_interest_delta_pp == 3.0


def test_short_interest_delta_none_when_no_prior():
    v = _v(shares_short=180_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=None)
    assert r.short_interest_delta_pp is None
