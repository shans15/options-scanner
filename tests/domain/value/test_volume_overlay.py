"""Volume overlay — 5d/20d ratio + tag."""
from domain.value.types import ValuationInputs
from domain.value.volume_overlay import compute_volume_overlay


def _v(v5, v20):
    return ValuationInputs(
        ticker='X', sector='Technology', price=100.0,
        forward_eps=None, revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None, book_value_per_share=None,
        enterprise_value=None, ebitda_ttm=None,
        shares_short=None, float_shares=None, avg_daily_volume_30d=None,
        volume_5d_avg=v5, volume_20d_avg=v20,
        price_1y_ago=None,
    )


def test_rising_when_ratio_at_or_above_1_15():
    r = compute_volume_overlay(_v(v5=115, v20=100))
    assert r.tag == 'RISING'
    assert r.score == 10.0
    assert abs(r.ratio - 1.15) < 1e-9


def test_flat_when_ratio_between_0_85_and_1_15():
    r = compute_volume_overlay(_v(v5=100, v20=100))
    assert r.tag == 'FLAT'
    assert r.score == 5.0


def test_declining_when_ratio_at_or_below_0_85():
    r = compute_volume_overlay(_v(v5=85, v20=100))
    assert r.tag == 'DECLINING'
    assert r.score == 0.0


def test_rising_just_at_boundary():
    r = compute_volume_overlay(_v(v5=1.151, v20=1.0))
    assert r.tag == 'RISING'


def test_flat_returned_when_data_missing():
    """Missing volume data → tag FLAT, score 5 (neutral)."""
    r = compute_volume_overlay(_v(v5=None, v20=None))
    assert r.tag == 'FLAT'
    assert r.score == 5.0
    assert r.ratio is None
