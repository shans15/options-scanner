"""Grading thresholds per spec 2026-06-29-aplus-rebalance-design.md."""
from domain.aplus.types import CategoryScores
from domain.aplus.grading import assign_grade


def _cs(t=7.0, v=7.0, c=7.0, m=7.0, l=7.0):
    return CategoryScores(technical=t, vol_vix=v, catalyst=c, macro_breadth=m, liquidity=l)


# ----- A+ -----
def test_a_plus_when_composite_at_or_above_75_and_floors_met():
    # All categories at 8 → composite = (0.35+0.10+0.25+0.10+0.20)*8*10 = 80.0
    cs = _cs(8.0, 8.0, 8.0, 8.0, 8.0)
    assert cs.composite() == 80.0
    assert assign_grade(cs) == 'A+'


def test_a_plus_blocked_when_any_category_below_6():
    cs = _cs(t=10.0, v=5.9, c=10.0, m=10.0, l=10.0)
    # composite ≥ 75 but vol_vix below 6 floor → not A+
    grade = assign_grade(cs)
    assert grade != 'A+'


def test_a_plus_blocked_by_expansion_regime():
    cs = _cs(9.0, 9.0, 9.0, 9.0, 9.0)
    assert assign_grade(cs, vix_regime='expansion') == 'F'


# ----- A -----
def test_a_when_composite_between_74_and_75_with_floors_met():
    # tech=8 v=6 c=8 m=6 l=7 → composite = 0.35*8 + 0.10*6 + 0.25*8 + 0.10*6 + 0.20*7 = 7.4 → *10 = 74.0
    cs = _cs(t=8.0, v=6.0, c=8.0, m=6.0, l=7.0)
    assert 74.0 <= cs.composite() < 75.0
    assert assign_grade(cs) == 'A'


def test_a_blocked_when_any_category_below_6():
    cs = _cs(t=10.0, v=5.0, c=10.0, m=6.0, l=10.0)
    grade = assign_grade(cs)
    assert grade not in ('A', 'A+')


# ----- B+ -----
def test_b_plus_when_composite_at_67_with_no_floor_required():
    # All categories at 6.7 → composite = 67
    cs = _cs(6.7, 6.7, 6.7, 6.7, 6.7)
    assert round(cs.composite(), 1) == 67.0
    assert assign_grade(cs) == 'B+'


def test_b_plus_does_not_require_category_floor():
    # tech=7.5 v=4 c=7.5 m=4 l=7.5 → composite = 6.8*10 = 68.  v and m below 6 floor;
    # B+ tier doesn't require the floor (per spec), so grade is still B+.
    cs = _cs(t=7.5, v=4.0, c=7.5, m=4.0, l=7.5)
    assert cs.composite() >= 67.0
    assert assign_grade(cs) == 'B+'


# ----- B -----
def test_b_when_composite_between_60_and_67():
    # Uniform 6.3 → composite = 63
    cs = _cs(6.3, 6.3, 6.3, 6.3, 6.3)
    assert 60.0 <= cs.composite() < 67.0
    assert assign_grade(cs) == 'B'


# ----- F -----
def test_f_when_composite_below_60():
    cs = _cs(5.0, 5.0, 5.0, 5.0, 5.0)
    assert assign_grade(cs) == 'F'


def test_f_when_expansion_regime_overrides_everything():
    cs = _cs(10.0, 10.0, 10.0, 10.0, 10.0)
    # Composite = 100 but expansion → F
    assert assign_grade(cs, vix_regime='expansion') == 'F'


# ----- vix_regime variants -----
def test_neutral_and_contraction_and_none_all_allow_grading():
    cs = _cs(8.0, 8.0, 8.0, 8.0, 8.0)  # composite 80
    assert assign_grade(cs, vix_regime='neutral') == 'A+'
    assert assign_grade(cs, vix_regime='contraction') == 'A+'
    assert assign_grade(cs, vix_regime=None) == 'A+'
    assert assign_grade(cs) == 'A+'  # default arg


def test_unknown_vix_regime_string_does_not_trigger_kill():
    cs = _cs(8.0, 8.0, 8.0, 8.0, 8.0)
    # Defensive: only the literal 'expansion' kills.
    assert assign_grade(cs, vix_regime='something_else') == 'A+'


def test_composite_at_old_a_threshold_is_now_b_plus():
    # Composite 73.0: at the previous A threshold of 73 this was A; under the
    # current threshold of 74 it lands at B+.  Values: t=8 v=6 c=8 m=6 l=6.5
    # → 0.35*8 + 0.10*6 + 0.25*8 + 0.10*6 + 0.20*6.5 = 7.30 → *10 = 73.0
    cs = _cs(t=8.0, v=6.0, c=8.0, m=6.0, l=6.5)
    assert 73.0 <= cs.composite() < 74.0
    assert assign_grade(cs) == 'B+'
