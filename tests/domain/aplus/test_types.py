import pytest
from domain.aplus.types import (
    FeatureScores, CategoryScores, MarketContext,
    TradeStructure, GradedCandidate,
)


def test_feature_scores_holds_named_features():
    fs = FeatureScores(values={'tech_setup_type': 9.0, 'tech_setup_strength': 8.0})
    assert fs.values['tech_setup_type'] == 9.0


def test_category_scores_has_five_categories():
    cs = CategoryScores(technical=8.0, vol_vix=7.0, catalyst=6.0,
                        macro_breadth=7.0, liquidity=9.0)
    assert cs.technical == 8.0
    assert cs.vol_vix == 7.0


def test_market_context_frozen():
    mc = MarketContext(
        spx_trend_score=8.0, sector_rotation_rank={'XLK': 1},
        dxy_trend_score=5.0, yield_10y_score=6.0, vvix_score=7.0,
        days_to_macro_event=3,
    )
    with pytest.raises(AttributeError):
        mc.spx_trend_score = 9.0


def test_trade_structure_is_long_premium_or_debit_spread():
    assert TradeStructure.LONG_PREMIUM.value == 'long_premium'
    assert TradeStructure.DEBIT_SPREAD.value == 'debit_spread'


def test_graded_candidate_carries_grade_and_structure():
    gc = GradedCandidate(
        ticker='AAPL', strategy='long_call', composite_score=85.2, grade='A',
        category_scores=CategoryScores(8.0, 8.0, 7.0, 7.0, 9.0),
        feature_scores=FeatureScores(values={}),
        structure=TradeStructure.DEBIT_SPREAD, structure_rationale='IV>70',
        sizing_pct=0.08, max_risk_dollars=80.0,
        raw_candidate={},
    )
    assert gc.grade == 'A'
    assert gc.structure == TradeStructure.DEBIT_SPREAD
