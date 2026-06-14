from datetime import date
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, LongCall
from domain.signals import Regime
from engine.stress import StressResult
from engine.risk_filters import FilterResult
from engine.scorer import composite_score, label_from, ScoredCandidate


def _make_contract(option_type='put'):
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=100.0,
        option_type=option_type, bid=1.95, ask=2.05, mid=2.0,
        volume=1000, open_interest=1000, implied_volatility=0.25,
        delta=-0.22 if option_type == 'put' else 0.50, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=100.0,
    )


def _regime(favored):
    return Regime(ticker='X', rv_30=0.20, iv_atm=0.20, rv_iv_ratio=1.0, favored=favored)


def test_composite_score_high_when_all_factors_strong():
    score = composite_score(
        pop_blended=0.85, ev=2.0, max_adverse_loss=2.0, strategy=NakedPut(),
        regime=_regime(['sell']), mid=2.0,
    )
    assert score > 90


def test_composite_score_penalized_when_against_regime():
    score = composite_score(
        pop_blended=0.85, ev=2.0, max_adverse_loss=2.0, strategy=NakedPut(),
        regime=_regime(['buy']), mid=2.0,
    )
    assert 75 < score < 80


def test_composite_score_buyer_normalizes_ev_by_mid():
    score = composite_score(
        pop_blended=0.50, ev=0.40, max_adverse_loss=0.0, strategy=LongCall(),
        regime=_regime(['buy']), mid=2.0,
    )
    assert 48 < score < 54


def test_label_trade_when_score_at_threshold():
    assert label_from(score=65.0, all_filters_pass=True) == 'TRADE'
    assert label_from(score=64.99, all_filters_pass=True) == 'WATCHLIST'
    assert label_from(score=50.0, all_filters_pass=True) == 'WATCHLIST'
    assert label_from(score=49.99, all_filters_pass=True) == 'NO_TRADE'


def test_label_no_trade_when_any_filter_fails():
    assert label_from(score=99.0, all_filters_pass=False) == 'NO_TRADE'


def test_scored_candidate_dataclass_carries_explainability():
    sc = ScoredCandidate(
        contract=_make_contract(), strategy=NakedPut(),
        pop_blended=0.80, pop_delta=0.78, pop_bs=0.79, pop_historical=0.81, pop_garch_mc=0.80,
        stress=StressResult(-1, -3, -2), ev=0.5, max_adverse_loss=3.0, margin_estimate=200.0,
        filter_result=FilterResult(passed=True), composite_score=85.0, label='TRADE',
        reason_for='High PoP, in regime', reason_against='',
    )
    assert sc.label == 'TRADE'
    assert sc.reason_for


def test_scored_candidate_supports_optional_setup_fields():
    from engine.scorer import ScoredCandidate
    fields_present = ScoredCandidate.__dataclass_fields__
    assert 'setup_name' in fields_present
    assert 'setup_direction' in fields_present
    assert 'setup_strength' in fields_present


def test_scored_candidate_has_vix_regime_fields():
    from engine.scorer import ScoredCandidate
    fields_present = ScoredCandidate.__dataclass_fields__
    assert 'vix_now' in fields_present
    assert 'vix_regime' in fields_present
    assert 'vix_pct_vs_7d' in fields_present
    # All three should default to None (positional defaults preserved)
    sc = ScoredCandidate(
        contract=_make_contract(), strategy=NakedPut(),
        pop_blended=0.80, pop_delta=0.78, pop_bs=0.79, pop_historical=0.81, pop_garch_mc=0.80,
        stress=StressResult(-1, -3, -2), ev=0.5, max_adverse_loss=3.0, margin_estimate=200.0,
        filter_result=FilterResult(passed=True), composite_score=85.0, label='TRADE',
        reason_for='r', reason_against='',
    )
    assert sc.vix_now is None
    assert sc.vix_regime is None
    assert sc.vix_pct_vs_7d is None


def test_scored_candidate_accepts_vix_fields_when_provided():
    sc = ScoredCandidate(
        contract=_make_contract(), strategy=NakedPut(),
        pop_blended=0.80, pop_delta=0.78, pop_bs=0.79, pop_historical=0.81, pop_garch_mc=0.80,
        stress=StressResult(-1, -3, -2), ev=0.5, max_adverse_loss=3.0, margin_estimate=200.0,
        filter_result=FilterResult(passed=True), composite_score=85.0, label='TRADE',
        reason_for='r', reason_against='',
        vix_now=22.5, vix_regime='expansion', vix_pct_vs_7d=0.18,
    )
    assert sc.vix_now == 22.5
    assert sc.vix_regime == 'expansion'
    assert sc.vix_pct_vs_7d == pytest.approx(0.18)
