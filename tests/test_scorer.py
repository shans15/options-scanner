from __future__ import annotations
from scanner.scorer import compute_composite_score, assign_decision, build_scan_result


def _make_pop():
    class MockStress:
        stress_1sd = -0.30
        stress_2sd = -1.10
        stress_expiry = -2.00

    class MockPop:
        pop_delta = 0.85
        pop_bs = 0.83
        pop_historical = 0.80
        pop_garch_mc = 0.82
        pop_blended = 0.823
        expected_value = 1.05
        breakeven = 183.80
        margin_estimate = 1850.0
        iv_rank = 72.0
        stress = MockStress()

    return MockPop()


def _make_contract(**overrides: object) -> dict:
    c = {
        'ticker': 'AAPL', 'strategy': 'naked_put', 'expiration': '2026-05-15',
        'strike': 185.0, 'bid': 1.10, 'ask': 1.30, 'mid': 1.20,
        'volume': 800, 'open_interest': 2000, 'implied_volatility': 0.28,
        'delta': -0.18, 'gamma': 0.05, 'theta': -0.04, 'vega': 0.12,
        'dte': 11, 'spot_price': 195.0,
    }
    c.update(overrides)
    return c


def test_composite_score_in_range():
    score = compute_composite_score(_make_contract(), _make_pop())
    assert 0 <= score <= 100


def test_trade_decision_above_threshold():
    assert assign_decision(score=70.0, hard_filter_passed=True) == 'TRADE'


def test_watchlist_decision():
    assert assign_decision(score=55.0, hard_filter_passed=True) == 'WATCHLIST'


def test_no_trade_when_filter_fails():
    assert assign_decision(score=80.0, hard_filter_passed=False) == 'NO TRADE'


def test_no_trade_when_score_too_low():
    assert assign_decision(score=40.0, hard_filter_passed=True) == 'NO TRADE'


def test_build_scan_result_has_all_keys():
    result = build_scan_result(_make_contract(), _make_pop(), hard_filter_passed=True, filter_reason='')
    required_keys = [
        'ticker', 'strategy', 'PoP_blended', 'composite_score',
        'decision', 'breakeven', 'margin_estimate', 'reason_for',
        'reason_against', 'stop_trigger', 'stress_1SD', 'stress_2SD',
        'hard_filter_passed', 'hard_filter_reason',
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_no_trade_overrides_score():
    result = build_scan_result(
        _make_contract(), _make_pop(),
        hard_filter_passed=False, filter_reason='Volume too low'
    )
    assert result['decision'] == 'NO TRADE'
    assert result['hard_filter_reason'] == 'Volume too low'
