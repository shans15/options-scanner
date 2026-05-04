from __future__ import annotations
from scanner.risk_filters import apply_hard_filters, FilterResult


def _base_contract(**overrides: object) -> dict:
    c = {
        'ticker': 'AAPL',
        'strategy': 'naked_put',
        'bid': 1.10,
        'ask': 1.30,
        'mid': 1.20,
        'volume': 500,
        'open_interest': 1000,
        'delta': -0.18,
        'implied_volatility': 0.28,
        'dte': 14,
        'spot_price': 195.0,
        'strike': 185.0,
    }
    c.update(overrides)
    return c


def _base_pop(**overrides: object) -> object:
    class MockStress:
        stress_2sd = -1.20

    class MockPop:
        pop_blended = 0.82
        expected_value = 0.90
        stress = MockStress()

    pop = MockPop()
    for k, v in overrides.items():
        setattr(pop, k, v)
    return pop


def test_good_contract_passes():
    result = apply_hard_filters(_base_contract(), _base_pop())
    assert result.passed is True
    assert result.reason == ''


def test_wide_spread_fails():
    c = _base_contract(bid=0.10, ask=1.50, mid=0.80)
    result = apply_hard_filters(c, _base_pop())
    assert result.passed is False
    assert 'spread' in result.reason.lower()


def test_low_volume_fails():
    result = apply_hard_filters(_base_contract(volume=50), _base_pop())
    assert result.passed is False
    assert 'volume' in result.reason.lower()


def test_low_oi_fails():
    result = apply_hard_filters(_base_contract(open_interest=100), _base_pop())
    assert result.passed is False
    assert 'open interest' in result.reason.lower()


def test_low_pop_fails():
    result = apply_hard_filters(_base_contract(), _base_pop(pop_blended=0.60))
    assert result.passed is False
    assert 'pop' in result.reason.lower()


def test_negative_ev_fails():
    result = apply_hard_filters(_base_contract(), _base_pop(expected_value=-0.50))
    assert result.passed is False
    assert 'expected value' in result.reason.lower()


def test_excessive_stress_loss_fails():
    # premium = 1.20, threshold = 3x = 3.60. stress_2sd = -5.0 → loss = 5.0 > 3.60
    class BadStress:
        stress_2sd = -5.0

    class BadPop:
        pop_blended = 0.82
        expected_value = 0.90
        stress = BadStress()

    result = apply_hard_filters(_base_contract(), BadPop())
    assert result.passed is False
    assert 'stress' in result.reason.lower()


def test_delta_out_of_range_fails():
    result = apply_hard_filters(_base_contract(delta=-0.50), _base_pop())
    assert result.passed is False
    assert 'delta' in result.reason.lower()
