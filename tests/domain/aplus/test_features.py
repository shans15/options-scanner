from datetime import date
from domain.aplus.features import (
    score_setup_type, score_setup_strength,
    score_weekly_ribbon_agreement, score_atr_pivot_position,
    score_volume_zscore, score_vix_regime, score_vix_zscore_30d,
    score_vvix_level, score_iv_percentile,
)


def test_score_setup_type_compression_high():
    assert score_setup_type('compression_breakout') == 10.0


def test_score_setup_type_pullback_low():
    assert score_setup_type('pullback_in_trend') <= 4.0


def test_score_setup_type_unknown_neutral():
    assert score_setup_type(None) == 5.0


def test_score_setup_strength_max():
    assert score_setup_strength(1.0) == 10.0


def test_score_setup_strength_zero():
    assert score_setup_strength(0.0) == 0.0


def test_score_setup_strength_none_neutral():
    assert score_setup_strength(None) == 5.0


def test_score_weekly_ribbon_agreement_true_high():
    assert score_weekly_ribbon_agreement(True) == 10.0


def test_score_weekly_ribbon_agreement_false_mid():
    assert score_weekly_ribbon_agreement(False) == 5.0


def test_score_weekly_ribbon_agreement_none_neutral():
    assert score_weekly_ribbon_agreement(None) == 5.0


def test_score_atr_pivot_position_at_pivot_high():
    # spot at the same as prior close → 0 ATR distance
    assert score_atr_pivot_position(spot=100.0, prev_close=100.0, atr=1.0) == 10.0


def test_score_atr_pivot_position_two_atr_low():
    # 2 ATR away → 0
    assert score_atr_pivot_position(spot=102.0, prev_close=100.0, atr=1.0) == 0.0


def test_score_volume_zscore_high_volume_high_score():
    assert score_volume_zscore(z=2.0) == 10.0


def test_score_volume_zscore_low_volume_zero():
    assert score_volume_zscore(z=-1.0) == 0.0


def test_score_vix_regime_expansion_zero():
    assert score_vix_regime('expansion') == 0.0


def test_score_vix_regime_contraction_max():
    assert score_vix_regime('contraction') == 10.0


def test_score_vix_regime_neutral_mid_high():
    assert score_vix_regime('neutral') == 7.0


def test_score_vix_regime_none_neutral():
    assert score_vix_regime(None) == 5.0


def test_score_vix_zscore_30d_low_high_score():
    assert score_vix_zscore_30d(-0.5) >= 8.0


def test_score_vix_zscore_30d_high_low_score():
    assert score_vix_zscore_30d(1.0) <= 4.0


def test_score_vvix_level_uses_market_context_value():
    assert score_vvix_level(8.5) == 8.5


def test_score_iv_percentile_low_iv_high_score():
    assert score_iv_percentile(0.18) >= 8.0


def test_score_iv_percentile_high_iv_low_score():
    assert score_iv_percentile(0.60) <= 4.0


from domain.aplus.features import (
    score_days_to_earnings, score_days_to_macro_event,
    score_skip_window, score_spx_alignment, score_sector_rotation,
    score_dxy_trend_pass_through,
)


def test_score_days_to_earnings_within_blackout_zero():
    assert score_days_to_earnings(3) == 0.0


def test_score_days_to_earnings_in_window_high():
    assert score_days_to_earnings(10) == 10.0


def test_score_days_to_earnings_far_neutral():
    assert score_days_to_earnings(60) == 6.0


def test_score_days_to_macro_event_within_2_days_zero():
    assert score_days_to_macro_event(1) == 0.0


def test_score_days_to_macro_event_5_days_out_high():
    assert score_days_to_macro_event(10) == 10.0


def test_score_skip_window_combines_blockers():
    # If both earnings and macro are clear, returns 10
    assert score_skip_window(days_earnings=15, days_macro=10) == 10.0
    # If either is blocking (≤2), returns 0
    assert score_skip_window(days_earnings=1, days_macro=10) == 0.0


def test_score_spx_alignment_pass_through():
    # Test that we just pass through the precomputed market_context value
    assert score_spx_alignment(8.0) == 8.0


def test_score_sector_rotation_top_sector_bullish_high():
    score = score_sector_rotation(rank=1, total_sectors=11, setup_direction='bullish')
    assert score >= 9.0


def test_score_sector_rotation_bottom_sector_bullish_low():
    score = score_sector_rotation(rank=11, total_sectors=11, setup_direction='bullish')
    assert score <= 1.0


def test_score_sector_rotation_top_sector_bearish_low():
    # If the sector is leading and the setup is bearish, that's bad
    score = score_sector_rotation(rank=1, total_sectors=11, setup_direction='bearish')
    assert score <= 1.0
