from domain.aplus.types import FeatureScores
from domain.aplus.scoring import score_categories


def _build_fs(uniform: float) -> FeatureScores:
    keys = [
        # Technical
        'tech_setup_type', 'tech_setup_strength', 'tech_weekly_ribbon_agreement',
        'tech_atr_pivot_position', 'tech_volume_zscore',
        # Vol/VIX
        'vol_vix_regime', 'vol_vix_zscore_30d', 'vol_vvix_level', 'vol_iv_percentile',
        # Catalyst
        'cat_days_to_earnings', 'cat_days_to_macro_event', 'cat_skip_window',
        # Macro/Breadth
        'macro_spx_trend', 'macro_sector_rotation', 'macro_dxy_trend',
        'macro_yield_10y_direction',
        # Liquidity
        'liq_bid_ask_spread', 'liq_open_interest', 'liq_oi_change_dod',
        'liq_volume_oi_ratio',
    ]
    return FeatureScores(values={k: uniform for k in keys})


def test_score_categories_all_tens_gives_perfect():
    cs = score_categories(_build_fs(10.0))
    assert cs.technical == 10.0
    assert cs.vol_vix == 10.0
    assert cs.catalyst == 10.0
    assert cs.macro_breadth == 10.0
    assert cs.liquidity == 10.0
    assert cs.composite() == 100.0


def test_score_categories_all_zeros_gives_zero():
    cs = score_categories(_build_fs(0.0))
    assert cs.composite() == 0.0


def test_score_categories_partial_weights_correctly():
    # Technical=10, others=0 → composite = 25 * 10 = 25.0 (25% weight × 10 × 10 normalize)
    fs = _build_fs(0.0)
    fs.values['tech_setup_type'] = 10.0
    fs.values['tech_setup_strength'] = 10.0
    fs.values['tech_weekly_ribbon_agreement'] = 10.0
    fs.values['tech_atr_pivot_position'] = 10.0
    fs.values['tech_volume_zscore'] = 10.0
    cs = score_categories(fs)
    assert cs.technical == 10.0
    assert cs.vol_vix == 0.0
    assert cs.composite() == 25.0


def test_score_categories_handles_missing_keys_as_neutral():
    # If a feature key is missing entirely, treat it as neutral (5)
    fs = FeatureScores(values={'tech_setup_type': 10.0})
    cs = score_categories(fs)
    # Only one of five technical features is present; others default to 5.
    assert 4.5 < cs.technical < 7.0
