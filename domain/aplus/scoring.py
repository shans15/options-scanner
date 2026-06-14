"""Aggregate feature scores into category scores + composite."""
from __future__ import annotations
from domain.aplus.types import FeatureScores, CategoryScores


_CATEGORY_FEATURES: dict[str, list[str]] = {
    'technical': [
        'tech_setup_type', 'tech_setup_strength', 'tech_weekly_ribbon_agreement',
        'tech_atr_pivot_position', 'tech_volume_zscore',
    ],
    'vol_vix': [
        'vol_vix_regime', 'vol_vix_zscore_30d', 'vol_vvix_level', 'vol_iv_percentile',
    ],
    'catalyst': [
        'cat_days_to_earnings', 'cat_days_to_macro_event', 'cat_skip_window',
    ],
    'macro_breadth': [
        'macro_spx_trend', 'macro_sector_rotation', 'macro_dxy_trend',
        'macro_yield_10y_direction',
    ],
    'liquidity': [
        'liq_bid_ask_spread', 'liq_open_interest', 'liq_oi_change_dod',
        'liq_volume_oi_ratio',
    ],
}


def score_categories(fs: FeatureScores) -> CategoryScores:
    """Compute category-level averages from individual feature scores.
    Missing keys default to 5.0 (neutral)."""

    def _avg(category: str) -> float:
        keys = _CATEGORY_FEATURES[category]
        vals = [fs.values.get(k, 5.0) for k in keys]
        return sum(vals) / len(vals)

    return CategoryScores(
        technical=_avg('technical'),
        vol_vix=_avg('vol_vix'),
        catalyst=_avg('catalyst'),
        macro_breadth=_avg('macro_breadth'),
        liquidity=_avg('liquidity'),
    )
