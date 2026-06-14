"""Trade structure selector — long premium vs debit spread."""
from __future__ import annotations
from domain.aplus.types import FeatureScores, TradeStructure


def select_structure(fs: FeatureScores) -> tuple[TradeStructure, str]:
    """Choose a trade structure based on the feature scores.

    Branch order matches the spec in §5 of the design doc.
    """
    vix_iv_percentile = fs.values.get('vol_iv_percentile', 5.0)
    if vix_iv_percentile <= 3.0:
        # Low score on vol_iv_percentile = HIGH IV — prefer debit spread
        return TradeStructure.DEBIT_SPREAD, "IV percentile high — vega exposure would hurt long premium"

    post_earnings = fs.values.get('cat_post_earnings_drift', 5.0)
    spread = fs.values.get('liq_bid_ask_spread', 5.0)
    if post_earnings >= 9.0 and spread >= 9.0:
        return TradeStructure.LONG_PREMIUM, "post-earnings drift + tight spreads = asymmetric upside"

    sector_rot = fs.values.get('macro_sector_rotation', 5.0)
    vix_regime = fs.values.get('vol_vix_regime', 5.0)
    if sector_rot >= 9.0 and vix_regime >= 7.0:
        return TradeStructure.LONG_PREMIUM, "sector leadership + vol-supportive = trend continuation"

    return TradeStructure.DEBIT_SPREAD, "defensive default — defined R:R, better PoP"
