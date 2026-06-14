"""Per-candidate feature extractors for the A+ confluence scorer.

Each function returns a float in [0, 10]. Missing inputs return 5.0 (neutral)
unless the spec specifies a different default (e.g., VIX expansion forces 0).
"""
from __future__ import annotations
from typing import Optional
from datetime import date


_SETUP_TYPE_SCORES = {
    'compression_breakout': 10.0,
    'stage_2_breakout': 8.0,
    'failed_breakdown_reversal': 5.0,
    'pullback_in_trend': 3.0,
}


def score_setup_type(setup_name: Optional[str]) -> float:
    if setup_name is None:
        return 5.0
    return _SETUP_TYPE_SCORES.get(setup_name, 5.0)


def score_setup_strength(strength: Optional[float]) -> float:
    if strength is None:
        return 5.0
    return max(0.0, min(10.0, strength * 10.0))


def score_weekly_ribbon_agreement(agreement: Optional[bool]) -> float:
    if agreement is None:
        return 5.0
    return 10.0 if agreement else 5.0


def score_atr_pivot_position(spot: float, prev_close: float, atr: float) -> float:
    """10 if spot is at prior daily close; linear decline to 0 at 2 ATR away."""
    if atr <= 0:
        return 5.0
    distance = abs(spot - prev_close) / atr
    return max(0.0, 10.0 - distance * 5.0)


def score_volume_zscore(z: float) -> float:
    """Higher relative volume scores higher. 10 at z=2, 0 at z=-1."""
    return max(0.0, min(10.0, (z + 1.0) * (10.0 / 3.0)))


def score_vix_regime(regime: Optional[str]) -> float:
    if regime == 'expansion':
        return 0.0
    if regime == 'contraction':
        return 10.0
    if regime == 'neutral':
        return 7.0
    return 5.0


def score_vix_zscore_30d(z: float) -> float:
    """Lower VIX z-score scores higher; higher z penalises."""
    return max(0.0, min(10.0, 10.0 - z * 6.0))


def score_vvix_level(category_score: float) -> float:
    """Pass-through — MarketContext already converted VVIX level to a score."""
    return max(0.0, min(10.0, category_score))


def score_iv_percentile(iv: float) -> float:
    """For LONG premium scoring: low IV is better. 10 at IV<=20%, 0 at IV>=60%."""
    if iv <= 0.20:
        return 10.0
    if iv >= 0.60:
        return 0.0
    return 10.0 - (iv - 0.20) / 0.40 * 10.0


def score_days_to_earnings(days: Optional[int]) -> float:
    """0 if <=5 days (blackout), 10 if 5-15 days (post-earnings window),
    6 if 15-30, 6 if >30 (no catalyst)."""
    if days is None:
        return 6.0
    if days <= 5:
        return 0.0
    if days <= 15:
        return 10.0
    return 6.0


def score_days_to_macro_event(days: Optional[int]) -> float:
    """0 if <=2 days, 10 if >5 days, linear in between."""
    if days is None:
        return 6.0
    if days <= 2:
        return 0.0
    if days >= 5:
        return 10.0
    return (days - 2) / 3.0 * 10.0


def score_skip_window(days_earnings: Optional[int], days_macro: Optional[int]) -> float:
    """0 if either earnings or macro event is within 2 days; 10 otherwise."""
    if (days_earnings is not None and days_earnings <= 2) or \
       (days_macro is not None and days_macro <= 2):
        return 0.0
    return 10.0


def score_spx_alignment(precomputed_score: float) -> float:
    """Pass-through from MarketContext.spx_trend_score."""
    return max(0.0, min(10.0, precomputed_score))


def score_sector_rotation(rank: int, total_sectors: int, setup_direction: str) -> float:
    """Sector leadership: rank 1 (best) → 10 for bullish, 0 for bearish.
    Sector laggard: rank N (worst) → 0 for bullish, 10 for bearish."""
    if rank < 1 or rank > total_sectors:
        return 5.0
    pct = (rank - 1) / (total_sectors - 1)  # 0.0 = best, 1.0 = worst
    if setup_direction == 'bullish':
        return 10.0 * (1.0 - pct)
    if setup_direction == 'bearish':
        return 10.0 * pct
    return 5.0


def score_dxy_trend_pass_through(precomputed_score: float) -> float:
    """Pass-through from MarketContext.dxy_trend_score."""
    return max(0.0, min(10.0, precomputed_score))


# ---------------------------------------------------------------------------
# Liquidity scorers (3)
# ---------------------------------------------------------------------------

def score_bid_ask_spread(bid: float, ask: float) -> float:
    """Score based on relative spread: tight (<3% of mid) = 10, wide (>15%) = 0."""
    mid = (bid + ask) / 2
    if mid <= 0:
        return 0.0
    spread_pct = (ask - bid) / mid
    if spread_pct <= 0.03:
        return 10.0
    if spread_pct >= 0.15:
        return 0.0
    return 10.0 - (spread_pct - 0.03) / 0.12 * 10.0


def score_open_interest(oi: int) -> float:
    """OI >= 1000 = 10; OI < 100 = 0; linear in between."""
    if oi >= 1000:
        return 10.0
    if oi <= 100:
        return 0.0
    return (oi - 100) / 900.0 * 10.0


def score_volume_oi_ratio(volume: int, open_interest: int) -> float:
    """Volume / OI > 0.3 → 10 (active flow). < 0.05 → 3 (dormant)."""
    if open_interest <= 0:
        return 3.0
    ratio = volume / open_interest
    if ratio >= 0.3:
        return 10.0
    if ratio <= 0.05:
        return 3.0
    return 3.0 + (ratio - 0.05) / 0.25 * 7.0


# Sector → ETF lookup for sector rotation feature. v1 covers liquid sector mappings.
_TICKER_SECTOR_ETF: dict[str, str] = {
    # Sector ETFs map to themselves
    'XLK': 'XLK', 'XLF': 'XLF', 'XLE': 'XLE', 'XLV': 'XLV', 'XLI': 'XLI',
    'XLP': 'XLP', 'XLY': 'XLY', 'XLB': 'XLB', 'XLU': 'XLU', 'XLRE': 'XLRE', 'XLC': 'XLC',
    # Mega cap → sector
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'GOOGL': 'XLC', 'META': 'XLC',
    'AMZN': 'XLY', 'TSLA': 'XLY', 'AMD': 'XLK', 'JPM': 'XLF', 'V': 'XLF',
    'UNH': 'XLV', 'COST': 'XLP', 'NFLX': 'XLC',
}


def extract_features(
    candidate: dict,
    market_context: 'MarketContext',
    days_to_earnings: Optional[int] = None,
) -> 'FeatureScores':
    """Combine all 20 feature scorers into a single FeatureScores object."""
    from domain.aplus.types import FeatureScores

    contract = candidate.get('contract', {})
    ticker = contract.get('ticker', '')
    setup_dir = candidate.get('setup_direction', 'bullish')

    bid = float(contract.get('bid') or 0.0)
    ask = float(contract.get('ask') or 0.0)
    iv = float(contract.get('implied_volatility') or 0.0)
    volume = int(contract.get('volume') or 0)
    oi = int(contract.get('open_interest') or 0)
    spot = float(contract.get('spot_price') or 0.0)

    sector_etf = _TICKER_SECTOR_ETF.get(ticker, ticker if ticker in market_context.sector_rotation_rank else None)
    sector_rank = market_context.sector_rotation_rank.get(sector_etf, 6) if sector_etf else 6

    days_macro = market_context.days_to_macro_event

    pct_4w = float(candidate.get('pct_change_4w') or 0.0)
    proxy_zscore = pct_4w * 10  # rough proxy for volume z-score from 4w momentum

    # Approximate ATR via the 4w/1w return divergence — simple proxy for v1.
    pct_1w = float(candidate.get('pct_change_1w') or 0.0)
    atr_proxy = max(0.01, abs(pct_1w) * spot)
    prev_close_proxy = spot * (1 - pct_1w / 5.0)

    vix_z = float(candidate.get('vix_pct_vs_7d') or 0.0)
    vix_regime = candidate.get('vix_regime')

    values = {
        # Technical (5)
        'tech_setup_type': score_setup_type(candidate.get('setup_name')),
        'tech_setup_strength': score_setup_strength(candidate.get('setup_strength')),
        'tech_weekly_ribbon_agreement': score_weekly_ribbon_agreement(candidate.get('weekly_ribbon_agreement')),
        'tech_atr_pivot_position': score_atr_pivot_position(spot, prev_close_proxy, atr_proxy),
        'tech_volume_zscore': score_volume_zscore(proxy_zscore),
        # Vol/VIX (4)
        'vol_vix_regime': score_vix_regime(vix_regime),
        'vol_vix_zscore_30d': score_vix_zscore_30d(vix_z),
        'vol_vvix_level': score_vvix_level(market_context.vvix_score),
        'vol_iv_percentile': score_iv_percentile(iv),
        # Catalyst (3)
        'cat_days_to_earnings': score_days_to_earnings(days_to_earnings),
        'cat_days_to_macro_event': score_days_to_macro_event(days_macro),
        'cat_skip_window': score_skip_window(days_to_earnings, days_macro),
        # Macro/Breadth (4)
        'macro_spx_trend': score_spx_alignment(market_context.spx_trend_score),
        'macro_sector_rotation': score_sector_rotation(sector_rank, 11, setup_dir),
        'macro_dxy_trend': score_dxy_trend_pass_through(market_context.dxy_trend_score),
        'macro_yield_10y_direction': max(0.0, min(10.0, market_context.yield_10y_score)),
        # Liquidity (4)
        'liq_bid_ask_spread': score_bid_ask_spread(bid, ask),
        'liq_open_interest': score_open_interest(oi),
        'liq_oi_change_dod': 5.0,  # v1 placeholder; v2 will compute from cached prior chain
        'liq_volume_oi_ratio': score_volume_oi_ratio(volume, oi),
    }
    return FeatureScores(values=values)
