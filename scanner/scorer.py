from __future__ import annotations
import os
from typing import Optional
import numpy as np

MIN_COMPOSITE_SCORE = int(os.getenv('MIN_COMPOSITE_SCORE', '65'))
WATCHLIST_FLOOR = 50


def compute_composite_score(contract: dict, pop_result: object) -> float:
    """
    Compute 0-100 composite trade score.
    Weights: PoP 30%, EV 20%, Premium/Margin 15%, Liquidity 15%, Spread 10%, IV Rank 10%
    """
    def _get(obj: object, key: str, default: float = 0.0) -> float:
        if hasattr(obj, key):
            val = getattr(obj, key)
            return float(val) if val is not None else default
        if isinstance(obj, dict):
            val = obj.get(key, default)
            return float(val) if val is not None else default
        return default

    pop_blended = _get(pop_result, 'pop_blended')
    ev = _get(pop_result, 'expected_value')
    margin = _get(pop_result, 'margin_estimate') or 1.0
    iv_rank = _get(pop_result, 'iv_rank', 50.0)

    premium = float(contract.get('mid', 0) or 0)
    volume = int(contract.get('volume', 0) or 0)
    oi = int(contract.get('open_interest', 0) or 0)
    bid = float(contract.get('bid', 0) or 0)
    ask = float(contract.get('ask', 0) or 0)
    mid = float(contract.get('mid', 1) or 1)

    # 1. PoP component (30%): scale 0.70-1.0 -> 0-100
    pop_score = float(np.clip((pop_blended - 0.70) / 0.30 * 100, 0, 100)) * 0.30

    # 2. EV component (20%): EV/premium ratio
    ev_ratio = ev / premium if premium > 0 else 0
    ev_score = float(np.clip(ev_ratio * 100, 0, 100)) * 0.20

    # 3. Premium/Margin ratio (15%)
    pm_ratio = (premium * 100) / margin if margin > 0 else 0
    pm_score = float(np.clip(pm_ratio * 200, 0, 100)) * 0.15

    # 4. Liquidity (15%)
    if volume >= 1000 and oi >= 5000:
        liq = 100.0
    elif volume >= 300 and oi >= 1000:
        liq = 70.0
    elif volume >= 100 and oi >= 500:
        liq = 40.0
    else:
        liq = 10.0
    liq_score = liq * 0.15

    # 5. Bid/ask tightness (10%)
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
    spread_score = float(np.clip((1 - spread_pct / 0.20) * 100, 0, 100)) * 0.10

    # 6. IV Rank (10%)
    iv_score = float(np.clip(iv_rank, 0, 100)) * 0.10

    total = pop_score + ev_score + pm_score + liq_score + spread_score + iv_score
    return round(float(np.clip(total, 0, 100)), 2)


def assign_decision(score: float, hard_filter_passed: bool) -> str:
    """Return 'TRADE', 'WATCHLIST', or 'NO TRADE'."""
    if not hard_filter_passed:
        return 'NO TRADE'
    if score >= MIN_COMPOSITE_SCORE:
        return 'TRADE'
    if score >= WATCHLIST_FLOOR:
        return 'WATCHLIST'
    return 'NO TRADE'


def _build_reason_for(contract: dict, pop_result: object) -> str:
    reasons = []
    iv_rank = getattr(pop_result, 'iv_rank', None) or 0.0
    if iv_rank > 60:
        reasons.append(f"High IV rank ({iv_rank:.0f})")
    pop = getattr(pop_result, 'pop_blended', 0.0) or 0.0
    if pop > 0.80:
        reasons.append(f"Strong PoP ({pop:.0%})")
    if contract.get('volume', 0) > 500:
        reasons.append("Liquid chain")
    dte = contract.get('dte', 0)
    if 7 <= dte <= 21:
        reasons.append(f"Optimal DTE ({dte})")
    return ', '.join(reasons) if reasons else 'Meets all quantitative criteria'


def _build_reason_against(contract: dict, pop_result: object) -> str:
    reasons = []
    strategy = contract.get('strategy', '')
    if strategy == 'naked_call':
        reasons.append('Theoretically unlimited upside risk')
    dte = contract.get('dte', 0)
    if dte < 7:
        reasons.append('Very short DTE — gamma risk elevated')
    iv = contract.get('implied_volatility', 0)
    if iv and iv > 0.60:
        reasons.append('Very high IV — elevated uncertainty')
    stress = getattr(pop_result, 'stress', None)
    if stress and getattr(stress, 'stress_2sd', 0) < -3.0:
        reasons.append('2SD stress scenario shows significant loss')
    return ', '.join(reasons) if reasons else 'Standard tail risk applies'


def _build_stop_trigger(contract: dict, pop_result: object) -> str:
    premium = float(contract.get('mid', 0) or 0)
    strategy = contract.get('strategy', '')
    strike = float(contract.get('strike', 0) or 0)
    stop_loss = round(premium * 2, 2)
    if strategy == 'naked_put':
        price_trigger = round(strike * 1.005, 2)
        return f"Close if loss > ${stop_loss} (2x premium) OR stock closes below ${price_trigger}"
    else:
        price_trigger = round(strike * 0.995, 2)
        return f"Close if loss > ${stop_loss} (2x premium) OR stock closes above ${price_trigger}"


def build_scan_result(
    contract: dict,
    pop_result: object,
    hard_filter_passed: bool,
    filter_reason: str,
) -> dict:
    """Assemble the full ScanResult dict from contract + PopResult."""
    score = compute_composite_score(contract, pop_result)
    decision = assign_decision(score, hard_filter_passed)
    stress = getattr(pop_result, 'stress', None)

    return {
        # Contract fields
        'ticker': contract.get('ticker'),
        'strategy': contract.get('strategy'),
        'expiration': contract.get('expiration'),
        'strike': contract.get('strike'),
        'bid': contract.get('bid'),
        'ask': contract.get('ask'),
        'mid': contract.get('mid'),
        'premium': contract.get('mid'),
        'delta': contract.get('delta'),
        'gamma': contract.get('gamma'),
        'theta': contract.get('theta'),
        'vega': contract.get('vega'),
        'IV': contract.get('implied_volatility'),
        'dte': contract.get('dte'),
        'spot_price': contract.get('spot_price'),
        # Quant fields
        'IV_rank': getattr(pop_result, 'iv_rank', None),
        'PoP_delta': getattr(pop_result, 'pop_delta', None),
        'PoP_BS': getattr(pop_result, 'pop_bs', None),
        'PoP_historical': getattr(pop_result, 'pop_historical', None),
        'PoP_GARCH_MC': getattr(pop_result, 'pop_garch_mc', None),
        'PoP_blended': getattr(pop_result, 'pop_blended', None),
        'expected_value': getattr(pop_result, 'expected_value', None),
        'breakeven': getattr(pop_result, 'breakeven', None),
        'margin_estimate': getattr(pop_result, 'margin_estimate', None),
        'stress_1SD': getattr(stress, 'stress_1sd', None) if stress else None,
        'stress_2SD': getattr(stress, 'stress_2sd', None) if stress else None,
        'stress_expiry': getattr(stress, 'stress_expiry', None) if stress else None,
        # Scoring
        'composite_score': score,
        'hard_filter_passed': hard_filter_passed,
        'hard_filter_reason': filter_reason,
        'decision': decision,
        # Narrative
        'reason_for': _build_reason_for(contract, pop_result),
        'reason_against': _build_reason_against(contract, pop_result),
        'stop_trigger': _build_stop_trigger(contract, pop_result),
    }
