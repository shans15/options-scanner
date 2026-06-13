"""Public API for computing fair probabilities on Kalshi BTC markets.

This module provides detect_edge(), the single entry point used by
scripts/kalshi_btc_watchlist.py and scripts/kalshi_btc_backtest.py.

detect_edge() orchestrates all three strategies:
  A:   vol-model only (vol_model_estimator + adjustments)
  B:   research-paper ensemble (GARCH + bootstrap + RV-scaling + adjustments)
  A+B: combined (all 4 estimators + adjustments)

The canonical fair probability is fair_prob_AB.
Edge = canonical_fair_prob - market_implied_prob.

# STUB — AGENT 2 WILL OVERWRITE parameter tuning; this implements the full
# logic required by Agent 3's scripts.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pandas as pd

from domain.prediction.market import KalshiMarket, PredictionMarketEdge
from engine.prediction.ensemble import (
    compute_strategy_A,
    compute_strategy_B,
    compute_strategy_AB,
    count_estimator_agreement,
)


# Kalshi fee rate — must be cleared by gross edge for a net-positive trade
_KALSHI_FEE_RATE = 0.07  # 7% of gross winnings


def _expected_value_per_dollar(fair_prob: float, market_implied_prob: float) -> float:
    """Kelly-like EV approximation.

    If we buy YES at price p (market_implied_prob):
      EV = fair_prob * (1 - p) - (1 - fair_prob) * p - fee
         = fair_prob - p - fair_prob * p + fair_prob * p - fee
         = fair_prob - p - fee * fair_prob   (approx, fee on winnings only)

    For a $1 position:
      EV per dollar = fair_prob * (1 / market_implied_prob - 1) * (1 - fee) - 1 * (1 - fair_prob)
    Simplified (Kalshi binary resolution):
      EV = (fair_prob - market_implied_prob) / market_implied_prob  (approx, no fee)
    Use: EV = fair_prob * (1 - _KALSHI_FEE_RATE) - market_implied_prob
    """
    if market_implied_prob <= 0 or market_implied_prob >= 1:
        return 0.0
    # Probability-weighted payoff net of fee vs cost
    return fair_prob * (1.0 - _KALSHI_FEE_RATE) - market_implied_prob


def detect_edge(
    market: KalshiMarket,
    btc_spot: float,
    btc_minute_returns_last_60: pd.Series,
    btc_hourly_returns_last_30d: pd.Series,
    vol_proba: float,
    historical_vol_quantiles: dict[float, float],
    last_hour_return: float,
    hourly_sigma: float,
    suggested_position_dollars: float,
    min_edge: float = 0.05,
) -> Optional[PredictionMarketEdge]:
    """Compute fair probabilities via all 3 strategies and identify mispricing.

    Returns a PredictionMarketEdge if |edge| >= min_edge, else None.

    Parameters
    ----------
    market : KalshiMarket
        The parsed Kalshi market.
    btc_spot : float
        Current BTC spot price.
    btc_minute_returns_last_60 : pd.Series
        1-minute log returns over last 60 minutes (for RV estimator).
    btc_hourly_returns_last_30d : pd.Series
        Hourly log returns over last 30 days (for GARCH + bootstrap).
    vol_proba : float
        P(vol expansion) from the trained vol model.
    historical_vol_quantiles : dict[float, float]
        {0.25: σ_low, 0.5: σ_med, 0.75: σ_high} from rolling hourly vol.
    last_hour_return : float
        Log return of the most recent hour (for mean-reversion adjustment).
    hourly_sigma : float
        Estimated hourly σ (from recent rolling vol).
    suggested_position_dollars : float
        Dollars the caller suggests for this position.
    min_edge : float
        Minimum |edge| to return a PredictionMarketEdge (default 5%).

    Returns
    -------
    PredictionMarketEdge or None
    """
    now = datetime.now()
    try:
        now = market.expiration.__class__.now(tz=market.expiration.tzinfo)
    except Exception:
        pass

    hours_to_expiry = (market.expiration - now).total_seconds() / 3600.0
    if hours_to_expiry <= 0:
        return None

    market_prob = market.market_implied_prob
    if market_prob <= 0 or market_prob >= 1:
        return None

    # --- Strategy A ---
    out_A = compute_strategy_A(
        spot=btc_spot,
        strike=market.strike,
        hours_to_expiry=hours_to_expiry,
        vol_proba=vol_proba,
        historical_vol_quantiles=historical_vol_quantiles,
        expiration_utc=market.expiration,
        last_hour_return=last_hour_return,
        hourly_sigma=hourly_sigma,
    )

    # --- Strategy B ---
    out_B = compute_strategy_B(
        spot=btc_spot,
        strike=market.strike,
        hours_to_expiry=hours_to_expiry,
        hourly_returns=btc_hourly_returns_last_30d,
        minute_returns_last_60=btc_minute_returns_last_60,
        expiration_utc=market.expiration,
        last_hour_return=last_hour_return,
        hourly_sigma=hourly_sigma,
    )

    # --- Strategy A+B ---
    out_AB = compute_strategy_AB(
        spot=btc_spot,
        strike=market.strike,
        hours_to_expiry=hours_to_expiry,
        vol_proba=vol_proba,
        historical_vol_quantiles=historical_vol_quantiles,
        hourly_returns=btc_hourly_returns_last_30d,
        minute_returns_last_60=btc_minute_returns_last_60,
        expiration_utc=market.expiration,
        last_hour_return=last_hour_return,
        hourly_sigma=hourly_sigma,
    )

    canonical = out_AB.fair_prob_adjusted
    edge = canonical - market_prob

    if abs(edge) < min_edge:
        return None

    # Estimator agreement: use AB's constituent estimates
    n_agree, n_total = count_estimator_agreement(
        out_AB.constituent_estimates, market_prob
    )

    ev_per_dollar = _expected_value_per_dollar(canonical, market_prob)

    # For 'below' markets, invert the fair probability
    # (market resolves YES if BTC < strike; fair_prob computed as P(BTC > strike))
    fair_A = out_A.fair_prob_adjusted
    fair_B = out_B.fair_prob_adjusted
    fair_AB = canonical

    if market.side == "below":
        fair_A = 1.0 - fair_A
        fair_B = 1.0 - fair_B
        fair_AB = 1.0 - fair_AB
        canonical = fair_AB
        edge = canonical - market_prob

        if abs(edge) < min_edge:
            return None

    return PredictionMarketEdge(
        market=market,
        fair_prob_A=fair_A,
        fair_prob_B=fair_B,
        fair_prob_AB=fair_AB,
        canonical_fair_prob=canonical,
        edge=edge,
        expected_value_per_dollar=ev_per_dollar,
        estimator_agreement=(n_agree, n_total),
        suggested_position_dollars=suggested_position_dollars,
        adjustments_applied=out_AB.adjustments_applied,
    )
