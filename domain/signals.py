from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class Regime:
    ticker: str
    rv_30: float
    iv_atm: float
    rv_iv_ratio: float
    favored: list[Literal['sell', 'buy']]


def _annualized_rv_30(log_returns: pd.Series) -> float:
    tail = log_returns.dropna().tail(30)
    if len(tail) < 5:
        return 0.0
    return float(tail.std() * np.sqrt(252))


def _is_traded(c) -> bool:
    bid = getattr(c, 'bid', 0.0) or 0.0
    ask = getattr(c, 'ask', 0.0) or 0.0
    return (bid + ask) > 0


_ATM_BAND_PCT = 0.05  # only contracts within ±5% of spot count as ATM
_MIN_ATM_CONTRACTS = 3


def _atm_iv(chain) -> float:
    """Mean IV of strikes within ±5% of spot in the nearest *liquid* expiration.

    Off-hours yahooquery snapshots leave ATM contracts with bid/ask=0 while deep
    ITM/OTM contracts (the ones that traded earlier) keep their last quotes — and
    those off-money quotes are skew-inflated. We require at least 3 liquid contracts
    within ±5% of spot before trusting an expiration's IV; otherwise we keep walking
    further out in DTE. Return 0.0 when no near-ATM liquidity exists anywhere —
    the regime gate then falls back to neutral.
    """
    if not chain:
        return 0.0
    tradeable = [c for c in chain if _is_traded(c) and getattr(c, 'implied_volatility', 0.0) > 0]
    if not tradeable:
        return 0.0
    spot = getattr(tradeable[0], 'spot_price', 0.0) or 0.0
    if spot <= 0:
        return 0.0
    near_atm = [c for c in tradeable if abs(c.strike - spot) / spot <= _ATM_BAND_PCT]
    if len(near_atm) < _MIN_ATM_CONTRACTS:
        return 0.0
    dtes_sorted = sorted({getattr(c, 'dte', 9999) for c in near_atm})
    for front_dte in dtes_sorted:
        front_month = [c for c in near_atm if getattr(c, 'dte', 9999) == front_dte]
        if len(front_month) < _MIN_ATM_CONTRACTS:
            continue
        atm_sorted = sorted(front_month, key=lambda c: abs(c.strike - spot))[:6]
        ivs = [c.implied_volatility for c in atm_sorted]
        return float(np.mean(ivs))
    return 0.0


def compute_regime(ticker: str, log_returns: pd.Series, chain: list) -> Regime:
    rv_30 = _annualized_rv_30(log_returns)
    iv_atm = _atm_iv(chain)

    if iv_atm <= 0 or rv_30 <= 0:
        return Regime(ticker=ticker, rv_30=rv_30, iv_atm=iv_atm, rv_iv_ratio=1.0,
                      favored=['sell', 'buy'])

    ratio = rv_30 / iv_atm
    if ratio < 0.8:
        favored = ['sell']
    elif ratio > 1.2:
        favored = ['buy']
    else:
        favored = ['sell', 'buy']

    return Regime(ticker=ticker, rv_30=rv_30, iv_atm=iv_atm, rv_iv_ratio=ratio, favored=favored)
