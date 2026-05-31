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


def _atm_iv(chain) -> float:
    if not chain:
        return 0.0
    by_dte = sorted(chain, key=lambda c: getattr(c, 'dte', 9999))
    if not by_dte:
        return 0.0
    front_dte = by_dte[0].dte
    front_month = [c for c in chain if getattr(c, 'dte', 9999) == front_dte]
    if not front_month:
        return 0.0
    spot = getattr(front_month[0], 'spot_price', 0.0) or 0.0
    if spot <= 0:
        ivs = [c.implied_volatility for c in front_month if c.implied_volatility > 0]
        return float(np.mean(ivs)) if ivs else 0.0
    atm_sorted = sorted(front_month, key=lambda c: abs(c.strike - spot))[:6]
    ivs = [c.implied_volatility for c in atm_sorted if c.implied_volatility > 0]
    return float(np.mean(ivs)) if ivs else 0.0


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
