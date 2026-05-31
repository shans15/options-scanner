from __future__ import annotations
from datetime import date
from typing import Optional

from domain.contract import Contract
from domain.greeks import compute_greeks
from data.sources.base import RawContract


def to_contract(raw: RawContract, spot: float, r: float, today: date) -> Optional[Contract]:
    dte = (raw.expiration - today).days
    if dte < 3 or dte > 45:
        return None
    if raw.bid + raw.ask <= 0:
        return None
    mid = (raw.bid + raw.ask) / 2.0
    T = max(dte / 252, 1e-9)
    flag = 'c' if raw.option_type == 'call' else 'p'
    g = compute_greeks(S=spot, K=raw.strike, T=T, r=r, sigma=raw.implied_volatility, flag=flag)

    return Contract(
        ticker=raw.ticker, expiration=raw.expiration, strike=raw.strike,
        option_type=raw.option_type, bid=raw.bid, ask=raw.ask, mid=mid,
        volume=raw.volume, open_interest=raw.open_interest,
        implied_volatility=raw.implied_volatility,
        delta=g['delta'], gamma=g['gamma'], theta=g['theta'], vega=g['vega'],
        dte=dte, spot_price=spot,
    )
