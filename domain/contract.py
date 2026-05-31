from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class Contract:
    ticker: str
    expiration: date
    strike: float
    option_type: Literal['put', 'call']
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    dte: int
    spot_price: float
