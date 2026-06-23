from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal
import pandas as pd


@dataclass(frozen=True)
class RawContract:
    ticker: str
    expiration: date
    strike: float
    option_type: Literal['put', 'call']
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    dte: int
    spot_price: float


class DataSource(ABC):
    @abstractmethod
    def fetch_spot(self, ticker: str, prefer_extended: bool = True) -> float: ...

    @abstractmethod
    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series: ...

    @abstractmethod
    def fetch_option_chain(self, ticker: str) -> list[RawContract]: ...

    @abstractmethod
    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Return DataFrame with lowercase columns: open, high, low, close, volume.
        Index is the date (ascending). Length up to lookback_days; may be less
        if the ticker has shorter history."""
