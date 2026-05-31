from __future__ import annotations
import pandas as pd

from data.sources.base import DataSource, RawContract


class StooqSource(DataSource):
    def _url(self, ticker: str) -> str:
        return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

    def fetch_spot(self, ticker: str) -> float:
        df = pd.read_csv(self._url(ticker))
        if df.empty or 'Close' not in df.columns:
            raise RuntimeError(f"stooq returned no Close column for {ticker}")
        return float(df['Close'].iloc[-1])

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        df = pd.read_csv(self._url(ticker))
        if df.empty or 'Close' not in df.columns:
            raise RuntimeError(f"stooq returned no Close column for {ticker}")
        return df['Close'].astype(float).tail(lookback_days).reset_index(drop=True)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        return []
