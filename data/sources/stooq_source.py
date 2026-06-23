from __future__ import annotations
import pandas as pd

from data.sources.base import DataSource, RawContract


class StooqSource(DataSource):
    def _url(self, ticker: str) -> str:
        return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

    def fetch_spot(self, ticker: str, prefer_extended: bool = True) -> float:
        # Stooq returns last available daily close. No intraday support.
        df = pd.read_csv(self._url(ticker))
        if df.empty or 'Close' not in df.columns:
            raise RuntimeError(f"stooq returned no Close column for {ticker}")
        return float(df['Close'].iloc[-1])

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        df = pd.read_csv(self._url(ticker))
        if df.empty or 'Close' not in df.columns:
            raise RuntimeError(f"stooq returned no Close column for {ticker}")
        return df['Close'].astype(float).tail(lookback_days).reset_index(drop=True)

    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        df = pd.read_csv(self._url(ticker))
        if df.empty:
            raise RuntimeError(f"stooq returned empty for {ticker}")
        rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        keep = ['open', 'high', 'low', 'close', 'volume']
        out = df[[c for c in keep if c in df.columns]].astype(float).tail(lookback_days)
        return out.dropna()

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        return []
