from __future__ import annotations
from datetime import date
import pandas as pd
import yfinance as yf

from data.sources.base import DataSource, RawContract


_DTE_MIN, _DTE_MAX = 3, 45


class YfinanceSource(DataSource):
    def fetch_spot(self, ticker: str) -> float:
        return float(yf.Ticker(ticker).fast_info.last_price or 0.0)

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        period = '1y' if lookback_days > 180 else '6mo'
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty or 'Close' not in df.columns:
            return pd.Series(dtype=float)
        return df['Close'].astype(float)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        tk = yf.Ticker(ticker)
        try:
            expirations = list(tk.options or [])
        except Exception:
            return []
        spot = self.fetch_spot(ticker)
        today = date.today()
        out: list[RawContract] = []
        for exp_str in expirations:
            try:
                exp = date.fromisoformat(exp_str)
            except ValueError:
                continue
            dte = (exp - today).days
            if dte < _DTE_MIN or dte > _DTE_MAX:
                continue
            try:
                chain = tk.option_chain(exp_str)
            except Exception:
                continue
            for df, opt_type in [(chain.puts, 'put'), (chain.calls, 'call')]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    out.append(RawContract(
                        ticker=ticker,
                        expiration=exp,
                        strike=float(row.get('strike', 0)),
                        option_type=opt_type,
                        bid=float(row.get('bid', 0) or 0),
                        ask=float(row.get('ask', 0) or 0),
                        volume=int(row.get('volume', 0) or 0),
                        open_interest=int(row.get('openInterest', 0) or 0),
                        implied_volatility=float(row.get('impliedVolatility', 0) or 0),
                        dte=dte,
                        spot_price=spot,
                    ))
        return out
