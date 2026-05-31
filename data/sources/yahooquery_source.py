from __future__ import annotations
from datetime import date
import pandas as pd
from yahooquery import Ticker

from data.sources.base import DataSource, RawContract


_DTE_MIN, _DTE_MAX = 3, 45


class YahooQuerySource(DataSource):
    def fetch_spot(self, ticker: str) -> float:
        t = Ticker(ticker)
        price_info = t.price.get(ticker, {}) if isinstance(t.price, dict) else {}
        return float(price_info.get('regularMarketPrice', 0.0))

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        period = '1y' if lookback_days > 180 else '6mo'
        df = Ticker(ticker).history(period=period)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol', drop_level=True)
        return df['close'].astype(float)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        t = Ticker(ticker)
        spot = self.fetch_spot(ticker)
        df = t.option_chain
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        today = date.today()
        out: list[RawContract] = []
        for idx, row in df.iterrows():
            try:
                _symbol, exp_ts, opt_type_str, strike = idx
                exp = pd.Timestamp(exp_ts).date()
                dte = (exp - today).days
                if dte < _DTE_MIN or dte > _DTE_MAX:
                    continue
                option_type = 'put' if 'put' in str(opt_type_str).lower() else 'call'
                out.append(RawContract(
                    ticker=ticker,
                    expiration=exp,
                    strike=float(strike),
                    option_type=option_type,
                    bid=float(row.get('bid', 0) or 0),
                    ask=float(row.get('ask', 0) or 0),
                    volume=int(row.get('volume', 0) or 0),
                    open_interest=int(row.get('openInterest', 0) or 0),
                    implied_volatility=float(row.get('impliedVolatility', 0) or 0),
                    dte=dte,
                    spot_price=spot,
                ))
            except Exception:
                continue
        return out
