from __future__ import annotations
from datetime import date
import pandas as pd
from yahooquery import Ticker

from data.sources.base import DataSource, RawContract


_DTE_MIN, _DTE_MAX = 3, 45


class YahooQuerySource(DataSource):
    def fetch_spot(self, ticker: str, prefer_extended: bool = True) -> float:
        """Latest price, preferring pre/post market when in those sessions."""
        from data.sources.market_session import current_session, is_extended_hours

        t = Ticker(ticker)
        price_info = t.price.get(ticker, {}) if isinstance(t.price, dict) else {}

        if prefer_extended:
            session = current_session()
            if session == "pre_market":
                pre = price_info.get('preMarketPrice')
                if pre and float(pre) > 0:
                    return float(pre)
            elif session == "post_market":
                post = price_info.get('postMarketPrice')
                if post and float(post) > 0:
                    return float(post)

        return float(price_info.get('regularMarketPrice', 0.0))

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        period = '1y' if lookback_days > 180 else '6mo'
        df = Ticker(ticker).history(period=period)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol', drop_level=True)
        out = df['close'].astype(float)
        out.index = pd.to_datetime(out.index)
        return out

    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        period = '1y' if lookback_days > 180 else '6mo'
        df = Ticker(ticker).history(period=period)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol', drop_level=True)
        cols = {c.lower(): c for c in df.columns}
        keep = ['open', 'high', 'low', 'close', 'volume']
        out = pd.DataFrame({k: df[cols[k]].astype(float) for k in keep if k in cols})
        out.index = pd.to_datetime(out.index)
        return out.tail(lookback_days).dropna()

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
                # yahooquery MultiIndex is (symbol, expiration, optionType) — 3 levels.
                # strike lives in the 'strike' column, not in the index.
                if len(idx) == 4:
                    _symbol, exp_ts, opt_type_str, strike_idx = idx
                    strike = float(strike_idx)
                elif len(idx) == 3:
                    _symbol, exp_ts, opt_type_str = idx
                    strike = float(row.get('strike', 0))
                else:
                    continue
                exp = pd.Timestamp(exp_ts).date()
                dte = (exp - today).days
                if dte < _DTE_MIN or dte > _DTE_MAX:
                    continue
                option_type = 'put' if 'put' in str(opt_type_str).lower() else 'call'
                import math
                def _safe_int(v, default=0):
                    try:
                        f = float(v)
                        return default if math.isnan(f) else int(f)
                    except (TypeError, ValueError):
                        return default
                out.append(RawContract(
                    ticker=ticker,
                    expiration=exp,
                    strike=strike,
                    option_type=option_type,
                    bid=float(row.get('bid', 0) or 0),
                    ask=float(row.get('ask', 0) or 0),
                    volume=_safe_int(row.get('volume', 0)),
                    open_interest=_safe_int(row.get('openInterest', 0)),
                    implied_volatility=float(row.get('impliedVolatility', 0) or 0),
                    dte=dte,
                    spot_price=spot,
                ))
            except Exception:
                continue
        return out
