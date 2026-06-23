from __future__ import annotations
from datetime import date
import pandas as pd
import yfinance as yf

from data.sources.base import DataSource, RawContract


_DTE_MIN, _DTE_MAX = 3, 45


class YfinanceSource(DataSource):
    def fetch_spot(self, ticker: str, prefer_extended: bool = True) -> float:
        """Latest price, preferring pre/post market when in those sessions."""
        from data.sources.market_session import current_session, is_extended_hours

        t = yf.Ticker(ticker)

        if prefer_extended:
            session = current_session()
            try:
                info = t.info
                if session == "pre_market":
                    pre = info.get('preMarketPrice')
                    if pre and float(pre) > 0:
                        return float(pre)
                elif session == "post_market":
                    post = info.get('postMarketPrice')
                    if post and float(post) > 0:
                        return float(post)
            except Exception:
                pass  # info call can fail; fall through to fast_info

        return float(t.fast_info.last_price or 0.0)

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        period = '1y' if lookback_days > 180 else '6mo'
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty or 'Close' not in df.columns:
            return pd.Series(dtype=float)
        out = df['Close'].astype(float)
        out.index = pd.to_datetime(out.index)
        return out

    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        period = '1y' if lookback_days > 180 else '6mo'
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename)
        df.index = pd.to_datetime(df.index)
        keep = ['open', 'high', 'low', 'close', 'volume']
        return df[[c for c in keep if c in df.columns]].astype(float).tail(lookback_days).dropna()

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
                    try:
                        # NaN is truthy so `NaN or 0` stays NaN; use explicit nan-guard
                        def _safe_int(v, default=0):
                            import math
                            try:
                                f = float(v)
                                return default if math.isnan(f) else int(f)
                            except (TypeError, ValueError):
                                return default
                        out.append(RawContract(
                            ticker=ticker,
                            expiration=exp,
                            strike=float(row.get('strike', 0) or 0),
                            option_type=opt_type,
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
