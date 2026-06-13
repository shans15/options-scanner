from __future__ import annotations
import logging
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "macro"
logger = logging.getLogger(__name__)


class YahooMacroSource:
    """Yahoo Finance public client for macro indicators (DXY, VIX, SPX, gold).
    Daily resolution. Cached to Parquet."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def fetch_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch OHLCV history for a yahoo ticker.
        ticker examples: 'DX-Y.NYB' (DXY), '^VIX' (VIX), '^GSPC' (SPX), 'GC=F' (gold).
        Returns DataFrame with UTC DatetimeIndex and columns ['open','high','low','close','volume']."""
        cache_key = f"yahoo_{ticker.replace('=','_').replace('^','_').replace('.','_')}_{period}"
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            logger.info("Loading from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty:
            logger.warning("Yahoo returned empty for %s", ticker)
            return pd.DataFrame(columns=['open','high','low','close','volume'])

        df = df.rename(columns={c: c.lower() for c in df.columns})
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df.index.name = 'datetime'
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df.to_parquet(cache_file)
        return df
