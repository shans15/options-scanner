"""Per-scan market context — computed once per watchlist run."""
from __future__ import annotations
from datetime import date
from typing import Literal
import pandas as pd

from data.sources.macro_calendar import days_to_next_event
from domain.aplus.types import MarketContext


_SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC']


def score_spx_trend(spy_closes: pd.Series, setup_direction: str) -> float:
    """10 if SPY > 50EMA > 200EMA AND setup is bullish (or inverse for bearish);
    4 if mixed; 0 if SPY trend opposes setup direction."""
    if len(spy_closes) < 50:
        return 5.0
    ema50 = spy_closes.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = (
        spy_closes.ewm(span=200, adjust=False).mean().iloc[-1]
        if len(spy_closes) >= 200
        else float(spy_closes.iloc[0])  # first bar as long-run anchor when < 200 bars
    )
    last = float(spy_closes.iloc[-1])

    bullish_stack = last > ema50 > ema200
    bearish_stack = last < ema50 < ema200

    if setup_direction == 'bullish':
        if bullish_stack:
            return 10.0
        if bearish_stack:
            return 0.0
        return 4.0
    if setup_direction == 'bearish':
        if bearish_stack:
            return 10.0
        if bullish_stack:
            return 0.0
        return 4.0
    return 5.0


def score_dxy_trend(dxy_closes: pd.Series, setup_direction: str) -> float:
    """DXY direction supports bullish stock setups when DXY is falling
    (USD weakness lifts multinationals); opposes when DXY is rising."""
    if len(dxy_closes) < 20:
        return 5.0
    sma20 = float(dxy_closes.rolling(20).mean().iloc[-1])
    last = float(dxy_closes.iloc[-1])
    dxy_falling = last < sma20 * 0.99
    dxy_rising = last > sma20 * 1.01
    if setup_direction == 'bullish':
        if dxy_falling:
            return 10.0
        if dxy_rising:
            return 2.0
        return 5.0
    if setup_direction == 'bearish':
        if dxy_rising:
            return 10.0
        if dxy_falling:
            return 2.0
        return 5.0
    return 5.0


def score_yield_10y(yield_closes: pd.Series, sector: str, setup_direction: str) -> float:
    """Map yield direction × sector × setup direction.
    Rising yields lift XLF/XLE, hurt XLK/XLRE; falling yields the reverse."""
    if len(yield_closes) < 20:
        return 5.0
    sma20 = float(yield_closes.rolling(20).mean().iloc[-1])
    last = float(yield_closes.iloc[-1])
    rising = last > sma20 * 1.02
    falling = last < sma20 * 0.98
    yields_helpful = {'XLF': 'rising', 'XLE': 'rising', 'XLK': 'falling', 'XLRE': 'falling'}
    helpful = yields_helpful.get(sector)
    if helpful is None:
        return 5.0
    if (helpful == 'rising' and rising) or (helpful == 'falling' and falling):
        return 10.0 if setup_direction == 'bullish' else 2.0
    if (helpful == 'rising' and falling) or (helpful == 'falling' and rising):
        return 2.0 if setup_direction == 'bullish' else 10.0
    return 5.0


def score_vvix(vvix_level: float) -> float:
    """10 if VVIX < 90 (stable vol-of-vol);
    scaling down to 3 if VVIX > 120 (unstable vol regime)."""
    if vvix_level < 90:
        return 10.0
    if vvix_level > 120:
        return 3.0
    # Linear interpolation between 90 and 120
    return 10.0 - (vvix_level - 90) / 30.0 * 7.0


def rank_sectors(returns_1w: dict[str, float]) -> dict[str, int]:
    """Assign rank 1..N to each sector by 1-week return (1 = best)."""
    sorted_etfs = sorted(returns_1w.items(), key=lambda kv: -kv[1])
    return {etf: i + 1 for i, (etf, _) in enumerate(sorted_etfs)}


def build_market_context(today: date, setup_direction: str = 'bullish') -> MarketContext:
    """Pull live macro data and build a MarketContext snapshot.

    NOTE: setup_direction is needed for SPX/DXY trend scoring;
    we default to bullish here and let the caller recompute
    direction-specific scores if needed at the candidate level.
    """
    spy_closes = _fetch_macro_series('SPY')
    dxy_closes = _fetch_macro_series('DX-Y.NYB')
    yield_closes = _fetch_macro_series('^TNX')
    vvix_level = _fetch_vvix()
    sector_returns = _fetch_sector_returns()

    return MarketContext(
        spx_trend_score=score_spx_trend(spy_closes, setup_direction),
        sector_rotation_rank=rank_sectors(sector_returns),
        dxy_trend_score=score_dxy_trend(dxy_closes, setup_direction),
        yield_10y_score=5.0,   # per-sector; computed lazily at feature time
        vvix_score=score_vvix(vvix_level),
        days_to_macro_event=days_to_next_event(today),
    )


def _fetch_macro_series(ticker: str) -> pd.Series:
    """Fetch ~60 days of daily closes for a ticker. Falls back to empty Series on failure."""
    from data.sources.yahoo_macro_source import YahooMacroSource
    try:
        df = YahooMacroSource().fetch_history(ticker, period='3mo')
        return df['close'] if not df.empty and 'close' in df.columns else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def _fetch_vvix() -> float:
    """Fetch most recent VVIX value (^VVIX on Yahoo). Returns neutral 100 on failure."""
    series = _fetch_macro_series('^VVIX')
    return float(series.iloc[-1]) if len(series) > 0 else 100.0


def _fetch_sector_returns() -> dict[str, float]:
    """1-week return per sector ETF."""
    out: dict[str, float] = {}
    for etf in _SECTOR_ETFS:
        series = _fetch_macro_series(etf)
        if len(series) >= 5:
            out[etf] = float(series.iloc[-1] / series.iloc[-5] - 1)
        else:
            out[etf] = 0.0
    return out
