from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
from data.universe import get_universe
from data.earnings_calendar import has_earnings_soon
import os

MIN_PRICE = 10.0
MIN_AVG_VOLUME = 1_000_000
EARNINGS_BLACKOUT_DAYS = int(os.getenv('EARNINGS_BLACKOUT_DAYS', '5'))
TOP_N = 50


@dataclass
class TickerScore:
    ticker: str
    score: float           # 0-100
    current_price: float
    avg_volume: float
    iv_rank: float
    trend: str             # 'up', 'down', 'neutral'
    rsi: float
    current_iv: float


def compute_iv_rank(current_iv: float, price_history: pd.Series) -> float:
    """IV rank proxy: position of current_iv within 52-week RV range."""
    log_returns = np.log(price_history / price_history.shift(1)).dropna()
    rv_30d = log_returns.rolling(30).std() * np.sqrt(252)
    low_rv = rv_30d.min()
    high_rv = rv_30d.max()
    if high_rv <= low_rv:
        return 50.0
    rank = (current_iv - low_rv) / (high_rv - low_rv) * 100
    return float(np.clip(rank, 0.0, 100.0))


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI(14) from a price series."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    # When loss is 0 and gain > 0, RSI is 100 (pure uptrend); avoid 0/0 nan
    last_loss = loss.iloc[-1] if not loss.empty else np.nan
    last_gain = gain.iloc[-1] if not gain.empty else np.nan
    if pd.isna(last_loss) or pd.isna(last_gain):
        return 50.0
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1]) if not rsi.empty else 50.0
    return val if not np.isnan(val) else 50.0


def _compute_trend(prices: pd.Series) -> str:
    if len(prices) < 50:
        return 'neutral'
    sma20 = prices.iloc[-20:].mean()
    sma50 = prices.iloc[-50:].mean()
    if sma20 > sma50 * 1.01:
        return 'up'
    if sma20 < sma50 * 0.99:
        return 'down'
    return 'neutral'


def score_ticker(
    ticker: str,
    current_price: float,
    avg_volume: float,
    current_iv: float,
    price_history: pd.Series,
    vix_level: float,
    spy_trend: str,
) -> TickerScore:
    """Compute a 0-100 score for a ticker as an options selling candidate."""
    iv_rank = compute_iv_rank(current_iv, price_history)
    rsi = compute_rsi(price_history)
    trend = _compute_trend(price_history)

    # IV Rank (30%)
    iv_score = iv_rank * 0.30

    # Trend alignment (25%): puts prefer uptrend, calls prefer downtrend
    trend_score = {'up': 75.0, 'neutral': 50.0, 'down': 25.0}.get(trend, 50.0) * 0.25

    # Momentum (20%): RSI 40-65 = good for premium selling
    if 40 <= rsi <= 65:
        momentum_score = 80.0
    elif rsi < 30 or rsi > 80:
        momentum_score = 20.0
    else:
        momentum_score = 50.0
    momentum_score *= 0.20

    # Liquidity (15%): based on avg volume
    if avg_volume >= 5_000_000:
        liq = 100.0
    elif avg_volume >= 1_000_000:
        liq = 70.0
    elif avg_volume >= 500_000:
        liq = 40.0
    else:
        liq = 10.0
    liq_score = liq * 0.15

    # Market regime (10%): prefer elevated but not extreme VIX
    if 15 <= vix_level <= 30:
        regime = 80.0
    elif vix_level > 35:
        regime = 30.0
    else:
        regime = 50.0
    regime_score = regime * 0.10

    total = iv_score + trend_score + momentum_score + liq_score + regime_score

    # Hard cap: options on illiquid tickers are not practically tradeable
    if avg_volume < MIN_AVG_VOLUME:
        total = min(total, 35.0)

    return TickerScore(
        ticker=ticker,
        score=round(total, 2),
        current_price=current_price,
        avg_volume=avg_volume,
        iv_rank=iv_rank,
        trend=trend,
        rsi=rsi,
        current_iv=current_iv,
    )


def screen_universe(top_n: int = TOP_N) -> list[TickerScore]:
    """
    Download market data for the focused universe, apply hard filters,
    score each ticker, and return the top N candidates.
    """
    tickers = get_universe()

    # Fetch VIX and SPY for market regime context
    try:
        vix_data = yf.download('^VIX', period='5d', progress=False)['Close']
        vix_level = float(vix_data.iloc[-1])
        spy_data = yf.download('SPY', period='60d', progress=False)['Close']
        spy_trend = _compute_trend(spy_data.squeeze())
    except Exception:
        vix_level = 20.0
        spy_trend = 'neutral'

    scored = []
    for ticker in tickers:
        try:
            hist = yf.download(ticker, period='1y', progress=False)['Close'].squeeze()
            if hist is None or len(hist) < 60:
                continue

            info = yf.Ticker(ticker).fast_info
            current_price = float(getattr(info, 'last_price', 0) or hist.iloc[-1])
            avg_volume = float(getattr(info, 'three_month_average_volume', 0) or 0)

            if current_price < MIN_PRICE:
                continue
            if avg_volume < MIN_AVG_VOLUME:
                continue
            if has_earnings_soon(ticker, days=EARNINGS_BLACKOUT_DAYS):
                continue

            # Use 30-day realized vol as IV proxy
            log_returns = np.log(hist / hist.shift(1)).dropna()
            current_iv = float(log_returns.iloc[-30:].std() * np.sqrt(252))

            ts = score_ticker(
                ticker=ticker,
                current_price=current_price,
                avg_volume=avg_volume,
                current_iv=current_iv,
                price_history=hist,
                vix_level=vix_level,
                spy_trend=spy_trend,
            )
            scored.append(ts)
        except Exception:
            continue

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]
