from __future__ import annotations

"""Ticker universe configuration for options trading.

Provides a focused list of high-liquidity tickers suitable for options strategies,
with an exclusion list for meme stocks.
"""

# High-liquidity tickers with deep options markets for strategy execution
FOCUSED_UNIVERSE = [
    # Index ETFs (deepest options liquidity)
    'SPY', 'QQQ', 'IWM',
    # Magnificent 7
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Bonus: liquid macro ETFs with low gap risk
    'GLD', 'TLT',
]

# Meme stocks excluded from universe for volatility/risk management
MEME_EXCLUSION_LIST = ['GME', 'AMC']


def get_universe() -> list[str]:
    """Return the focused high-liquidity universe. Static list — no caching needed."""
    return sorted(FOCUSED_UNIVERSE)
