from __future__ import annotations

FOCUSED_UNIVERSE = [
    # Index ETFs (deepest options liquidity)
    'SPY', 'QQQ', 'IWM',
    # Magnificent 7
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Bonus: liquid macro ETFs with low gap risk
    'GLD', 'TLT',
]

MEME_EXCLUSION_LIST = ['GME', 'AMC']  # kept for safety checks


def get_universe(use_cache: bool = False) -> list[str]:
    """Return the focused high-liquidity universe. Static list — no caching needed."""
    return sorted(FOCUSED_UNIVERSE)
