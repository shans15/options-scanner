from __future__ import annotations

# Universe fallback used when the live S&P 500 source (Wikipedia) is unreachable.
# Curated for $1k-friendly option premiums + themes actually moving in the
# current tape. Update quarterly as themes rotate.
KNOWN_GOOD_FALLBACK: list[str] = [
    # Indices & broad ETFs (4)
    'SPY', 'QQQ', 'IWM', 'DIA',

    # Sector SPDR ETFs (12) — drives A+ sector_rotation feature + cheapest options
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC',
    'SMH',  # Semis ETF — captures sector rotation without single-name risk

    # AI semis basket (5) — the action this month
    'NVDA', 'AMD', 'MU', 'MRVL', 'AVGO',

    # AI infrastructure & semicap equipment (5)
    'ANET',  # AI datacenter networking
    'AMAT',  # Applied Materials — semicap
    'LRCX',  # Lam Research — etch/deposition
    'KLAC',  # KLA — process control
    'VRT',   # Vertiv — datacenter cooling + power

    # AI power / datacenter electrification (3) — booming theme
    'CEG',   # Constellation Energy — nuclear for AI datacenters
    'VST',   # Vistra — independent power producer
    'ETN',   # Eaton — electrical systems

    # AI software & data (2)
    'PLTR',  # Palantir — enterprise AI software
    'CRWD',  # CrowdStrike — AI-driven cybersecurity

    # Bitcoin / crypto proxies (2)
    'COIN',  # Coinbase
    'MSTR',  # MicroStrategy — bitcoin treasury

    # Mega-cap stocks (6)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',

    # Sector leaders w/ liquid options (4)
    'JPM',   # Money-center bank
    'V',     # Payment networks
    'COST',  # Defensive growth
    'UNH',   # Healthcare leader

    # Macro / hedge instruments (2)
    'GLD',   # Gold
    'TLT',   # 20-year Treasury bonds
]
