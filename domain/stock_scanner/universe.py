from __future__ import annotations

# Curated 290-ticker swing-trade universe across all sectors + AI themes.
# Update quarterly. Last updated 2026-06-22.
UNIVERSE_290: list[str] = [
    # ---------------- Indices & broad ETFs (5) ----------------
    'SPY', 'QQQ', 'IWM', 'DIA', 'SMH',

    # ---------------- Sector SPDR ETFs (11) ----------------
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC',

    # ---------------- Mega-cap tech (15) ----------------
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'AVGO',
    'ORCL', 'CRM', 'ADBE', 'INTC', 'IBM',

    # ---------------- AI semis & semicap (15) ----------------
    'MU', 'MRVL', 'ANET', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TSM', 'TXN', 'MCHP',
    'ON', 'NXPI', 'ADI', 'QCOM', 'ARM',

    # ---------------- AI infra & datacenter power (10) ----------------
    'VRT', 'CEG', 'VST', 'ETN', 'PWR', 'CARR', 'JCI', 'EMR', 'NEE', 'SO',

    # ---------------- AI software / SaaS (15) ----------------
    'PLTR', 'SNOW', 'NOW', 'CRWD', 'PANW', 'ZS', 'NET', 'DDOG', 'MDB', 'OKTA',
    'TEAM', 'WDAY', 'SHOP', 'SQ', 'HUBS',

    # ---------------- Crypto / Bitcoin proxies (8) ----------------
    'COIN', 'MSTR', 'MARA', 'RIOT', 'CLSK', 'HUT', 'CIFR', 'HOOD',

    # ---------------- Quantum / emerging tech (5) ----------------
    'IONQ', 'RGTI', 'QBTS', 'BBAI', 'SOUN',

    # ---------------- Financials — banks + payments (20) ----------------
    'JPM', 'BAC', 'WFC', 'C', 'MS', 'GS', 'V', 'MA', 'AXP', 'COF',
    'SCHW', 'BLK', 'BX', 'KKR', 'APO', 'ICE', 'CME', 'SPGI', 'MCO', 'PYPL',

    # ---------------- Insurance (8) ----------------
    'BRK-B', 'PGR', 'TRV', 'ALL', 'MET', 'PRU', 'CB', 'AIG',

    # ---------------- Healthcare — pharma + insurers + devices (20) ----------------
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'ISRG', 'VRTX',
    'REGN', 'GILD', 'AMGN', 'BMY', 'CVS', 'HUM', 'CI', 'ELV', 'DHR', 'BSX',

    # ---------------- GLP-1 / weight loss (4) ----------------
    'NVO', 'VKTX', 'MNKD', 'TLRY',

    # ---------------- Consumer staples (10) ----------------
    'WMT', 'COST', 'PG', 'KO', 'PEP', 'MDLZ', 'PM', 'MO', 'CL', 'KMB',

    # ---------------- Consumer discretionary (12) ----------------
    'AMZN', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'BKNG', 'MAR', 'HLT',
    'DIS', 'NFLX',

    # ---------------- Retail (5) ----------------
    'DG', 'DLTR', 'ROST', 'TJX', 'BBY',

    # ---------------- Energy (14) ----------------
    'XOM', 'CVX', 'COP', 'EOG', 'PSX', 'MPC', 'OXY', 'SLB', 'HAL', 'VLO',
    'KMI', 'WMB', 'LNG', 'PXD',

    # ---------------- Industrials (12) ----------------
    'CAT', 'DE', 'BA', 'HON', 'UPS', 'FDX', 'UNP', 'CSX', 'GE', 'MMM',
    'ITW', 'PH',

    # ---------------- Defense & aerospace (6) ----------------
    'LMT', 'NOC', 'RTX', 'GD', 'HII', 'TXT',

    # ---------------- Materials (10) ----------------
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'AA', 'NUE', 'STLD', 'CF',

    # ---------------- Real estate (8) ----------------
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'O', 'DLR', 'SPG',

    # ---------------- Utilities (additional) (5) ----------------
    'DUK', 'D', 'AEP', 'EXC', 'XEL',

    # ---------------- Communications / Media (8) ----------------
    'NFLX', 'DIS', 'T', 'VZ', 'TMUS', 'CMCSA', 'WBD', 'PARA',

    # ---------------- Auto / EVs (6) ----------------
    'F', 'GM', 'LCID', 'RIVN', 'STLA', 'NIO',

    # ---------------- China ADRs (6) ----------------
    'BABA', 'JD', 'PDD', 'BIDU', 'LI', 'XPEV',

    # ---------------- High-momentum mid/small caps (10) ----------------
    'SMCI', 'APP', 'HIMS', 'AFRM', 'TOST', 'DASH', 'ABNB', 'RBLX', 'CVNA', 'ROKU',

    # ---------------- Macro / hedges (10) ----------------
    'GLD', 'SLV', 'USO', 'UUP', 'TLT', 'TIP', 'VXX', 'UVXY', 'SQQQ', 'TZA',

    # ---------------- Cyber (additional) (4) ----------------
    'FTNT', 'S', 'ZS', 'CRWD',

    # ---------------- Misc liquid options names (8) ----------------
    'WBA', 'KO', 'PEP', 'MO', 'F', 'BAC', 'C', 'AAL',
]

# Deduplicate while preserving order (some tickers are intentionally listed in multiple themes)
def _dedupe(lst: list[str]) -> list[str]:
    seen = set()
    result = []
    for t in lst:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


UNIVERSE = _dedupe(UNIVERSE_290)
