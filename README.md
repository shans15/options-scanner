# Options Scanner

EOD options scanner for naked puts/calls (sell premium when overpriced) **and** long puts/calls (buy premium when underpriced). Runs once at end of day on a dynamic S&P 500 universe filtered by liquidity. Outputs a ranked candidate table to CSV/JSON and a Streamlit dashboard.

Read-only — no order execution.

## How it works

1. **Universe builder** — pulls S&P 500 from Wikipedia, filters by 30-day avg volume + options availability, caches daily.
2. **Per-ticker regime gate** — compares 30-day realized vol vs front-month ATM implied vol.
   - `RV/IV < 0.8` → favor **selling** (premium overpriced)
   - `RV/IV > 1.2` → favor **buying** (premium underpriced)
   - in-between → both directions, scoring sorts
3. **4 strategies** evaluated per contract: NakedPut, NakedCall, LongPut, LongCall.
4. **4 PoP models** per candidate: delta approx, Black-Scholes risk-neutral, historical rolling-window, GARCH(1,1) + Student-t Monte Carlo. Blended (20/25/20/35).
5. **Stress** — 1σ/2σ adverse moves + 5th-percentile expiry move.
6. **Risk filters** — strategy-specific spread, volume, OI, delta, PoP, EV, stress thresholds.
7. **Composite score (0-100)** = 50% PoP + 30% normalized-EV + 20% regime alignment.
8. **Label**: TRADE (pass + ≥65) / WATCHLIST (pass + 50-64) / NO_TRADE (fail or <50).

## Delta ranges (research-backed)

| Strategy | Delta range | Source |
|---|---|---|
| NakedPut / NakedCall | 0.16 – 0.30 | Tastytrade 16-delta sweet spot + wheel-strategy 0.20-0.30 consensus |
| LongPut / LongCall | 0.40 – 0.60 | Gamma sweet spot for 3-45 DTE directional buys |

## Architecture

```
options-scanner/
├── data/           # I/O — yahooquery + yfinance + stooq fallback chain
├── domain/         # Pure logic — Contract, Strategy, Regime, Greeks
├── engine/         # PoP models, stress, risk filters, scorer
├── pipeline/       # Universe builder, earnings blackout, run_scan
├── ui/             # Streamlit dashboard, CSV/JSON exporter
├── tests/
└── main.py         # CLI: scan / dashboard / universe
```

## Setup

```bash
git clone https://github.com/shans15/options-scanner.git
cd options-scanner
pip install -r requirements.txt
cp .env.example .env   # optional — tune thresholds
```

## Usage

```bash
# Run an EOD scan (top 50 by liquidity from S&P 500)
python -m main scan

# Faster scan for iteration (1000 MC paths vs 10000)
python -m main scan --fast

# Wider net
python -m main scan --top 100 --min-volume 500000

# Manual ticker override (skips screener)
python -m main scan --tickers SPY,QQQ,AAPL

# Scan and launch dashboard immediately after
python -m main scan --then-dashboard

# Dashboard only (reads latest.json)
python -m main dashboard

# Force universe cache rebuild
python -m main universe rebuild
```

Dashboard opens at `http://localhost:8501`.

## Tests

```bash
pytest                    # unit tests (~95)
```

## Data sources

| Source | Purpose | Notes |
|---|---|---|
| `yahooquery` | Primary: options chains, history, spot, earnings | Quotes ~15-20 min delayed |
| `yfinance` | Fallback: options chains, history, spot | Different scrape path — resilient when yahooquery breaks |
| `stooq` | Fallback²: prices only | No options |
| Wikipedia | S&P 500 constituents | Universe build |
| `py_vollib` | Greeks computation | We provide IV; library returns delta/gamma/theta/vega |

## Disclaimer

Quantitative research tooling only. Not financial advice.
