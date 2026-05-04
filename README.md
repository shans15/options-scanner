# Options Scanner

A live, time-aware options scanner for short naked puts and naked calls. Scans a focused universe of high-liquidity ETFs and Mag 7 stocks during market hours, runs five probability models per contract, applies hard risk filters, and outputs a ranked trade table to a Streamlit dashboard. Read-only — no trade execution.

---

## Universe

SPY, QQQ, IWM, AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, GLD, TLT

---

## How It Works

**Scan schedule (ET, market days only):** 09:45 · 11:00 · 13:00 · 15:00

Each cycle:
1. **Screener** — scores each ticker on IV rank, trend alignment, RSI momentum, liquidity, and market regime (VIX/SPY)
2. **Chain fetcher** — pulls live option chains from Robinhood (bid/ask, Greeks, IV, volume, OI) for the nearest 2–4 expirations, delta range 0.05–0.35, DTE 3–45
3. **Quant engine** — runs 5 probability models per contract and blends them into a composite PoP
4. **Risk filters** — applies 7 hard pass/fail filters; any failure → NO TRADE
5. **Scorer** — computes a 0–100 composite score and assigns TRADE / WATCHLIST / NO TRADE
6. **Output** — updates the live dashboard and writes `output/scans/scan_YYYYMMDD_HHMM.csv` + `.json`

---

## Probability Models

| Model | Weight | Method |
|---|---|---|
| Delta approximation | 20% | `1 - \|delta\|` |
| Black-Scholes | 25% | `N(d2)` for puts, `N(-d2)` for calls |
| Historical | 20% | Empirical rolling-window log-return distribution |
| GARCH(1,1) Monte Carlo | 35% | 10,000 paths, Student-t innovations |

**Blended PoP** = weighted average of all four.

---

## Hard Filters

All must pass. Any failure → NO TRADE.

| Filter | Threshold |
|---|---|
| Bid/ask spread | <= 20% of mid |
| Contract volume | >= 100 |
| Open interest | >= 500 |
| Delta | 0.05 – 0.35 (puts and calls) |
| Blended PoP | >= 70% |
| Expected value | > 0 |
| 2SD stress loss | <= 3x premium received |

---

## Decision Labels

| Label | Criteria |
|---|---|
| **TRADE** | All filters pass + composite score >= 65 |
| **WATCHLIST** | All filters pass + score 50–64 |
| **NO TRADE** | Any filter fails OR score < 50 |

---

## Dashboard

- Header: last scan time, next scan, market status, VIX, SPY % change
- Filter tabs: All / Naked Put / Naked Call / TRADE only / WATCHLIST only
- Ranked table: color-coded by decision, sortable by composite score
- Expandable rows: all 5 PoP models, stress scenarios, margin estimate, reason for/against, stop trigger
- Auto-refreshes every 60 seconds

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone
git clone https://github.com/shans15/options-scanner.git
cd options-scanner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
```

Edit `.env`:

```
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
RISK_FREE_RATE=0.053
MIN_POP_THRESHOLD=0.70
MIN_COMPOSITE_SCORE=65
MAX_SPREAD_PCT=0.20
MIN_VOLUME=100
MIN_OI=500
EARNINGS_BLACKOUT_DAYS=5
MONTE_CARLO_PATHS=10000
```

```bash
# 4. Launch
streamlit run main.py
```

Dashboard opens at `http://localhost:8501`.

---

## File Structure

```
options-scanner/
├── main.py                       # Entry point
├── requirements.txt
├── scanner/
│   ├── clock.py                  # Market hours + scan schedule
│   ├── screener.py               # Ticker scoring
│   ├── chain_fetcher.py          # Robinhood option chain fetcher
│   ├── quant_engine.py           # PoP models + stress tests
│   ├── risk_filters.py           # Hard filters
│   ├── scorer.py                 # Composite score + decision
│   ├── run_scan.py               # Full scan orchestrator
│   └── scheduler.py              # APScheduler singleton
├── data/
│   ├── universe.py               # Ticker universe
│   └── earnings_calendar.py      # Earnings blackout via yfinance
└── output/
    ├── dashboard.py              # Streamlit UI
    ├── exporter.py               # CSV + JSON writer
    └── scans/                    # Auto-saved scan results
```

---

## Running Tests

```bash
pytest tests/ -v
```

59 tests across all modules.

---

## Data Sources

| Source | Purpose |
|---|---|
| Robinhood (`robin_stocks`) | Live option chains — read-only, no order submission |
| yfinance | Historical prices, realized volatility, earnings dates |
| exchange_calendars | NYSE market calendar, ET timezone |

---

**Disclaimer:** This is quantitative research tooling only, not financial advice.
