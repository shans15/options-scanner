# Options Scanner — Design Spec
**Date:** 2026-05-04
**Status:** Approved
**Scope:** Daily live scanner for short naked puts and naked calls using Robinhood data

---

## 1. Purpose

Build a time-aware, live options scanner that:
- Dynamically identifies the best stock/ETF candidates each trading day
- Analyzes their option chains for high-probability short naked put and short naked call setups
- Applies rigorous quantitative probability models and hard risk filters
- Outputs a ranked trade table to a live Streamlit dashboard and saves results to CSV/JSON
- Does NOT execute trades — read-only data pipeline only

**Disclaimer:** This is quantitative research tooling only, not financial advice.

---

## 2. Data Sources

| Source | Purpose | Library |
|---|---|---|
| Robinhood | Live option chains: bid/ask, volume, OI, IV, Greeks | `robin_stocks` |
| yfinance | Historical daily prices, IV rank baseline, earnings dates | `yfinance` |
| yfinance | VIX, SPY/QQQ trend, sector ETF data | `yfinance` |
| pandas-market-calendars | Market hours, holidays, ET timezone | `pandas_market_calendars` |

Credentials stored in `.env` (gitignored). Read-only Robinhood access — no order submission.

---

## 3. Architecture

```
Market Clock
    |-- triggers scans at 09:45, 11:00, 13:00, 15:00 ET
    v
Stock Screener
    |-- Universe: S&P 500 + liquid ETFs (~550 names)
    |-- Filters: price > $10, avg_volume > 1M, options_volume > 500
    |-- Scores: IV rank, momentum, trend, sector, VIX regime
    |-- Output: top 30-50 candidates per scan cycle
    v
Option Chain Fetcher (robin_stocks)
    |-- Per candidate: nearest 2-4 expirations
    |-- Data: bid, ask, mid, volume, OI, IV, delta, gamma, theta, vega, strike, expiry
    |-- Earnings filter: skip if earnings within 5 calendar days
    v
Quantitative Engine
    |-- 5 PoP models per contract
    |-- Composite blended PoP
    |-- Hard filter pass/fail
    |-- Composite score
    v
Output Layer
    |-- Streamlit dashboard (live, auto-refresh per scan cycle)
    |-- CSV + JSON export per cycle (timestamped)
```

---

## 4. File Structure

```
options-scanner/
├── .env                          # Robinhood credentials (gitignored)
├── .gitignore
├── requirements.txt
├── main.py                       # Entry point: starts scheduler + Streamlit
│
├── scanner/
│   ├── __init__.py
│   ├── clock.py                  # Market hours, ET timezone, scan schedule
│   ├── screener.py               # Stock universe filter + scoring
│   ├── chain_fetcher.py          # robin_stocks option chain puller
│   ├── quant_engine.py           # BS, GARCH, Monte Carlo, stress tests
│   ├── risk_filters.py           # Hard filters (liquidity, earnings, EV, spread)
│   └── scorer.py                 # Composite score + TRADE/WATCHLIST/NO TRADE
│
├── data/
│   ├── __init__.py
│   ├── universe.py               # S&P 500 + ETF ticker list
│   └── earnings_calendar.py      # Earnings date fetcher via yfinance
│
├── output/
│   ├── __init__.py
│   ├── dashboard.py              # Streamlit UI
│   ├── exporter.py               # CSV + JSON writer
│   └── scans/                    # Auto-saved timestamped scan results
│
└── docs/
    └── superpowers/specs/
        └── 2026-05-04-options-scanner-design.md
```

---

## 5. Stock Screener

**Universe:** S&P 500 constituents + core liquid ETFs (SPY, QQQ, IWM, GLD, TLT, XLF, XLE, XLK, XBI, ARKK).

**Hard filters (applied first):**
- Underlying price > $10
- 30-day average volume > 1,000,000 shares
- Options volume (from chain) > 500 contracts
- Not in earnings blackout (within 5 calendar days of earnings)

**Scoring factors (0-100 scale):**
| Factor | Weight | Source |
|---|---|---|
| IV Rank (0-100) | 30% | (current_IV - 52wk_low_RV) / (52wk_high_RV - 52wk_low_RV) using 1-year realized volatility range from yfinance daily prices as IV proxy |
| Trend alignment (put: uptrend; call: downtrend) | 25% | 20/50-day SMA comparison |
| Momentum score | 20% | RSI(14), rate of change |
| Liquidity score | 15% | Options volume + OI |
| Sector/market regime | 10% | VIX level, SPY/QQQ trend |

Top 30-50 scorers proceed to chain analysis.

---

## 6. Option Chain Fetcher

For each screened candidate:
- Pull option chains for the nearest 2-4 expirations (DTE range: 3-45 days)
- Focus on strikes within delta range 0.05-0.35 (5-35 delta) for both puts and calls
- Collect per contract: bid, ask, mid, volume, OI, implied_volatility, delta, gamma, theta, vega, strike, expiration_date
- Apply earnings blackout filter before fetching (skip the ticker entirely if within 5 days)
- Rate-limit robin_stocks calls: 1 request per 0.5 seconds to avoid throttling

---

## 7. Quantitative Engine

### 7.1 Probability of Profit Models

**Model 1 — Delta Approximation**
```
PoP = 1 - |delta|
```
Fast, uses Robinhood-provided delta directly.

**Model 2 — Black-Scholes Risk-Neutral Probability**
```
d2 = [ln(S/K) + (r - sigma^2/2) * T] / (sigma * sqrt(T))
PoP_put  = N(d2)     # probability stock stays above strike
PoP_call = N(-d2)    # probability stock stays below strike
```
Uses: spot price S, strike K, IV sigma from chain, risk-free rate r (3-month T-bill from yfinance), T in years.

**Model 3 — Historical Move Distribution**
```
Pull 1 year of daily log returns via yfinance
Scale to DTE horizon: sigma_horizon = sigma_daily * sqrt(DTE)
Empirically compute % of rolling DTE-day windows where stock
stayed on profitable side of strike
```

**Model 4 — GARCH(1,1) Monte Carlo**
```
Fit GARCH(1,1) on 1-year daily returns (arch library)
Forecast conditional volatility for DTE horizon
Run 10,000 Monte Carlo simulations with Student-t innovations
PoP = fraction of paths expiring on profitable side of strike
```
Uses Student-t for fat-tail realism.

**Model 5 — Stress Tests**
```
Scenario A: 1 SD overnight move (unrealized P&L)
Scenario B: 2 SD gap open (unrealized P&L)
Scenario C: GARCH worst-case at expiry (5th percentile path)
```

### 7.2 Blended PoP
```
Blended_PoP = 0.20 * PoP_delta
            + 0.25 * PoP_BS
            + 0.20 * PoP_historical
            + 0.35 * PoP_GARCH_MC
```

### 7.3 Expected Value
```
EV = (premium * PoP_blended) - (max_stress_loss * (1 - PoP_blended))
```
Where max_stress_loss = Scenario B stress loss.

### 7.4 Composite Trade Score (0-100)
| Component | Weight |
|---|---|
| Blended PoP | 30% |
| Expected value normalized | 20% |
| Premium / margin ratio | 15% |
| Liquidity score | 15% |
| Bid/ask tightness | 10% |
| Trend alignment | 10% |

---

## 8. Hard Filters

All must pass. Any failure = NO TRADE (not WATCHLIST).

| Filter | Threshold |
|---|---|
| Bid/ask spread | Must be <= 20% of mid price |
| Contract volume | Must be >= 100 contracts |
| Open interest | Must be >= 500 contracts |
| Earnings proximity | No earnings within 5 calendar days |
| Expected value | Must be > 0 |
| Blended PoP | Must be >= 70% |
| 2-day stress loss | Must be <= 3x premium received |
| Delta (puts) | Must be between -0.05 and -0.35 |
| Delta (calls) | Must be between 0.05 and 0.35 |

**Additional naked call restrictions:**
- Skip if short interest > 10% of float
- Skip if stock has moved > 30% in past 30 days (momentum/squeeze risk)
- Skip if stock is a known meme/low-float name (maintained exclusion list)
- Recommend converting to call credit spread if tail risk is borderline

---

## 9. Output

### 9.1 Decision Labels
- **TRADE:** All hard filters pass, composite score >= 65
- **WATCHLIST:** All hard filters pass, composite score 50-64
- **NO TRADE:** Any hard filter fails OR composite score < 50

### 9.2 Streamlit Dashboard

Layout:
- Header bar: last scan time, next scan time, market status, VIX, SPY % change
- Filter tabs: All / Naked Put / Naked Call / TRADE only / WATCHLIST only
- Main table: ranked by composite score, color-coded by decision
- Click-to-expand row: full detail panel with all 5 PoP models, stress scenarios, reason for/against, stop trigger
- Auto-refreshes on each scan cycle completion

### 9.3 Ranked Output Table Columns
ticker, strategy, expiration, strike, bid, ask, mid, premium, delta, gamma, theta, vega, IV, IV_rank, PoP_delta, PoP_BS, PoP_historical, PoP_GARCH_MC, PoP_blended, expected_value, breakeven, margin_estimate, stress_1SD, stress_2SD, stress_expiry, liquidity_score, bid_ask_score, composite_score, reason_for, reason_against, stop_trigger, decision

### 9.4 File Export
- Path: `output/scans/scan_YYYYMMDD_HHMM.csv` and `.json`
- Written after every scan cycle completes
- Includes all columns above

---

## 10. Scan Schedule (ET)

| Time | Rationale |
|---|---|
| 09:45 | After open volatility settles |
| 11:00 | Mid-morning — IV often peaks here |
| 13:00 | Midday — good for 0DTE/weekly setups |
| 15:00 | Power hour — premium decay plays |

Outside market hours: dashboard shows last scan results with "Market Closed" banner. No scanning runs.

---

## 11. Dependencies

```
robin_stocks>=2.1.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
arch>=6.3.0              # GARCH models
pandas-market-calendars>=4.3.0
streamlit>=1.33.0
python-dotenv>=1.0.0
apscheduler>=3.10.0      # Scan scheduler
plotly>=5.20.0           # Dashboard charts
requests>=2.31.0
```

Python version: 3.10+

---

## 12. Configuration (.env)

```
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
RISK_FREE_RATE=0.053        # Update periodically
MIN_POP_THRESHOLD=0.70
MIN_COMPOSITE_SCORE=65
MAX_SPREAD_PCT=0.20
MIN_VOLUME=100
MIN_OI=500
EARNINGS_BLACKOUT_DAYS=5
MONTE_CARLO_PATHS=10000
```

---

## 13. Out of Scope

- Trade execution of any kind
- Portfolio management or position tracking
- Options spreads (focus is naked puts and naked calls only)
- Real-time tick-by-tick streaming (scan-cycle based only)
- Backtesting (future phase)
