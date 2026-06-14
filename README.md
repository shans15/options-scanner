# Options Scanner

End-of-day options scanner that finds **directional swing/day-trade setups first**, then ranks the best long-premium contracts to trade them with.

The pipeline narrows a 500-ticker universe down to a small watchlist by looking for four well-defined technical setups (compression breakouts, pullbacks in trend, stage-2 breakouts, and failed-breakdown reversals). For every surviving ticker, it evaluates real option chains and scores each contract by probability of profit, expected value, and stress-tested risk.

Read-only. No order execution.

## Purpose

Most option scanners are noisy because they evaluate every contract on every ticker, then try to sort the firehose. This one inverts the problem:

1. **Direction first.** A ticker only enters the scan if a Saty-style technical setup (compression / pullback / stage-2 / failed-breakdown) fires on its daily chart.
2. **Strategy is chosen by the setup.** Bullish setup → only long calls (and cash-secured puts) are evaluated. Bearish setup → only long puts (and naked calls). You don't waste analysis on contracts the setup doesn't support.
3. **The math you'd want either way runs on the survivors.** Four probability-of-profit models, stress on adverse moves, an EV-aware composite score.

The output is a ranked, labelled candidate list: TRADE / WATCHLIST / NO_TRADE — backed by which technical setup put the ticker on the list.

## How it works

1. **Universe builder.** Pulls the S&P 500 from Wikipedia, filters by 30-day average volume + options availability, caches daily.
2. **Technical filter (the new front gate).** For each ticker, fetches daily OHLCV and runs four detectors. Tickers without any setup are dropped before the expensive option-chain pull.
   - Compression breakout — Phase Oscillator squeeze with stacked ribbon
   - Pullback in trend — price taps the 13/21 EMA in a stacked trend with PO near zero
   - Stage 2 breakout / Stage 4 breakdown — fresh EMA48 cross after compression, near a 52w extreme
   - Failed breakdown / breakout reversal — new 20d low (or high) reclaimed on high volume with PO divergence
3. **Strategy gate.** The setup direction picks the eligible strategies:
   - Bullish setup → LongCall, NakedPut
   - Bearish setup → LongPut, NakedCall
4. **PoP blend.** Four models per contract — delta approx, Black-Scholes risk-neutral, historical rolling-window, GARCH(1,1) + Student-t Monte Carlo — blended 20/25/20/35.
5. **Stress.** 1σ / 2σ adverse moves plus 5th-percentile expiry move.
6. **Risk filters.** Strategy-specific spread, volume, OI, delta, PoP, EV, and stress thresholds.
7. **Composite score (0–100).** 50% PoP + 30% normalized EV + 20% regime alignment.
8. **Label.** TRADE (filters pass + score ≥ 65) / WATCHLIST (filters pass + 50–64) / NO_TRADE (otherwise).

Each candidate in the output is tagged with the setup that qualified it (`setup_name`, `setup_direction`, `setup_strength`), so you can see *why* the ticker is on the list — not just the option contract.

## A+ Confluence Scorer (optional second layer)

The scanner above ranks contracts. The A+ layer answers a different question: **of the candidates that survived, which ones cluster enough confluence to actually trade with a $1,000 account?**

It reads `output/scans/latest.json` and re-grades each candidate across 5 weighted categories using 20 features, then picks a structure (long premium vs debit spread) based on the IV / liquidity / regime mix.

| Category | Weight | What it looks at |
|---|---|---|
| Technical | 25% | Setup type, setup strength, weekly ribbon agreement, ATR-pivot proximity, volume z-score |
| Vol/VIX | 25% | VIX regime, 30-day VIX z-score, VVIX level, IV percentile |
| Catalyst | 15% | Days to earnings, days to next FOMC/CPI/PPI/NFP, skip-window blocker |
| Macro/Breadth | 15% | SPX trend stack, sector rotation rank, DXY trend, 10y yield direction |
| Liquidity | 20% | Bid-ask spread %, open interest, OI change DoD (v1 placeholder), volume/OI ratio |

**Grades:**
- **A+** — composite ≥ 90 AND every category ≥ 8 → size 12.5% of account
- **A** — composite ≥ 80 AND every category ≥ 7 → size 7.5% of account
- B+ / B / F — filtered out (not in the watchlist)

**Structure selector:** high IV percentile → debit spread (avoid vega bleed). Post-earnings drift + tight spreads → long premium (asymmetric upside). Sector leadership + supportive vol regime → long premium (trend continuation). Else → debit spread (defensive default with defined R:R).

The macro-event calendar lives at `data/sources/macro_calendar.py` and is hardcoded — update it quarterly from fomc.gov / bls.gov.

### Validation

We can't truly backtest the A+ grader at the contract level because historical option chains (for expired contracts) aren't available from free data sources. Instead:

- **Historical backtest of grades only** (`scripts/aplus_backtest_90d.py`) — re-grades 90 days of detected setups using cached OHLCV/sector/VIX data, with a fixture for liquidity. With `--liq good`, A-grade setups in the sample (n=43 over 90 days) showed **55.8% 1-day directional win rate** vs 48.3% baseline (+7.5pp lift, CI [41.1, 69.6] — underpowered but directionally positive). A+ remained unreachable historically because DXY/yield/VVIX scores were neutralized. The strongest single-category lift came from Technical (+4.2%) and Catalyst (+3.0%); Vol/VIX and Macro/Breadth added near-zero signal in this sample.
- **Forward test via paper trading** (`scripts/aplus_paper_trade.py`) — daily ingest A+/A signals into a JSON store, mark-to-market against live option chains, close at +100% target / -50% stop / expiration. Use this to gather real outcomes over 4–6 weeks (~20 trades) before risking real money.

## Delta ranges (research-backed)

| Strategy | Delta range | Source |
|---|---|---|
| NakedPut / NakedCall | 0.16 – 0.30 | Tastytrade 16-delta sweet spot + wheel-strategy 0.20–0.30 consensus |
| LongPut / LongCall | 0.40 – 0.60 | Gamma sweet spot for 3–45 DTE directional buys |

## Architecture

```
options-scanner/
├── data/           # I/O — yahooquery + yfinance + stooq fallback chain
│   └── sources/
│       ├── binance_source.py      # Binance API (US-geo-blocked; kept for reference)
│       ├── coinbase_source.py     # Coinbase Exchange: spot OHLCV (no auth, 12+ months)
│       └── hyperliquid_source.py  # Hyperliquid DEX: hourly perp funding rates
├── domain/         # Pure logic — Contract, Strategy, Regime, Greeks, TechnicalSetup + detectors
│   ├── aplus/                  # A+ Confluence Scorer (features, scoring, grading, structure)
│   └── crypto/
│       └── features.py         # Funding-rate feature engineering (no lookahead)
├── engine/         # PoP models, stress, risk filters, scorer
│   └── ml/
│       └── walkforward.py      # Expanding-window walk-forward CV splits
├── pipeline/       # Universe builder, earnings blackout, technical_filter, run_scan
├── scripts/
│   ├── aplus_watchlist.py         # A+/A confluence-graded daily watchlist
│   ├── aplus_backtest_90d.py      # Historical A+ grade win-rate validation
│   ├── aplus_paper_trade.py       # Daily forward-test (ingest/mark/close/summary)
│   └── btc_funding_validation.py  # Phase 1 edge-validation script
├── ui/             # Streamlit dashboard, CSV/JSON exporter
├── tests/
└── main.py         # CLI: scan / dashboard / watchlist / aplus_watchlist / universe
```

The technical filter (`domain/technical_signals.py` + `pipeline/technical_filter.py`) is the new front-of-pipeline stage. Everything downstream — option chain pull, PoP, stress, scoring — only runs on tickers that produced a setup.

## Crypto prediction module (Phase 1 — validation only)

Validates whether perpetual-futures funding-rate features predict short-horizon
BTC direction.

**Data sources:**
- **OHLCV:** Coinbase Exchange BTC-USD spot 15m (US-accessible, no auth, 12+ months history)
- **Funding:** Hyperliquid BTC perp hourly funding rates (US-accessible DEX, no auth)

Hyperliquid's public OHLCV retention is too short (~30 days) for ML training,
so we use Coinbase spot for price action. Funding is independent of where you
trade the spot — it's a sentiment signal.

If the edge exists (OOS accuracy >= 53%), Phase 2 builds the production pipeline.

```bash
python -m scripts.btc_funding_validation
```

This fetches 12 months of BTC-USD 15m OHLCV from Coinbase Exchange and hourly
funding rates from Hyperliquid (both cached to `cache/crypto/`), engineers
funding-derived features, runs 6-fold walk-forward cross-validation, and
reports out-of-sample accuracy on an untouched 20% holdout.

## Setup

```bash
git clone https://github.com/shans15/options-scanner.git
cd options-scanner
pip install -r requirements.txt
cp .env.example .env   # optional — tune thresholds
```

## Usage

```bash
# Run an EOD scan (Saty technical filter on by default; top 50 by liquidity)
python -m main scan

# Bypass the technical filter (legacy behavior — RV/IV regime gate only)
python -m main scan --no-technical-filter

# Faster scan for iteration (1000 MC paths vs 10000)
python -m main scan --fast

# Wider net
python -m main scan --top 100 --min-volume 500000

# Manual ticker override (skips screener)
python -m main scan --tickers SPY,QQQ,AAPL

# Run as if it were the morning of 2026-06-05 (use Thursday 6/4 EOD)
python -m main scan --as-of 2026-06-04 --tickers SPY,QQQ,AAPL

# Scan and launch dashboard immediately after
python -m main scan --then-dashboard

# Dashboard only (reads latest.json)
python -m main dashboard

# Print a focused long-only morning report from the latest scan
python -m main watchlist

# Read a specific scan file
python -m main watchlist --scan output/scans/scan_20260606_0531.json

# Grade today's scan against the A+ confluence rubric ($1k account by default)
python -m main aplus_watchlist

# Grade with custom account size + scan path
python -m main aplus_watchlist --scan output/scans/scan_20260606_0531.json --account-size 2500

# Historical backtest of A+ grades on the last 90 days of detected setups
python -m scripts.aplus_backtest_90d                   # neutral liquidity (~6 avg)
python -m scripts.aplus_backtest_90d --liq good        # good liquidity (~8.75 avg) — isolates other-category edge

# Paper trade tracker — runs ingest + mark + close + summary
python -m scripts.aplus_paper_trade                    # default: full daily cycle
python -m scripts.aplus_paper_trade ingest             # open new A+/A positions from latest.json
python -m scripts.aplus_paper_trade mark               # mark open positions to live chains
python -m scripts.aplus_paper_trade close              # close at +100% target / -50% stop / expiration
python -m scripts.aplus_paper_trade summary            # P&L table only

# Force universe cache rebuild
python -m main universe rebuild
```

Dashboard opens at `http://localhost:8501`. It has a sidebar widget to filter the candidate table by setup name.

`--as-of YYYY-MM-DD` pins the scanner to a specific EOD snapshot. Spot price comes from the close of that date, and the technical detectors only see bars on or before it. Option chains remain live (yahooquery doesn't expose historical chains), so use this primarily for the morning-of "what does the scanner see now" question, not deep historical backtesting.

## Tests

```bash
pytest                    # 522 unit + integration tests
```

## Data sources

| Source | Purpose | Notes |
|---|---|---|
| `yahooquery` | Primary: options chains, OHLCV history, spot, earnings | Quotes ~15–20 min delayed |
| `yfinance` | Fallback: options chains, OHLCV history, spot | Different scrape path — resilient when yahooquery breaks |
| `stooq` | Fallback²: OHLCV history only | No options |
| Wikipedia | S&P 500 constituents | Universe build |
| `py_vollib` | Greeks computation | We provide IV; library returns delta/gamma/theta/vega |

## Disclaimer

Quantitative research tooling only. Not financial advice.
