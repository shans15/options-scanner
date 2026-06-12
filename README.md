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
│       └── binance_source.py   # Public Binance API: OHLCV + perpetual funding rates
├── domain/         # Pure logic — Contract, Strategy, Regime, Greeks, TechnicalSetup + detectors
│   └── crypto/
│       └── features.py         # Funding-rate feature engineering (no lookahead)
├── engine/         # PoP models, stress, risk filters, scorer
│   └── ml/
│       └── walkforward.py      # Expanding-window walk-forward CV splits
├── pipeline/       # Universe builder, earnings blackout, technical_filter, run_scan
├── scripts/
│   └── btc_funding_validation.py  # Phase 1 edge-validation script
├── ui/             # Streamlit dashboard, CSV/JSON exporter
├── tests/
└── main.py         # CLI: scan / dashboard / universe
```

The technical filter (`domain/technical_signals.py` + `pipeline/technical_filter.py`) is the new front-of-pipeline stage. Everything downstream — option chain pull, PoP, stress, scoring — only runs on tickers that produced a setup.

## Crypto prediction module (Phase 1 — validation only)

Phase 1 validates whether funding-rate features predict short-horizon BTC direction.
If the edge exists (OOS accuracy >= 53%), Phase 2 builds the production pipeline.

Run the validation:

```bash
python -m scripts.btc_funding_validation
```

This fetches 12 months of BTC 15m OHLCV + funding from Binance (cached to
`cache/crypto/`), engineers funding-derived features, runs 6-fold walk-forward
cross-validation, and reports out-of-sample accuracy on an untouched 20%
holdout.

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

# Force universe cache rebuild
python -m main universe rebuild
```

Dashboard opens at `http://localhost:8501`. It has a sidebar widget to filter the candidate table by setup name.

`--as-of YYYY-MM-DD` pins the scanner to a specific EOD snapshot. Spot price comes from the close of that date, and the technical detectors only see bars on or before it. Option chains remain live (yahooquery doesn't expose historical chains), so use this primarily for the morning-of "what does the scanner see now" question, not deep historical backtesting.

## Tests

```bash
pytest                    # ~128 unit + integration tests
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
