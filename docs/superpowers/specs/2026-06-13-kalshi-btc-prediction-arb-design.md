# Kalshi BTC Prediction Arbitrage — Design

**Date:** 2026-06-13
**Author:** Sarthak (with Claude)
**Status:** Approved (architecture sign-off received)

## Goal

Find short-horizon (15-min to 24-hr) BTC price prediction markets on Kalshi that are mispriced relative to research-paper-grounded probability estimates, output an actionable watchlist of mispricings ranked by edge × expected value, and backtest three strategies (A: vol-model only, B: research-paper ensemble only, A+B: full combined ensemble) to determine which produces the highest win rate after fees.

## Strategy comparison (the central experiment)

The whole point of this module is to empirically settle which approach has edge. Three strategies, same backtest harness, same data, compared head-to-head:

| Strategy | Estimators used | Adjustments | Source |
|---|---|---|---|
| **A — Vol-model only** | Existing AUC-0.745 vol model + lognormal | All 4 adjustments applied | Phase 1 vol model |
| **B — Research-paper ensemble** | GARCH(1,1) + Historical bootstrap + Realized-vol scaling | All 4 adjustments applied | Hansen & Lunde (2005), Engle (1982), Cont (2001) |
| **A+B — Combined ensemble** | All 4 estimators above, weighted average | All 4 adjustments applied | Combined |

Backtest output: head-to-head comparison of win rate, mean edge per trade, Sharpe ratio, max drawdown, total return.

## Scope

**In scope:**
- New `data/sources/kalshi_source.py` for public Kalshi REST API (no auth needed for read endpoints)
- New `domain/prediction/` module with `market.py`, `fair_probability.py`
- New `engine/prediction/` module with 4 estimators, ensemble, adjustments
- 3 runnable scripts: watchlist (live), backtest (3-strategy comparison), paper trade
- Risk controls baked into watchlist defaults
- Tests at the same density as existing modules (~unit + integration)

**Out of scope:**
- Order placement (read-only — user trades manually on Robinhood)
- Authenticated Kalshi endpoints
- Polymarket integration (defer to v2)
- Other markets beyond BTC (defer to v2)
- ML model retraining (reuses existing vol artifact)
- Real-time WebSocket subscription (polling REST is sufficient for 15m+ markets)

## Architecture

### Module layout

```
options-scanner/
├── data/sources/
│   └── kalshi_source.py              # NEW — public REST client
├── domain/prediction/
│   ├── __init__.py                   # NEW (empty)
│   ├── market.py                     # NEW — KalshiMarket, PredictionMarketEdge dataclasses
│   └── fair_probability.py           # NEW — public API: estimate P(BTC condition)
├── engine/prediction/
│   ├── __init__.py                   # NEW (empty)
│   ├── vol_model_estimator.py        # NEW — wraps existing vol_model.py
│   ├── garch_estimator.py            # NEW — GARCH(1,1) via `arch` library
│   ├── historical_bootstrap.py       # NEW — resample empirical returns
│   ├── realized_vol_estimator.py     # NEW — last-hour RV scaling
│   ├── ensemble.py                   # NEW — weighted average + 3-strategy harness
│   └── adjustments.py                # NEW — favorite-longshot, round-number, mean-rev, TOD
├── scripts/
│   ├── kalshi_btc_watchlist.py       # NEW — live watchlist
│   ├── kalshi_btc_backtest.py        # NEW — 3-strategy comparison
│   └── kalshi_paper_trade.py         # NEW — paper trade tracker
└── tests/
    ├── data/test_kalshi_source.py
    ├── domain/prediction/
    │   ├── test_market.py
    │   └── test_fair_probability.py
    ├── engine/prediction/
    │   ├── test_estimators.py
    │   ├── test_ensemble.py
    │   └── test_adjustments.py
```

### Data flow

```
[Live watchlist]
  Kalshi API → list open BTC markets, get current bid/ask
  Coinbase  → current BTC spot
  Existing vol model artifact → current vol regime probability
  
  for each market:
    market_implied_prob = (bid + ask) / 2
    fair_prob_A = vol_model_estimator(market, spot, vol_proba) + adjustments
    fair_prob_B = ensemble(GARCH, bootstrap, RV_scaling)(market, spot) + adjustments
    fair_prob_AB = ensemble(all 4)(market, spot) + adjustments
    
    Use fair_prob_AB as canonical → edge = fair_prob_AB - market_implied_prob
    Also output fair_prob_A and fair_prob_B for transparency
    
  Filter: |edge| >= 5%, volume >= $500, estimator agreement >= 3/4
  Sort: by expected_value = edge × position_size
  Output: ranked JSON + console table

[Backtest]
  Pull all closed Kalshi BTC markets in date range
  For each, replay: spot price at market open, then compute all 3 strategies' fair probs
  Compare to actual resolution
  Aggregate per strategy: win rate, mean edge, Sharpe, drawdown
  
  Output: comparison table A vs B vs A+B

[Paper trade]
  On each watchlist run, record signals → cache/paper_trades.json
  On resolution, mark realized P&L
  Generate periodic report
```

## Research-paper grounding

Each estimator and adjustment maps to a specific published finding. Documented in code comments + this spec.

### Estimators

**1. Vol-model estimator (Strategy A)**
- Uses your trained `cache/models/btc_vol_model.pkl` (AUC 0.745, audit-validated)
- Maps vol regime probability → implied σ via lookup table:
  - High-vol regime (proba > 0.65) → σ = historical 75th percentile
  - Low-vol regime (proba < 0.35) → σ = historical 25th percentile
  - Neutral → σ = historical median
- Computes P(BTC > X at T) using lognormal CDF
- **Theoretical grounding:** Black-Scholes (1973) — lognormal price distribution assumption

**2. GARCH(1,1) (Strategy B)**
- Library: `arch` (already pinned in requirements)
- Fit on last 30 days of hourly log returns
- Forecast σ for next N hours, then aggregate
- Compute P(BTC > X at T) using Student-t CDF (df=4) per Christoffersen et al. (2016) finding of heavy tails in crypto
- **Theoretical grounding:** Engle (1982), Hansen & Lunde (2005) — GARCH dominates simple vol models in forecast accuracy

**3. Historical bootstrap (Strategy B)**
- Last 90 days of hourly returns
- For an N-hour-forward market, draw 5,000 bootstrap samples of N consecutive hourly returns
- Compute fraction where BTC ended > X
- **Theoretical grounding:** Efron (1979) — non-parametric bootstrap. Cont (2001) — fat tails better captured by empirical distribution than parametric.

**4. Realized-vol scaling (Strategy B)**
- Compute realized vol from last 60 minutes of 1-min returns (RV = sum of squared log returns)
- σ_forward = √(RV × (T/1h)) — naive scaling
- Use Student-t for tails
- **Theoretical grounding:** Andersen et al. (2003) — RV is an unbiased efficient estimator of integrated volatility

### Adjustments (applied to all strategies after raw estimation)

**a. Favorite-longshot bias** (Snowberg & Wolfers, 2010)
- Bias: long-shot contracts (< 10%) are systematically overpriced by 3-7%; favorites (> 90%) underpriced
- Adjustment: shift fair_prob toward 50% by `0.03 × |fair_prob - 0.5| × 2` for contracts with fair_prob < 0.15 or > 0.85
- Effect: makes us less aggressive on tail bets

**b. Round-number magnetism** (Donaldson & Kim, 1993)
- Bias: BTC pins to round numbers ($63,000, $64,000, $65,000)
- Adjustment: if market strike X is a round number (X % $1000 == 0), increase P(BTC near X) by 2-3%
- Effect: bumps probability for "tests strike X" markets at round levels

**c. Mean reversion** (Lo & MacKinlay, 1988)
- Bias: returns show negative autocorrelation at sub-hourly horizons
- Adjustment: if last-hour return > 1σ (using GARCH σ), dampen forward σ forecast by factor 0.7 and shift forecast mean opposite to recent return by 0.3σ
- Effect: fades extreme recent moves

**d. Time-of-day vol** (Andersen et al., 2001)
- Bias: vol is higher during session opens (Asia 00:00 UTC, EU 08:00 UTC, US 14:00 UTC)
- Adjustment: vol multiplier table:
  - 22:00-02:00 UTC: 1.15 (Asia open)
  - 06:00-09:00 UTC: 1.10 (EU open)
  - 13:30-15:00 UTC: 1.20 (US open)
  - 19:30-20:00 UTC: 1.15 (US close)
  - other hours: 1.0
- Effect: scales forward σ depending on market expiration time

## Risk controls (baked into watchlist defaults)

| Control | Default | Rationale |
|---|---|---|
| Max position size | 5% of account | Limits single-trade blow-up |
| Daily exposure cap | 15% total | Prevents over-concentration on a single day's signals |
| Minimum edge | 5% after fees | Must clear Kalshi's 7% fee + bid-ask spread |
| Minimum daily volume | $500 | Avoids illiquid markets where exit is hard |
| Stop-trading rule | 3 consecutive losses OR -5% on day | Forces breaks during drawdowns |
| Estimator agreement | >= 3 of 4 estimators agree on direction | Filters noisy single-estimator signals |
| Hold to resolution | No early exits | Avoid paying spread twice |

## Backtest comparison framework

The backtest is the central deliverable. Output table format:

```
=== Kalshi BTC Backtest: 2026-01-01 → 2026-05-31 ===

Markets analyzed: 8,432
Markets resolved: 8,432

| Strategy           | Trades | Win % | Mean edge | Total P&L | Sharpe | Max DD |
|--------------------|--------|-------|-----------|-----------|--------|--------|
| A: Vol-model only  |   543  | 56.2% |    +3.1%  |   +28.4%  |  1.8   | -7.2%  |
| B: Research ensemble|  601  | 58.8% |    +3.4%  |   +35.7%  |  2.3   | -6.1%  |
| A+B: Combined      |   612  | 60.1% |    +3.8%  |   +47.2%  |  2.6   | -5.8%  |

Verdict: A+B Combined wins.
  Edge over A: +18.8% return
  Edge over B: +11.5% return
  Lower drawdown than either standalone
```

## Output formats

### Live watchlist (JSON to `output/kalshi/latest.json`)
```json
{
  "generated_at": "2026-06-13T14:00:00Z",
  "btc_spot": 63621,
  "vol_regime": {"direction": "contraction", "probability": 0.04},
  "markets_analyzed": 24,
  "markets_with_edge": 5,
  "recommendations": [
    {
      "ticker": "KXBTC-26JUN1316-T65000",
      "title": "BTC > $65,000 at 4:00 PM",
      "expiration": "2026-06-13T16:00:00Z",
      "strike": 65000,
      "side": "above",
      "market_implied_prob": 0.08,
      "fair_prob_A": 0.13,
      "fair_prob_B": 0.14,
      "fair_prob_AB": 0.14,
      "canonical_fair_prob": 0.14,
      "edge": 0.06,
      "estimator_agreement": "3/4",
      "expected_value_per_dollar": 0.075,
      "suggested_position_dollars": 80,
      "suggested_position_pct_account": 0.04,
      "adjustments_applied": ["favorite-longshot:-2%", "TOD:+1%"],
      "volume_24h": 1240,
      "spread": 0.02
    }
  ]
}
```

### Backtest comparison (CSV + console summary)
```
output/kalshi/backtest_YYYYMMDD_HHMM.csv
  Columns: timestamp, market_ticker, strategy, fair_prob, market_prob, edge,
           position_size, resolution_yes_no, pnl_per_dollar, cumulative_pnl
```

### Paper trade log (JSON)
```
output/kalshi/paper_trades.json
  Append-only log of every recommendation taken, with resolution + P&L back-fill
```

## Testing

**Unit tests per estimator**: feed synthetic vol/return paths, assert estimator output is correct (e.g., GARCH on a 30-day flat series → low σ forecast).

**Adjustment tests**: verify each adjustment applies in the right direction with the right magnitude.

**Ensemble test**: verify weighted average + agreement counter.

**Integration test (backtest)**: feed a small set of mocked Kalshi markets with known resolutions, verify all 3 strategies produce expected metrics.

**No-lookahead test**: critical — estimators at time T must use only data ≤ T. Mirror the existing `test_no_temporal_leakage_in_features` pattern.

**Anti-cheat test**: deliberately corrupt one estimator (return constant 0.5), verify ensemble degrades gracefully (other estimators still work).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kalshi rate-limits us | Cache aggressively; 1-minute polling cadence is more than enough |
| Backtest survivorship | Note: Kalshi BTC markets only exist since 2024. ~18 months of data. Results may not generalize. |
| Strategy overfitting | Compare A vs B vs A+B on the SAME backtest window. No hyperparameter tuning on backtest data. |
| Vol model drift | Re-train weekly via existing `train_btc_vol_model.py` script |
| Live execution slippage | Paper trade 2 weeks before live execution. Stop if paper P&L is negative. |

## Implementation notes

- Agents work in parallel. No inter-agent dependencies for development.
- Agent 1 (data) and Agent 2 (estimators) have clean APIs that Agent 3 (scripts) can mock for tests.
- After all 3 agents complete, integration test runs the full backtest to populate the comparison table.
- `arch` library is already a project dependency (used by existing GARCH PoP model in `engine/pop_models.py`).

## Open questions

None at sign-off. Architecture is approved. Empirical question (which strategy wins) is resolved by the backtest itself.
