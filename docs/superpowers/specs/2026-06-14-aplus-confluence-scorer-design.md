# A+ Confluence Scorer for Long-Premium and Debit-Spread Trades — Design

**Date:** 2026-06-14
**Author:** Sarthak (with Claude)
**Status:** Approved (architecture sign-off received)
**Updated:** 2026-06-29 — Sizing rules ($1k account) and grade thresholds in this document are **superseded** by `docs/superpowers/specs/2026-06-29-aplus-rebalance-design.md`. The architecture, features, structure selector, and forward-test protocol described below remain authoritative.

## Goal

Build a multi-factor confluence scorer that augments the existing options scanner with A+/A/B+/B grading on each candidate, recommends an appropriate trade structure (long premium or debit spread) per candidate, and filters output to a small daily watchlist (1-5 trades) sized for a $1,000 account.

The motivating constraints:
- 90-day v1 backtest invalidated the assumption that daily-bar setups predict direction (48.3% win rate).
- v2 P&L backtest with the targeted subset still showed coin-flip win rate (50.5%), with apparent profits driven by underlying drift on long-dated contracts rather than 5-day directional capture.
- Credit-spread strategies require 80%+ win rate just to break even at small widths (1:4 R:R per contract), which is too tight a margin without backtest validation.
- $1,000 account cannot cash-secure puts on most liquid names and cannot fit naked premium-selling structures.

Therefore: the system focuses on debit structures (long premium and debit spreads) with strict confluence filtering to compensate for the structural ceiling on win rate (~45-55%) by amplifying per-trade R:R.

## Non-goals

These are explicitly out of scope for this spec:
- **Credit spreads** of any kind (the math is unsuitable at $1k without backtest validation).
- **Iron condors and other multi-leg structures** beyond debit spreads.
- **Auto-execution.** All trades remain manual; the scanner only outputs recommendations.
- **Backtest validation.** Established earlier in this thread that free data sources cannot backtest the strategy cleanly. Forward paper-trading is the validation path.
- **Per-ticker custom models.** A universal scorer applies to all tickers in v1.
- **Paid data integrations** (News API, Glassnode, ORATS, etc.).
- **Real-time streaming.** Poll-based execution on scan-run cadence only.
- **Re-architecting the existing scanner.** Net-additive on top of the current `output/scans/latest.json` pipeline.

## Acceptance criteria

The fix is done when all of the following hold:
1. New CLI command `python -m main aplus_watchlist` runs against `output/scans/latest.json` and produces an A+ watchlist output.
2. Every candidate in the scan input gets a composite confluence score in [0, 100] and a grade in {A+, A, B+, B}.
3. Candidates graded B+ or B are excluded from the output.
4. Each A+/A candidate has an explicit recommended trade structure (`long_premium` or `debit_spread`) with a written rationale tied to the inputs.
5. Sizing per the rules in §5 below is reflected in the output.
6. Forward-test protocol document (§5) is committed alongside the code.
7. Existing 434-test baseline still passes (no regressions).
8. New module has ≥85% test coverage at the unit level.
9. Committed and pushed to the existing PR branch.

## Architecture

### Module layout

```
options-scanner/
├── domain/aplus/                       # NEW — pure logic for the scorer
│   ├── __init__.py
│   ├── features.py                     # extract per-feature values from a candidate
│   ├── scoring.py                      # category scores + composite
│   ├── grading.py                      # composite → A+/A/B+/B label
│   └── structure.py                    # long-premium vs debit-spread selector
├── data/sources/
│   └── macro_calendar.py               # NEW — hardcoded FOMC/CPI/PPI/jobs schedule
├── scripts/
│   └── aplus_watchlist.py              # NEW — CLI entry, reads scan JSON, renders
└── tests/
    ├── domain/aplus/                   # NEW
    │   ├── test_features.py
    │   ├── test_scoring.py
    │   ├── test_grading.py
    │   └── test_structure.py
    └── data/test_macro_calendar.py     # NEW
```

No edits to existing pipeline. The new code reads `output/scans/latest.json` (the existing artifact) and renders to `output/aplus/latest.json` and console.

### Data flow

```
existing scanner output (output/scans/latest.json — 1000+ candidates)
   ↓
domain/aplus/features.py — feature extraction per candidate
   ↓
domain/aplus/scoring.py — category scores → composite (0-100)
   ↓
domain/aplus/grading.py — composite → A+ / A / B+ / B
   ↓
filter to grade ≥ A
   ↓
domain/aplus/structure.py — long premium vs debit spread per candidate
   ↓
scripts/aplus_watchlist.py — render + write output/aplus/latest.json
```

## Features (categories and feature definitions)

5 categories, ~20 individual features. Each feature normalizes to a 0-10 score. Category scores are arithmetic means of their features.

### Category 1 — Technical (weight 25%)
- **`tech_setup_type`** — 10 if `compression_breakout`, 8 if `stage_2_breakout`, 5 if `failed_breakdown_reversal`, 3 if `pullback_in_trend`. Based on v1 backtest detector-level win rates.
- **`tech_setup_strength`** — `min(10, strength * 10)`. Direct from existing `setup_strength` field.
- **`tech_weekly_ribbon_agreement`** — 10 if True, 5 if False (per v1 backtest, agreement only weakly predictive but not anti-predictive).
- **`tech_atr_pivot_position`** — 10 if price within 0.5 ATR of prior daily close, scaling down to 0 at 2 ATR (extended bars are less reliable entry points).
- **`tech_volume_zscore`** — `clip((volume_z - 0) * 5, 0, 10)`. Higher relative volume = stronger confirmation.

### Category 2 — Vol / VIX Context (weight 25%)
- **`vol_vix_regime`** — 10 if `contraction`, 7 if `neutral`, **0 if `expansion`** (hard kill per v1: 31.7% win rate during expansion).
- **`vol_vix_zscore_30d`** — favor below-average VIX for long premium. `clip(10 - (vix_zscore_30d * 2), 0, 10)`.
- **`vol_vvix_level`** — 10 if VVIX < 90, scaling down to 3 if VVIX > 120 (VVIX > 110 implies unstable vol regime).
- **`vol_iv_percentile`** — for long premium scoring: 10 at IV percentile 30, decreasing as IV percentile rises. Influences structure selector (high IV → prefer debit spread).

### Category 3 — News / Catalyst (weight 15%)
- **`cat_days_to_earnings`** — 0 if earnings ≤ 5 days away (SKIP), 10 if 5-15 days away (clean window), 6 if > 30 days (no catalyst soon).
- **`cat_days_to_macro_event`** — 0 if FOMC/CPI/PPI within 2 days, 10 if event > 5 days away. Macro events introduce binary risk.
- **`cat_post_earnings_drift`** — 10 if ticker had earnings in last 3-7 days AND gap held (post-earnings drift is a documented effect), 5 otherwise.
- **`cat_sector_earnings_density`** — 10 if no sector peers reporting this week, 5 if 1-2, 0 if 3+ (sector-wide vol from peer reports).

### Category 4 — Macro / Breadth (weight 15%)
- **`macro_spx_trend`** — 10 if SPY > 50EMA > 200EMA AND setup is bullish (or inverse for bearish), 4 if mixed, 0 if SPY trend opposes setup direction.
- **`macro_sector_rotation`** — 10 if ticker's sector ETF is in top 3 of 11 sectors by 1-week return AND bullish setup (or bottom 3 for bearish), 5 if mid-pack, 0 if leadership opposes.
- **`macro_dxy_trend`** — 10 if DXY direction supports setup (down DXY → bullish for most stocks; up DXY → bearish for multinationals), 5 if neutral, 2 if opposes.
- **`macro_10y_yield_direction`** — 10 if yield direction supports the sector context (e.g., yields up → bullish for XLF; yields down → bullish for XLK), 5 if neutral, 0 if opposes.

### Category 5 — Liquidity / Flow (weight 20%)
- **`liq_bid_ask_spread`** — 10 if spread/mid < 3%, scaling down to 0 at > 15%.
- **`liq_open_interest`** — 10 if OI ≥ 1000 at target strike, scaling down to 0 below 100.
- **`liq_oi_change_dod`** — 10 if open interest increased > 25% day-over-day (institutional accumulation signal), 5 if flat, 3 if declining. Requires caching yesterday's chain.
- **`liq_volume_oi_ratio`** — 10 if today's volume/OI > 0.3 (active flow), 5 if normal, 3 if dormant.

### Category weights

```
Technical:        25%   (raw signal)
Vol / VIX:        25%   (v1 validated as kill switch)
Catalyst:         15%   (event risk filter)
Macro / Breadth:  15%   (regime alignment)
Liquidity:        20%   (execution viability — high weight because of $1k account)
                ────
Total:           100%
```

VIX category is weighted equally with Technical because v1 backtest showed `VIX_REGIME=EXPANSION` flipped win rate from 48.3% to 31.7%. Liquidity is at 20% because at $1,000 a single mispriced fill (10% slippage on a $50 trade = $5) compounds the per-trade math significantly.

## Grading thresholds

```
A+  composite ≥ 90  AND every category ≥ 8/10  →  top conviction, full sizing
A   composite ≥ 80  AND every category ≥ 7/10  →  solid, reduced sizing
B+  composite ≥ 70  AND every category ≥ 6/10  →  display, do NOT trade
B   composite ≥ 60                              →  ignore
F   composite < 60                              →  filtered out before output
```

Critical: the "every category ≥ X" floor prevents one strong category from offsetting a fatal weakness in another. A trade can't be A+ on technicals alone if the VIX category is dragging.

## Structure selection logic

Per A+/A candidate, the structure selector picks one of `{long_premium, debit_spread}`:

```python
def select_structure(candidate, features):
    # Hard rule: never long premium when IV is expensive
    if features["vol_iv_percentile"] > 70:
        return DEBIT_SPREAD, "IV percentile > 70 — vega exposure would hurt long premium"

    # Lottery: post-earnings drift + excellent liquidity = long premium
    if features["cat_post_earnings_drift"] == 10 and features["liq_bid_ask_spread"] >= 9:
        return LONG_PREMIUM, "post-earnings drift + tight spreads = asymmetric upside"

    # Trend continuation: strong 4w + neutral vol = long premium
    if features["macro_sector_rotation"] >= 9 and features["vol_vix_regime"] >= 7:
        return LONG_PREMIUM, "sector leadership + vol-supportive = trend continuation"

    # Default: debit spread (better PoP, capped cost)
    return DEBIT_SPREAD, "defensive default — defined R:R, better PoP"
```

Rationale per branch:
- IV > 70 percentile: long premium has unfavorable vega; debit spread half-hedges the IV exposure.
- Post-earnings drift + tight spreads: documented effect (Bernard & Thomas 1989) where stocks drift in earnings direction for ~60 days. Combined with liquidity to capture cleanly.
- Sector leadership + supportive vol: trend continuation with vega tailwind. Long calls/puts amplify the asymmetric move.
- Otherwise: debit spread gives 1:1 to 1:2 R:R with ~45-55% PoP and lower outright cost.

## Sizing rules for $1,000 account

```
A+ grade trade:    10-15% account risk per trade  ($100-150)
A  grade trade:     5-10% account risk per trade  ($50-100)
B+ grade trade:    not traded (watchlist display only)
B  grade trade:    not output at all

Max concurrent positions:           3
Max daily new entries:              1 (avoids overtrading at small size)
Max consecutive losses rule:        2 losses in a row → 48h cooldown
Max daily drawdown rule:            -3% account → stop trading day
```

Sizing math sanity check at A+ grade:
```
Account: $1,000
A+ trade max risk: $150 (15%)

If win rate = 55% at A+ grade with 1:2 R:R debit spreads:
    Per 10 trades: 5.5 wins × $300 = $1,650 gained
                   4.5 losses × $150 = $675 lost
    Net: +$975 per 10 trades (highly variable)

If win rate = 50% at A grade with 1:2 R:R:
    Per 10 trades: 5 wins × $300 = $1,500
                   5 losses × $150 = $750
    Net: +$750 per 10 trades

Realistic monthly: 6-12 A+/A signals → +$200-500/month if execution holds
```

These are aspirational; actual results depend entirely on the confluence scorer's predictive power.

## Forward-test protocol (since real backtest isn't viable)

```
Phase 1 (Week 1-2): Paper-trade every A+/A signal
  - Log to output/aplus/paper_trades.csv
  - Track entry, exit, P&L, win/loss, grade
  - Annotate with reasoning per signal
  - Target: ≥ 20 paper trades

Phase 2 (Week 3-4): Continue paper-trading, add journal comparison
  - At end of each day, journal which signals you would have taken
  - Compare against what system flagged
  - Discrepancies → tune scorer weights
  - Target: ≥ 40 cumulative paper trades

Phase 3 (Week 5+): Begin live trading at half sizing
  - If paper P&L > 0 after 30+ trades: live at 5-7% account risk per trade
  - Track live performance separately
  - Target: 20+ live trades at half size

Phase 4 (post-validation): Scale to full sizing
  - If live P&L > 0 after 20 trades at half size: scale to 10-15% sizing
  - Continue monitoring for regime breaks
```

This is the standard forward-test discipline for any strategy that can't be cleanly backtested. The $1k account is small enough that drawdown during paper phase doesn't hurt; the discipline of not trading until paper-validated is the entire risk control.

## Trade structure details

### Long premium (single-leg)
```
Bullish: BUY call at 0.40-0.55 delta, 14-21 DTE
Bearish: BUY put at 0.40-0.55 delta, 14-21 DTE

Exit rules:
  Take profit at +50% of premium paid
  Stop loss at -50% of premium paid
  Time stop at 5 DTE remaining (avoid gamma blow-up at expiration)
```

### Debit spread (defined risk vertical)
```
Bullish: BUY call at 0.50 delta, SELL call $5 higher
Bearish: BUY put at 0.50 delta, SELL put $5 lower
DTE: 21-35 days

Exit rules:
  Take profit at +50% of max profit
  Stop loss at full debit paid (100% loss of risk)
  Time stop at 5 DTE remaining
```

Width is fixed at $5 for v1 simplicity. v2 may adapt width to underlying price.

## Error handling

- Missing earnings date in candidate row → treat as if > 30 days out (neutral score).
- Missing VIX data → set `vol_vix_regime = 0` (skip the trade).
- Missing OI on long leg of debit spread → fall back to long premium.
- All-NaN feature row → grade as F, exclude.
- Empty scan input → exit cleanly with "no candidates to score" message.

## Testing strategy

- Unit tests per feature (≥85% coverage on `features.py`)
- Unit tests on scoring math (edge cases: all zeros, all tens, partial NaN)
- Unit tests on grading thresholds (boundary conditions)
- Unit tests on structure selector (each branch fires correctly)
- Integration test: run `aplus_watchlist` against a known scan JSON fixture, assert specific candidate grades and structures.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scorer overfits to current market regime (June 2026) | Forward-test protocol catches regime drift in 30 trades; weights are documented and easy to retune. |
| 1-2 high-weight features dominate scoring | Category floors (every category ≥ X) prevent single-category compensation. |
| Liquidity feature requires day-over-day OI cache | First-run handles missing prior chain gracefully (score 5 = neutral); subsequent runs work normally. |
| Macro calendar drifts (FOMC dates change) | Hardcoded list versioned in source; update quarterly. |
| Real win rate stays at ~50% even at A+ grade | Forward-test catches it within 20-30 trades; at small sizing, max drawdown during validation is ≤ 30% of account. |
| Account size makes "every category ≥ 8" too rare to produce A+ trades | Tune thresholds downward after Phase 1 paper-test reveals base rate. |

## Open questions

None at sign-off. Empirical questions (real win rate at each grade, frequency of A+ signals) are resolved by forward-testing the live system.
