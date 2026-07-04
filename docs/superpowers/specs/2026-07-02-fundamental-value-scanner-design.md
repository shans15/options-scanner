# Fundamental Value Scanner — Design

**Date:** 2026-07-02
**Author:** Sarthak (with Claude)
**Status:** Approved — ready for implementation planning
**Related:** Add-on to existing scanner; independent of A+ confluence scorer (`2026-06-29-aplus-rebalance-design.md`)

## Goal

Build a stand-alone monthly scanner that identifies **fundamentally mispriced stocks** by comparing each stock's current market price to a fair-value estimate derived from historical fundamentals and sector peer context. The output is a **planning tool**, not a real-time trading system — it produces a monthly report used to build a prospect list for the coming month.

The scanner covers the **Russell 2000** in Phase 1 (~2,000 tickers), with expansion to all US listed equities in Phase 2 (out of scope for this spec).

Two ranking methodologies are computed in parallel and both surfaced in every output, enabling empirical A/B comparison of methodologies across future monthly runs.

## Non-goals

- Not real-time — fundamentals refresh monthly, prices refresh per run
- Not integrated with A+ confluence scorer, GEX levels, or any options tooling
- Not an options recommender — output is stock-focused (long/short prospects)
- Not automated trading — output is a planning report only
- No DCF (discounted cash flow) — methodology is comparable-multiples + fundamental-vs-price divergence, not assumption-heavy modeling
- No daily short volume — v1 uses bi-weekly FINRA short interest only. Daily short volume feed is out of scope.
- No US-equity-wide universe in v1 — Russell 2000 first
- No backtest of Role 1 vs Role 2 methodologies — that's an A/B utility for Phase 2 once we have 3-6 monthly snapshots archived

## Acceptance criteria

1. `python -m scripts.value_watchlist` completes end-to-end on Russell 2000 with ≥ 1500 tickers graded (≥ 75% coverage of the constituent list).
2. Output JSON at `output/value/latest.json` contains both `role1_ranked_longs`/`role1_ranked_shorts` and `role2_ranked_longs`/`role2_ranked_shorts`, each with ≥ 20 entries.
3. Monthly archive `output/value/value_report_YYYY-MM.json` is written alongside `latest.json`.
4. Short interest fields (`short_interest_pct`, `days_to_cover`, `short_interest_delta`) populated for ≥ 85% of graded tickers.
5. `python -m scripts.value_check TICKER` runs against the cached fundamentals file and returns a single-ticker deep-dive.
6. Existing test suite still passes (no regression on A+ or scanner code).
7. New value-scanner module hits ≥ 85% unit coverage plus at least one integration test per CLI.
8. On a cold run (no cache), full scan completes in ≤ 60 minutes. On a warm run (cache hit), ≤ 10 minutes.

## Architecture

### Module layout

```
domain/value/
├── __init__.py
├── types.py                     # ValuationInputs, ValuationSnapshot, MispricingReport
├── fundamentals.py              # raw financials → structured inputs; missing-data handling
├── sector_relative.py           # Methodology A — peer-multiple comparison
├── fundamental_divergence.py    # Methodology B — revenue/EPS trend vs price trend
├── volume_overlay.py            # 5d/20d volume ratio + tag
├── short_overlay.py             # short interest overlay (direction-aware)
├── scoring.py                   # Role 1 tags + Role 2 composite
└── report.py                    # ranked-list rendering + JSON schema

data/sources/
└── yahooquery_fundamentals.py   # thin wrapper: income statement, balance sheet, key stats

data/universe/
├── russell2000_constituents.json   # committed monthly-refreshed snapshot
└── (existing sp500_constituents.json unchanged)

scripts/
├── refresh_russell2000.py       # pull latest Russell 2000 list from public source
├── value_watchlist.py           # monthly full-universe scan CLI
└── value_check.py               # on-demand single-ticker lookup

cache/value/
└── fundamentals_YYYY-MM.parquet # one file per monthly run

output/value/
├── latest.json                  # overwritten each run
└── value_report_YYYY-MM.json    # archived per month

tests/domain/value/              # unit tests per module
tests/scripts/
├── test_value_watchlist.py
└── test_value_check.py
```

### Data flow

```
data/universe/russell2000_constituents.json
   ↓
yahooquery_fundamentals.fetch_batch(tickers)   # 30-45 min cold; ~5 min warm
   ↓
cache/value/fundamentals_YYYY-MM.parquet       # persist for month
   ↓
   ┌─────────────┬────────────────────┬──────────────┬─────────────┐
   ↓             ↓                    ↓              ↓             ↓
sector_relative  fundamental_divergence  volume_overlay  short_overlay
   ↓             ↓                    ↓              ↓
   └─────────────┴────────────────────┴──────────────┴─────────────┘
   ↓
scoring.py — Role 1 tags + Role 2 composite (both computed in parallel)
   ↓
report.py — assemble MispricingReport
   ↓
output/value/latest.json + output/value/value_report_YYYY-MM.json
```

No changes to existing modules. Entirely additive.

## Methodology A — Sector-relative multiples

Compare each ticker's four core valuation ratios to the median of its GICS sector peers.

### Ratios

| Metric | Formula | Skip if |
|---|---|---|
| P/E (forward) | `price / forward_eps` | `forward_eps ≤ 0` |
| P/S (TTM) | `price / (revenue_ttm / shares_outstanding)` | `revenue_ttm ≤ 0` |
| P/B | `price / book_value_per_share` | `book_value ≤ 0` |
| EV/EBITDA (TTM) | `enterprise_value / ebitda_ttm` | `ebitda_ttm ≤ 0` |

### Per-ratio scoring

For each ratio available on the ticker:
```
peer_median = median of ratio across all tickers in same GICS sector (require min n=10 peers)
discount_pct = (peer_median - ticker_ratio) / peer_median   # negative if premium
```

Map `discount_pct` to a 0-10 score:

| `discount_pct` | Score |
|---|---:|
| > +30% | 10 (very undervalued vs peers) |
| +15 to +30% | 7-9 (linear interp) |
| ±15% | 5-7 (fair) |
| -15 to -30% | 2-4 (linear interp) |
| < -30% | 0-1 (very overvalued vs peers) |

### Sector-relative score

Weighted average of the 4 per-ratio scores, weighted by data availability (drop missing ratios, don't penalize).

If fewer than 2 ratios are available, ticker is **skipped** (insufficient data).

### Sector taxonomy source

GICS 11-sector classification pulled from yahooquery's `asset_profile.sector` field:
Technology, Healthcare, Financials, Consumer Discretionary, Consumer Staples, Energy, Industrials, Materials, Utilities, Real Estate, Communication Services.

Any ticker whose sector is null or "Unknown" is skipped.

## Methodology B — Fundamental momentum divergence

Compare 12-month fundamental trends to 12-month price trends. When the two diverge, the market is mispricing the fundamentals.

### Inputs

Per ticker:
- `revenue_growth_yoy` = (revenue_ttm - revenue_ttm_1y_ago) / revenue_ttm_1y_ago
- `eps_growth_yoy` = (eps_ttm - eps_ttm_1y_ago) / eps_ttm_1y_ago
- `price_growth_yoy` = (price_now - price_1y_ago) / price_1y_ago

### Divergence

```
fundamental_growth = 0.6 * revenue_growth_yoy + 0.4 * eps_growth_yoy
divergence = fundamental_growth - price_growth_yoy
```

### Score mapping (0-10)

| `divergence` (percentage points) | Score | Meaning |
|---|---:|---|
| > +25pp | 8-10 | Fundamentals sprinting ahead of price → undervalued |
| +10 to +25 | 6-8 | Mild undervaluation |
| -10 to +10 | 4-6 | Aligned |
| -25 to -10 | 2-4 | Mild overvaluation |
| < -25pp | 0-2 | Price sprinted ahead of fundamentals → overvalued |

### Guardrails

- Skip if `revenue_ttm < $50M` (micro-caps have noisy financials)
- Skip if `eps_ttm` crossed zero YoY (growth rate is undefined near zero)
- If EPS negative both periods, use revenue-only weighting (100% revenue, 0% EPS)

## Volume overlay

Applies to both Role 1 and Role 2.

```
volume_5d_avg  = mean of last 5 trading days volume
volume_20d_avg = mean of last 20 trading days volume
volume_ratio   = volume_5d_avg / volume_20d_avg
```

Tag + score:

| `volume_ratio` | Tag | Score |
|---|---|---:|
| ≥ 1.15 | RISING | 10 |
| 0.85 – 1.15 | FLAT | 5 |
| ≤ 0.85 | DECLINING | 0 |

## Short interest overlay (direction-aware)

Pulled from yahooquery `key_stats`:
- `short_interest_pct` = `shares_short / float_shares`
- `days_to_cover` = `shares_short / avg_daily_volume_30d`
- `short_interest_delta` = current SI minus SI from prior monthly report (arrow ↑/↓/→)

### Tag mapping — Role 1

Combined with base rank (mispricing rank score 0-10):

| Condition | Tag |
|---|---|
| Undervalued (rank ≥ 7) + SI > 15% + DTC > 5 | SQUEEZE SETUP ⚡ |
| Undervalued (rank ≥ 7) + SI 5-15% | HIGH PRIORITY |
| Undervalued (rank ≥ 7) + SI < 5% | HIGH PRIORITY (clean) |
| Undervalued + `short_interest_delta_pp > +1.0` (SI up more than 1 pp from prior monthly report) | SHORTS BUILDING — THESIS AT ODDS |
| Overvalued (rank ≤ 3) + SI < 5% | CLEAN SHORT ✓ |
| Overvalued (rank ≤ 3) + SI 5-15% | PATIENT SHORT |
| Overvalued (rank ≤ 3) + SI > 15% | CROWDED SHORT — AVOID ⚠️ |

Middle ranks (3 < rank < 7) receive no priority tag — just displayed with raw metrics.

### Score mapping — Role 2

`short_signal_score` (0-10), direction-aware:

```
if base_rank >= 5:  # leaning long
    if SI > 15%: short_signal_score = 10   # squeeze validation
    elif SI 5-15%: short_signal_score = 6
    else: short_signal_score = 5           # neutral
else:  # leaning short
    if SI > 15%: short_signal_score = 0    # crowded short — dangerous
    elif SI 5-15%: short_signal_score = 4
    else: short_signal_score = 10          # fresh short opportunity
```

If SI data is missing, `short_signal_score = 5` (neutral, no signal).

## Role 1 — Mispricing rank + priority tag

`combined_rank_score = mean(available_scores)` in 0-10 scale, where `available_scores` is `[sector_relative_score, fundamental_divergence_score]` filtered to non-null values. If both are available it's the simple average; if only one is available it's that single score.

Rank longs by descending `combined_rank_score`; rank shorts by ascending `combined_rank_score`.

Priority tag assigned per the short-overlay table above.

Output top 20 longs and top 20 shorts.

## Role 2 — Weighted composite

Base weights:
```
sector_relative:        0.35
fundamental_divergence: 0.35
volume:                 0.15
short_signal:           0.15
```

If any component is null (see missing-data guardrails per methodology), its weight is redistributed proportionally across the remaining components before the weighted-average step. For example, if `fundamental_divergence_score` is null (revenue too small), the remaining three components get their weights scaled up: sector_relative 0.35 / 0.65 = 0.538, volume 0.15 / 0.65 = 0.231, short_signal 0.15 / 0.65 = 0.231.

```
composite_score = weighted_average(available_scores, renormalized_weights) * 10
```

Scale is 0-100 to match the A+ system convention.

Rank longs by descending composite; rank shorts by ascending composite. Output top 20 of each.

## Output JSON schema

Path: `output/value/latest.json` (overwritten each run) + `output/value/value_report_YYYY-MM.json` (archived).

```json
{
  "run_timestamp_utc": "2026-07-02T18:22:00Z",
  "universe": "russell2000",
  "universe_size": 2000,
  "graded_size": 1847,
  "skipped": 153,
  "skipped_reasons": {
    "missing_fundamentals": 98,
    "revenue_too_small": 42,
    "eps_sign_flip": 13
  },
  "role1_ranked_longs": [ /* top 20 sorted by combined_rank_score DESC */ ],
  "role1_ranked_shorts": [ /* top 20 sorted by combined_rank_score ASC */ ],
  "role2_ranked_longs": [ /* top 20 sorted by composite_score DESC */ ],
  "role2_ranked_shorts": [ /* top 20 sorted by composite_score ASC */ ]
}
```

Every entry in any of the four ranked lists contains:

```json
{
  "ticker": "INTC",
  "sector": "Technology",
  "spot": 24.11,
  "sector_relative_score": 8.5,
  "fundamental_divergence_score": 8.2,
  "combined_rank_score": 8.35,
  "composite_score": 74.5,
  "volume_tag": "RISING",
  "volume_score": 10.0,
  "short_tag": "SQUEEZE_SETUP",
  "short_signal_score": 10.0,
  "priority": "HIGH_PRIORITY_SQUEEZE",
  "short_interest_pct": 17.3,
  "days_to_cover": 6.2,
  "short_interest_delta_pp": 1.2,
  "sector_multiples": {
    "pe_fwd": 8.4, "pe_fwd_peer_median": 22.1, "pe_discount_pct": -62.0,
    "ps_ttm": 1.2, "ps_ttm_peer_median": 3.8, "ps_discount_pct": -68.4,
    "pb": 1.1, "pb_peer_median": 3.2, "pb_discount_pct": -65.6,
    "ev_ebitda": 6.5, "ev_ebitda_peer_median": 15.2
  },
  "fundamental_trend": {
    "revenue_growth_yoy_pct": 12.3,
    "eps_growth_yoy_pct": 8.7,
    "price_growth_yoy_pct": -18.4,
    "divergence_pp": 28.7
  },
  "volume": {
    "vol_5d_avg": 45000000,
    "vol_20d_avg": 38000000,
    "ratio": 1.18
  }
}
```

## CLI

### `python -m scripts.value_watchlist`

Full-universe monthly scan.

Arguments:
- `--universe {russell2000,sp500}` — default `russell2000`
- `--top N` — number of ranked entries per bucket, default 20
- `--no-cache` — force fresh fundamentals fetch even if this month's cache exists
- `--out PATH` — override default `output/value/`

Console output (summary):
```
=== Value Watchlist — Russell 2000 — 2026-07-02 ===
Universe: 2000  Graded: 1847  Skipped: 153

TOP 5 LONGS (Role 1 rank):
Rank Ticker Sector       Score Vol   Short  Priority
1    INTC   Technology   8.4   ↑     17.3%  SQUEEZE SETUP ⚡
2    XYZ    Materials    8.1   ↑      6.8%  HIGH PRIORITY
...

TOP 5 LONGS (Role 2 composite):
Rank Ticker Composite  SR   FD   Vol  Short  Δ vs Role1
1    INTC   74.5       8.5  8.2  10   10     (same)
2    DEF    72.0       9.2  6.5  10   5      (+3 slots)
...

TOP 5 SHORTS (Role 1):
...

Report saved to: output/value/latest.json
                 output/value/value_report_2026-07.json
```

### `python -m scripts.value_check TICKER`

On-demand single-ticker deep-dive against the current month's cached fundamentals.

Arguments:
- `TICKER` (positional, required)
- `--show-peers` — display top 5 sector peers with their ratios

Console output as shown in Section 3 of the brainstorming discussion — full metric breakdown for one ticker.

If `TICKER` is not in the current month's cache, print a warning and offer to run the full scan.

## Data source strategy

### yahooquery (primary, only)

The `yahooquery` library is already used elsewhere in the codebase. We add a new wrapper `data/sources/yahooquery_fundamentals.py` that pulls:

| Field | yahooquery attribute |
|---|---|
| `sector` | `Ticker.asset_profile[t]['sector']` |
| `forward_eps` | `Ticker.key_stats[t]['forwardEps']` |
| `revenue_ttm` | `Ticker.income_statement[t]['TotalRevenue']` (last row) |
| `revenue_ttm_1y_ago` | `Ticker.income_statement[t]['TotalRevenue']` (4 rows back) |
| `eps_ttm` | `Ticker.income_statement[t]['DilutedEPS']` (last row) |
| `eps_ttm_1y_ago` | `Ticker.income_statement[t]['DilutedEPS']` (4 rows back) |
| `book_value_per_share` | `Ticker.key_stats[t]['bookValue']` |
| `enterprise_value` | `Ticker.key_stats[t]['enterpriseValue']` |
| `ebitda_ttm` | `Ticker.income_statement[t]['EBITDA']` (last row) |
| `shares_short` | `Ticker.key_stats[t]['sharesShort']` |
| `float_shares` | `Ticker.key_stats[t]['floatShares']` |
| `avg_daily_volume_30d` | `Ticker.summary_detail[t]['averageVolume']` |
| `price_now` | `Ticker.summary_detail[t]['regularMarketPrice']` |
| `price_1y_ago` | `Ticker.history(period='1y').iloc[0]['close']` |
| `volume_5d`, `volume_20d` | derived from `Ticker.history(period='1mo')` |

### Rate limiting

- Batches of 20 tickers per yahooquery call (library supports this)
- 1 second sleep between batches
- Total for 2000 tickers: `2000/20 = 100 batches × ~5 sec = ~500 seconds` (~8 min) baseline; account for retries brings it to 30-45 min cold
- Retry per batch: 2 retries with exponential backoff on any failure

### Missing data policy

- If ticker returns entirely empty payload: skip, count as `missing_fundamentals`
- If ticker has only some fields: compute scores from what's available, don't penalize for missing data
- Track per-ticker skip reason in `skipped_reasons`

### Russell 2000 constituent list

Source: `https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund` (iShares IWM holdings CSV, publicly available).

`refresh_russell2000.py` pulls this CSV, extracts the ticker column, normalises symbols (e.g., `BRK.B` → `BRK-B`), writes `data/universe/russell2000_constituents.json` as a plain JSON list.

Fall back to a committed snapshot if the fetch fails, same pattern as `sp500_constituents.json`.

## A/B tracking (Phase 2 — out of scope for v1)

Every monthly run archives `output/value/value_report_YYYY-MM.json`. After 3-6 monthly cycles, a future utility `scripts.value_compare_runs` will:

1. Load N archived reports
2. For each ticker that appeared in Role 1 top-20, look up the actual 30d/60d/90d forward return
3. Same for Role 2 top-20
4. Compare distributions: median return, win rate (positive returns), max return, max drawdown
5. Report which methodology (Role 1 vs Role 2) demonstrably outperforms

Not built in v1. The archive schema supports it later.

## Error handling

- `refresh_russell2000` fails → fall back to committed snapshot, warn user
- yahooquery batch fails → retry 2× with exponential backoff, then skip batch
- Any single ticker throws → catch, mark as skipped, continue
- Empty scan output (0 graded) → exit cleanly with warning message, no crash
- Missing sector peers (n < 10 in that GICS sector) → skip sector-relative scoring for that ticker, use only fundamental-divergence

## Testing strategy

| Module | Test type | Test data |
|---|---|---|
| `fundamentals.py` | Unit | Synthetic JSON payloads including missing / NaN / zero / negative edge cases |
| `sector_relative.py` | Unit | 10-ticker synthetic sector, verify peer-median math and score mapping |
| `fundamental_divergence.py` | Unit | Synthetic growth series covering all 5 score bands + guardrail cases |
| `volume_overlay.py` | Unit | Synthetic volume series, boundary values (0.849 / 0.85 / 1.15 / 1.151) |
| `short_overlay.py` | Unit | Direction-aware branches: (rank<5, SI>15), (rank>=5, SI>15), missing data |
| `scoring.py` | Unit | Pre-computed inputs → verify both Role 1 tags and Role 2 composites |
| `report.py` | Unit | Serialize a fixture, assert schema fields present, sort orders correct |
| `value_watchlist.py` | Integration | 10-ticker mini-universe fixture Parquet → run full pipeline, assert JSON output |
| `value_check.py` | Integration | Query mini-universe fixture, verify single-ticker output structure |

Target: ≥ 85% unit coverage on `domain/value/*`. New tests must not break existing 700+ test suite.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| yahooquery gets rate-limited or blocks scanner | Batch size + sleeps + retry; committed constituent list falls back on failure; A+ scanner uses same library and has been stable |
| Fundamentals data quality varies wildly across Russell 2000 (small caps) | Explicit guardrails (min revenue, sector peer count) + `skipped_reasons` tracking makes coverage visible |
| Role 1 vs Role 2 empirical winner not decidable within 1-2 months | v2 A/B tracking utility built on archived snapshots — decision deferred until data supports it |
| Ticker classification errors (wrong GICS sector) | Use single canonical source (yahooquery `asset_profile.sector`); accept ~1% error rate as noise |
| Sector median gets pulled down by outliers | Median is by definition outlier-resistant; no additional trimming needed for v1 |
| Short interest data lag (~2 weeks per FINRA cycle) | Acceptable for monthly cadence; bi-weekly refresh matches use case |
| User expects real-time recommendations | Scope explicitly non-goal; CLI console output makes clear "planning tool" |

## Open questions

None at sign-off. Empirical questions (Role 1 vs Role 2 outperformance, ideal top-N cutoff, whether short interest weight should be tuned) are resolved by monthly archive comparison in Phase 2.
