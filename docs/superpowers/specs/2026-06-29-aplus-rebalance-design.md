# A+ Confluence Scorer — Recalibration Design

**Date:** 2026-06-29
**Author:** Sarthak (with Claude)
**Status:** Approved — ready for implementation planning
**Supersedes:** sizing + threshold sections of `2026-06-14-aplus-confluence-scorer-design.md`

## Goal

Recalibrate the A+ confluence scorer so it produces actionable trades on a regular basis without sacrificing the conviction filter that motivates the system. The recalibration is purely a tuning change: feature definitions, structure selector, and the grading pipeline architecture are unchanged.

The motivating evidence:
- A re-grading of the 90-day backtest (932 setups, 81 trading days, Feb 17 → Jun 11, 2026) under the current code produces **zero A+ and zero A grades**.
- The same backtest under the Jun 14 code produced 43 A grades. The regression is not a code change (verified by git audit: no commits to `domain/aplus/` since Jun 14). The shift is data-driven: `cat_vol_vix` dropped uniformly by 0.625 and `cat_liquidity` by 2.714 across all rows, attributable to a 15-day slide of the 2-year VIX history used by `domain/equity/vix_regime.py`.
- Today's full S&P 500 scan (2,925 candidates, 84 tickers, Schwab-backed chains) also produced zero A+/A — confirming the issue is calibration, not market regime.
- The score-level analysis reveals that the top 1% of composite (n=10) has a **40% win rate**, while the 95–99% bucket (n=37) has **62.2%**. More selective is empirically worse at the extreme.

## Non-goals

- No changes to feature extraction (`domain/aplus/features.py` math is preserved).
- No changes to category definitions.
- No changes to the structure selector (`domain/aplus/structure.py`).
- No changes to the underlying scanner pipeline.
- No backfill or correction of the historical VIX series; future drift is handled by basing thresholds on empirical percentiles rather than fixed cutoffs.
- No new strategies (credit spreads, naked options, iron condors remain out of scope).
- No stock-ownership comparison (that capability exists separately in `domain/stock_scanner/` and `scripts/trade_compare.py`).
- No auto-execution.

## Acceptance criteria

1. `python -m main aplus_watchlist` against the current scan produces at least one A+/A candidate when market conditions warrant (positive empirical lift validates the cutoff).
2. Re-running `python -m scripts.aplus_backtest_90d` against `scanner_90d_20260614_0819.csv` produces a non-empty A+ and A bucket with a 1-day win rate ≥ 53% on each (the A tier of the original empirical distribution).
3. VIX expansion regime forces grade F regardless of composite. Verified by unit test on a synthetic candidate with `vix_regime='expansion'`.
4. Category weights sum to 1.0 exactly.
5. Existing 434+ test suite still passes. New tests cover the threshold boundaries and the expansion kill.
6. The spec file from 2026-06-14 is updated with a forward reference to this document; sizing and threshold sections are marked superseded.

## Changes

### 1. Composite weights (lift-derived)

Pulled from the 90-day backtest top-vs-bottom-tertile 1-day win-rate lift, with liquidity preserved at 20% because the backtest neutralizes it.

```
                  Before    After    Δ
Technical:         25%      35%     +10
Catalyst:          15%      25%     +10
Liquidity:         20%      20%      0
Macro/Breadth:     15%      10%      -5
Vol/VIX:           25%      10%     -15
                  ────     ────
TOTAL:            100%     100%
```

Vol/VIX drops from 25% to 10% because its tertile lift is +0.3 percentage points (essentially zero). The regime kill captures the actual signal and is preserved as a hard floor (below).

Macro/Breadth drops from 15% to 10% (lift +0.5pp, near zero).

### 2. Grade thresholds (percentile-derived)

Cutoffs are now defined by the empirical composite distribution rather than fixed absolute numbers. This makes the grade ladder self-correcting against future helper-data drift.

```
Grade  Composite cutoff   Empirical %ile    Category floor    Other gates
A+     ≥ 75               top 5%            every cat ≥ 6     not vix expansion
A      ≥ 74               top 15%           every cat ≥ 6     not vix expansion
B+     ≥ 67               top 50%           none              display only
B      ≥ 60               next 30%          none              display only
F      < 60   OR vix_regime = 'expansion'
```

Category floors drop from ≥8/≥7 to a uniform ≥6 because the historical data never produced any candidate with every category ≥ 7 (the floors were too strict for the actual distribution).

### 3. Expansion-regime hard kill

If `candidate.vix_regime == 'expansion'`, the grade is forced to F regardless of composite or category scores.

Justification: in the 90-day backtest, expansion regime had a 31.7% 1d win rate (n=41) vs. 48.9% in neutral. This is the single strongest categorical signal in the data set. The kill is a separate gate from the Vol/VIX weight so the rest of the score can still discriminate within neutral and contraction regimes.

Implementation: a check at the top of `assign_grade(cs, vix_regime)` in `domain/aplus/grading.py`. The function signature gains a `vix_regime` parameter; callers in `scripts/aplus_watchlist.py` and `scripts/aplus_backtest_90d.py` are updated to pass it.

### 4. Sizing rules ($10k account)

Replaces the $1,000 sizing block of the 2026-06-14 spec. Quarter-Kelly haircut applied because the win rate has only n=37 historical samples at A+; the 95% CI on 62.2% is 46–76%.

```
A+ trade:    7–10% account risk   ($700–1,000)
A  trade:    4–5%  account risk   ($400–500)
B+ trade:    not traded (watchlist only)
B  trade:    not output

Max concurrent positions:           5      (was 3)
Max daily new entries:              2      (was 1)
Max same-ticker positions:          2
Max consecutive losses:             3 → 48h cooldown   (was 2)
Max daily drawdown:                 -3%  ($300)
Max weekly drawdown:                -7%  ($700)  → review all open positions
VIX expansion regime:               No new entries; existing positions follow their per-trade SL
```

The sizing percentages apply to whichever structure the existing `domain/aplus/structure.py` selector chooses for each candidate (long_premium or debit_spread). Per-trade exit rules unchanged from the original spec:
- Long premium: TP +50% of premium paid, SL -50% of premium paid, time stop at 5 DTE.
- Debit spread: TP +50% of max profit, SL = full debit paid, time stop at 5 DTE.

## Architecture

No new modules. Files changed:

```
domain/aplus/grading.py        — accept vix_regime, apply hard kill, new thresholds
domain/aplus/scoring.py        — new weights in CategoryScores.composite()
scripts/aplus_watchlist.py     — pass vix_regime to assign_grade, update _SIZING_PCT,
                                  update concurrent/daily/cooldown rules
scripts/aplus_backtest_90d.py  — pass vix_regime to assign_grade
tests/domain/aplus/test_grading.py    — new boundary + expansion tests
tests/domain/aplus/test_scoring.py    — new weights expected in composite()
tests/scripts/test_aplus_watchlist.py — sizing rule output, max-positions check
docs/superpowers/specs/2026-06-14-aplus-confluence-scorer-design.md
                              — front-matter pointer to this spec
```

No edits to features.py, structure.py, market_context.py, or types.py.

## Data flow (unchanged)

```
output/scans/latest.json
   ↓
domain/aplus/features.py  — same feature extraction
   ↓
domain/aplus/scoring.py   — new weights in composite()
   ↓
domain/aplus/grading.py   — new thresholds + vix expansion hard kill
   ↓
filter to A+/A
   ↓
domain/aplus/structure.py — unchanged (long_premium vs debit_spread selector)
   ↓
scripts/aplus_watchlist.py — new sizing percentages + concurrent caps
```

## Validation plan

The recalibration is data-driven, so validation is by backtest and forward paper test.

### Phase 1 — backtest verification (pre-merge)

Re-run `aplus_backtest_90d` and confirm:
- A+ bucket non-empty, n ≥ 30
- A bucket non-empty, n ≥ 80
- A+ 1d win rate ≥ 53% (CI lower bound)
- Expansion-labeled rows have grade=F
- Composite-decile relationship is monotonic at the top end

### Phase 2 — forward paper test (post-merge, 30 calendar days)

- Paper-trade every A+/A signal that fires.
- Log to `output/aplus/paper_trades.csv` (entry, exit, P&L, grade, structure).
- Target ≥ 15 paper trades before review.
- If A+ win rate falls below 50% over 30+ paper trades, return to brainstorming with new evidence.

### Phase 3 — half-size live (post-paper)

If paper P&L is positive after 30 trades, begin live at half sizing (3.5–5% A+, 2–2.5% A).

## Error handling

- Missing `vix_regime` on a candidate → treat as neutral (warn once per run).
- Composite NaN → grade F.
- Any category NaN → that category floor fails → grade F.
- Empty scan input → empty watchlist with explanatory message (unchanged).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Helper data continues to drift (e.g. VIX 2y window slides further) | Percentile-based thresholds self-correct; absolute cutoffs do not. |
| Backtest sample (n=37 at A+) overstates win rate | Quarter-Kelly sizing absorbs CI uncertainty. Forward-test gates scale-up. |
| Expansion regime kill never fires in production | Acceptable — kill is asymmetric protection, not a profit driver. |
| Live distribution of vol_vix scores widens, exposing the 10% weight as too low | Forward test catches regime-level edge degradation; weights are easy to tune. |
| Day-over-day OI feature (`liq_oi_change_dod`) caches stale data | Same as previous spec; first-run handles missing cache, subsequent runs work normally. |
| 5-position concurrent cap leaves edge on the table when many A+ fire same day | Daily-new-entries cap of 2 already throttles; revisit after Phase 2 paper test. |

## Open questions

None at this design stage. Empirical questions (true win rate at each grade, whether 5-day holds erode A+ edge) are resolved by Phase 2 paper-trading.
