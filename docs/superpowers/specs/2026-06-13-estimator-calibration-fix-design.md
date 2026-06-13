# Estimator Calibration Fix — Design

**Date:** 2026-06-13
**Author:** Sarthak (with Claude)
**Status:** Approved (architecture sign-off received)

## Goal

Make the four fair-probability estimators in `engine/prediction/` produce arithmetically correct probabilities for any `(spot, strike, σ, hours_to_expiry)` input. "Done" is defined by an automated calibration test suite — all four estimators must pass every test.

The motivating bug: Sprint 2's backtest revealed all four estimators output `fair_prob ≈ 0.55` regardless of strike, causing the strategy to lose 95% of trades by buying far-OTM contracts at "claimed +50% edge." The math machinery is structurally correct (lognormal CDF, z-scores) so the bug is in σ unit consistency or input scaling. This spec proves the bug is fixed without committing to where it lives.

## Non-goals

These are explicitly out of scope for this spec:

- **Backtest profitability.** Whether the Kalshi prediction-market strategy is profitable after correct calibration is a separate investigation. This spec does not run the 90-day backtest.
- **Re-architecting estimators.** The four-estimator design (vol_model, GARCH, historical_bootstrap, realized_vol) stays. Only the math and inputs change.
- **Replacing adjustments.** The four adjustments (favorite-longshot, round-number, mean-reversion, time-of-day) are untouched.
- **σ forecasting accuracy.** Whether the σ values fed into each estimator are good forecasts of future BTC vol is out of scope. The tests verify that *given* a σ, the resulting probability is correct.
- **Restructuring the watchlist/backtest scripts.** Only the estimator math is in scope.

## Acceptance criteria

The fix is done when all of the following hold:

1. All single-estimator calibration sanity tests pass (group a, ~32 tests).
2. All cross-estimator consistency tests pass (group b).
3. The realistic-regime grid (group c) shows monotone, well-shaped probability curves with no constant-output rows.
4. The existing 416-test baseline still passes (no regressions).
5. The fix is committed and pushed to the existing PR branch.

## Architecture

This is a **test-first** fix. No new production modules. Only edits to existing estimator files in `engine/prediction/` and new test files.

```
options-scanner/
├── engine/prediction/                     # EDITED — estimator bug fixes
│   ├── vol_model_estimator.py
│   ├── garch_estimator.py
│   ├── historical_bootstrap.py
│   └── realized_vol_estimator.py
├── tests/engine/prediction/
│   ├── test_estimator_calibration.py      # NEW — groups (a) and (b)
│   └── test_estimators.py                 # EXISTING — kept, no changes required
└── scripts/
    └── calibration_grid.py                # NEW — group (c) inspection script
```

No changes to `domain/prediction/`, `data/sources/kalshi_*`, or any script other than the new `calibration_grid.py`.

## Test taxonomy (the meat)

Three groups of tests, all parameterized so they run in seconds.

### Group (a) — Single-estimator calibration sanity

Each estimator must satisfy the following properties for any `(spot, strike, σ, T)` input. These become parameterized pytest cases. Approximately 8 properties × 4 estimators = ~32 tests.

```
ATM strike (strike = spot):                    |P − 0.5| < 0.01
Far OTM   (strike = spot × 1.5, σ = 0.005/hr, T = 1h):   P < 0.001
Far ITM   (strike = spot × 0.5, σ = 0.005/hr, T = 1h):   P > 0.999
Symmetry:                                      P(>X) + P(<X) ≈ 1.0  (tolerance 0.005)
Monotone in strike (above-side):               P(strike + ε) ≤ P(strike) for ε > 0
Monotone in σ:                                 |P − 0.5| decreases as σ rises
Monotone in T (OTM):                           |P − 0.5| decreases as T rises
Boundary σ → 0:                                P → 1 if spot > strike, else 0
Boundary T = 0:                                P → 1 if spot > strike, else 0
```

Tolerance values are chosen so that floating-point rounding doesn't cause flakiness while still catching the "all 0.55" bug.

### Group (b) — Cross-estimator consistency

Given the **same σ** input fed to all four estimators (bypassing each estimator's σ-source step — vol_model regime, GARCH fit, bootstrap, RV — and injecting a known σ instead), all four estimators must agree on the same `(spot, strike, T)` within ±2 percentage points.

This isolates "is the probability math right" from "is the σ forecast right." If estimators disagree by more than ±2% on identical inputs, one of them has a math bug.

To make this testable, each estimator may need a small refactor: extract the probability computation from the σ-sourcing logic so the probability step can be called with an explicit σ. This is a minimal change — the existing public API stays.

### Group (c) — Realistic-regime grid (visual inspection)

For realistic BTC inputs:

```
σ ∈ {0.001, 0.003, 0.005, 0.01} per hour
T ∈ {0.5, 1, 4, 24} hours
strike ∈ {spot × 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1}
```

Each combination runs through each estimator. Outputs go to `output/calibration/grid_YYYYMMDD_HHMM.csv`. No automated assertion — the script writer (and reviewer) eyeballs the grid to confirm:

- Outputs vary across strike (catches the "all 0.55" bug)
- Outputs are monotone in expected directions
- Far-OTM probabilities are small (< 5% for σ=0.005, T=1, strike = spot × 1.1)
- Different estimators broadly agree

The CSV is committed to `output/calibration/` so future runs can be diffed against historical baselines.

## Fix workflow

1. Add the test suite first. Run it. Most tests will fail.
2. Pick the first failing test on the first estimator. Trace the math to find the bug.
3. Fix the bug. Re-run the suite for that estimator.
4. Repeat until that estimator passes all of group (a).
5. Move to the next estimator.
6. After all four pass individually, run group (b) cross-estimator consistency tests.
7. Run group (c) grid generation. Visually inspect.
8. Run full project pytest to confirm no regressions in the existing 416 tests.

### Bugs we expect to find (best-guess priors, not commitments)

These are written down so the fixer knows where to look first, not as scope commitments:

- σ unit mismatch (daily σ used as hourly somewhere in the call chain)
- `historical_vol_quantiles` derived from a too-coarse aggregation window
- Callers passing the wrong-frequency series to `minute_returns_last_60` (e.g., hourly data labeled as minute)
- Risk-neutral drift correction applied in the wrong sign direction
- GARCH fit returning daily-frequency forecasts that aren't downscaled to hourly

## Error handling

Each estimator must handle pathological inputs without raising:

- `T = 0`: return `1.0` if `spot > strike`, else `0.0`
- `σ = 0`: same as above
- `σ` negative: raise `ValueError` (programming error, not data error)
- `strike` ≤ 0: raise `ValueError`
- `spot` ≤ 0: raise `ValueError`
- Empty/NaN return series: return `0.5` (no information)

These behaviors become test cases in group (a).

## Testing strategy

Beyond the three test groups defined above:

- All existing tests in `tests/engine/prediction/` continue to pass.
- The 416-test baseline runs at the end as a final regression check.
- The realistic-regime grid CSV is committed so reviewers can diff against expected output.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Multiple bugs compound, making any single fix appear to make things worse | Test suite catches regressions per-estimator. Fix one estimator fully before moving to the next. |
| Tolerance values too loose hide real bugs | Tolerances chosen tightly enough that the "all 0.55" pathology is caught. Documented in the test file. |
| Cross-estimator agreement test fails because different estimators legitimately model differently (e.g., Student-t vs normal) | Tolerance for cross-estimator consistency is ±2 percentage points, which accommodates legitimate Student-t vs normal differences while catching gross math bugs. |
| Refactor to extract σ-injection breaks adjustment callers | Public API is preserved. Internal extraction is additive (new helper functions, existing entry points unchanged). |

## Open questions

None at sign-off. Architecture is approved. The empirical question (which bugs exist) is resolved by writing and running the tests.
