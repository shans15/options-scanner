# Saty Technical Filter — Design

**Date:** 2026-06-06
**Author:** Sarthak (with Claude)
**Status:** Approved (pending spec review)

## Goal

Add a technical-signal pre-filter to the EOD options scanner that narrows the universe to tickers showing one of four Saty-style technical setups, and gates which long strategy (LongCall vs LongPut) is evaluated per surviving ticker based on the setup's direction.

The scanner currently evaluates all four strategies (NakedPut, NakedCall, LongPut, LongCall) on every universe ticker, using only an RV/IV regime gate to bias direction. After this change, the user (long-premium only) sees a smaller, higher-conviction candidate list driven by directional technical setups.

## Scope

**In scope:**
- New module `domain/technical_signals.py` with four detectors and shared indicator helpers.
- New pipeline stage `pipeline/technical_filter.py` that filters the universe by detector matches.
- Modifications to `pipeline/run_scan.py` to insert the filter and gate strategy evaluation.
- Extension of `ScoredCandidate` with three optional setup fields.
- Output additions (CSV/JSON columns, dashboard column).
- CLI flag `--no-technical-filter` for debugging/legacy behavior.
- Tests at the densities the codebase already maintains (~unit + integration + regression).

**Out of scope:**
- Changes to `composite_score` weights. The 50/30/20 split stays. Technical conviction is encoded in filter membership, not in scoring (avoids double-counting).
- Changes to the four PoP models, stress, or risk filters.
- Removal of NakedPut/NakedCall strategy classes. They remain in the codebase and will continue to fire when a direction-aligned setup matches (NakedPut on bullish setups, NakedCall on bearish). The user will simply ignore them in output. We're keeping the codebase usable by accounts that have naked-selling approval.
- Real-time / intraday scanning. EOD only.
- New data sources. All detectors operate on existing OHLCV history.

## Architecture

### Current data flow (`pipeline/run_scan.py:62-82`)

```
build_universe_cached (S&P 500 → liquidity → top N)
  └─ for each ticker:
       fetch history + spot + chain         ← chain is the expensive call
       compute_regime (RV/IV)
       for each of 4 strategies:
           if strategy.direction in regime.favored:
               evaluate contract → score
```

### New data flow

```
build_universe_cached (unchanged)
  └─ [NEW] filter_by_technicals(universe, sources):
       for each ticker:
           fetch history ONLY (cheap)
           detect_setups(history) → list[TechnicalSetup]
           keep if non-empty
       output: dict[ticker → list[TechnicalSetup]]
  └─ for each surviving ticker:
       fetch spot + chain (now only for ~5-15 tickers, not 50)
       compute_regime (kept — feeds composite score's regime component)
       for each strategy:
           if not _strategy_matches_setup(strategy, setups): continue
           evaluate contract → score with setup fields attached
       output is enriched with setup_name / setup_direction / setup_strength
```

### Key design decisions

1. **History fetch split out of main loop.** History is cheap (single OHLC pull); option chains are expensive (multiple expirations × strikes × IV recompute). Filtering on history before the chain pull is the entire speed win.
2. **Tickers may match multiple setups.** Output is `list[TechnicalSetup]` per ticker. The strongest setup is surfaced in the candidate row; all contribute to the eligible-direction set.
3. **Regime gate retained.** RV/IV still computed and feeds composite score's regime alignment term. For long premium, RV/IV ≥ 1 is desirable (you don't want to pay overpriced IV).
4. **Strategy gate composition.** `LongCall` is eligible only if some setup is bullish. `LongPut` only if some setup is bearish. `NakedPut` / `NakedCall` paths are technically still in the codebase but will be filtered out by the same gate (NakedPut needs bullish, NakedCall needs bearish — they map the same direction as the long counterparts).
5. **Deterministic detection.** Given history H and EOD date D, `detect_setups(H)` returns the same list. No randomness. Enables exact-match unit tests.
6. **`--no-technical-filter` flag.** Bypasses the new stage entirely; restores legacy behavior. Useful for regression debugging.

## Components

### `domain/technical_signals.py`

```python
@dataclass(frozen=True)
class TechnicalSetup:
    setup_name: Literal[
        'compression_breakout',
        'pullback_in_trend',
        'stage_2_breakout',
        'failed_breakdown_reversal',
    ]
    direction: Literal['bullish', 'bearish']
    strength: float                   # 0.0 - 1.0
    notes: str                        # human-readable rationale

def detect_setups(history: pd.DataFrame) -> list[TechnicalSetup]:
    """Run all four detectors against a daily OHLCV history.
    Returns 0+ setups. Requires len(history) >= 220 or returns []."""
```

**Internal helpers (private):**
```python
def _emas(close: pd.Series) -> dict[int, pd.Series]   # n ∈ {8,13,21,48,200}
def _atr14(high, low, close) -> pd.Series              # Wilder's ATR
def _phase_oscillator(close, ema21, atr14) -> pd.Series  # (close - ema21) / atr14
def _po_bandwidth_percentile(po: pd.Series, window=126) -> pd.Series
```

### Detector specifications

All operate on a 220+ bar daily OHLCV DataFrame. All produce `Optional[TechnicalSetup]` — the outer `detect_setups` aggregates non-None returns into a list.

**Detector 1 — Compression breakout**

Bullish conditions (all must hold on the last bar):
- `po_bbw_percentile[-1] < 0.20` (PO bandwidth in bottom quintile over 6 months — squeeze)
- `ema8 > ema13 > ema21` (short-term ribbon bullish)
- `close > ema48` (mid-term trend not against)
- `abs(close[-1] - close[-2]) < 0.5 * atr14[-1]` (not already extended)

Bearish: inverse — `ema8 < ema13 < ema21`, `close < ema48`, same compression and not-extended gates.

Strength: `1 - po_bbw_percentile[-1]` (tighter squeeze = higher conviction).

**Detector 2 — Pullback in trend**

Bullish (all on last bar):
- `ema8 > ema13 > ema21 > ema48` (full ribbon stacked bullish)
- `close > ema200` (long-term bias up)
- `ema21[-1] * 0.99 <= low[-1] <= ema13[-1] * 1.01` (low touched the 13-21 EMA zone)
- `-23.6 <= po[-1] <= 23.6` (oscillator in neutral / launch zone)
- `volume[-1] < 1.1 * volume[-21:-1].mean()` (pullback on lower volume — healthy)

Bearish: inverse ribbon, `close < ema200`, pullback to the EMA zone from below, same PO and volume gates.

Strength: `1 - abs(po[-1]) / 23.6` (closer to zero-line = cleaner entry).

**Detector 3 — Stage 2 breakout**

Bullish (all):
- `ema200[-1] > ema200[-30]` (long-term EMA rising)
- `close crossed above ema48 within last 5 bars` (mid-term breakout fresh)
- `po crossed above 0 within last 5 bars` (momentum confirmation)
- `close[-1] >= 0.90 * max(high[-252:])` (within 10% of 52-week high — near, not at)
- `min(po_bbw_percentile[-20:]) < 0.25` (came out of compression in the last month)

Bearish — Stage 4 breakdown: inverse. EMA200 falling, close crossed below EMA48 freshly, PO crossed below 0, within 10% of 52-week low, came out of compression.

Strength: `clip((close - ema48) / atr14, 0, 2) / 2` (more ATR-extension beyond breakout = stronger).

**Detector 4 — Failed breakdown reversal**

Bullish (all):
- `min(low[-5:]) < min(low[-25:-5])` (made a new 20-day low in last 5 bars)
- `close[-1] > min(low[-25:-5])` (reclaimed prior support)
- PO divergence: price low in window made a lower low while PO low made a higher low (compare bar of `argmin(low[-20:])` vs `argmin(po[-20:])`)
- `volume[-1] > 1.5 * volume[-21:-1].mean()` (reversal candle has conviction)

Bearish — failed breakout reversal: inverse. New 20-day high in last 5 bars, close back below prior resistance, PO divergence (higher high in price, lower high in PO), high reversal volume.

Strength: `min(1.0, volume[-1] / (1.5 * volume[-21:-1].mean()))`.

**Edge cases (all detectors):**
- `len(history) < 220` → return `None`
- NaN or zero `atr14[-1]` → return `None` (would divide by zero in PO)
- Empty / flat history → return `None`, no exception

### `pipeline/technical_filter.py`

```python
def filter_by_technicals(
    tickers: list[str],
    sources: list[DataSource],
) -> dict[str, list[TechnicalSetup]]:
    """For each ticker, fetch history and run detect_setups.
    Returns dict containing only tickers with >=1 setup matched.
    Tickers with fetch failure are silently dropped (logged at INFO)."""
```

**Failure handling:** wraps history fetch in try/except matching the existing `DataFetchError` pattern in `run_scan.py`. Failed fetches → ticker excluded, logged, no exception propagates.

### `pipeline/run_scan.py` changes

Three surgical edits:

1. After line 62 (`universe = build_universe_cached(...)`), insert:
   ```python
   setups_by_ticker = (
       filter_by_technicals(universe, sources)
       if config.use_technical_filter else
       {t: [] for t in universe}
   )
   ```

2. Replace loop iterator (line 66) — iterate `setups_by_ticker.keys()` instead of `universe`.

3. Inside the strategy loop, replace:
   ```python
   if strategy.direction not in regime.favored:
       continue
   ```
   with:
   ```python
   if config.use_technical_filter:
       setup_dirs = {s.direction for s in setups_by_ticker[ticker]}
       if not _strategy_matches_setup(strategy, setup_dirs):
           continue
   else:
       if strategy.direction not in regime.favored:
           continue
   ```

`_strategy_matches_setup` is a module-private helper:
```python
def _strategy_matches_setup(strategy: Strategy, setup_dirs: set[str]) -> bool:
    if isinstance(strategy, (LongCall, NakedPut)):
        return 'bullish' in setup_dirs
    if isinstance(strategy, (LongPut, NakedCall)):
        return 'bearish' in setup_dirs
    return False
```

When constructing each `ScoredCandidate`, pick the strongest setup matching the strategy's direction and populate the three new fields.

### `ScanConfig` extension

```python
@dataclass
class ScanConfig:
    ...existing fields...
    use_technical_filter: bool = True
```

### `engine/scorer.py` — `ScoredCandidate` extension

```python
@dataclass
class ScoredCandidate:
    ...existing fields...
    setup_name: Optional[str] = None
    setup_direction: Optional[str] = None
    setup_strength: Optional[float] = None
```

`composite_score` is unchanged. Setup data does not feed scoring (filter encodes conviction; scoring still ranks on PoP / EV / regime).

### `main.py` CLI flag

Add to `scan` subparser:
```python
s.add_argument('--no-technical-filter', action='store_true',
               help='Bypass Saty technical filter; use legacy regime-only gating')
```

In `cmd_scan`: `config.use_technical_filter = not args.no_technical_filter`.

### `ui/exporter.py` columns

CSV row gains: `setup_name`, `setup_direction`, `setup_strength`. JSON serialization mirrors the dataclass.

### `ui/dashboard.py`

Add a "Setup" column to the candidate table and a sidebar filter widget for setup name.

## Thresholds (locked)

| Detector | Knob | Value |
|---|---|---|
| Compression | PO bandwidth percentile cutoff | < 0.20 |
| Compression | "Not already extended" gate | < 0.5 × ATR from prior close |
| Pullback | PO band for "launch zone" | −23.6 to +23.6 |
| Pullback | Pullback volume threshold | < 1.1 × 20d avg |
| Stage 2 | "Fresh" crossover lookback | 5 bars |
| Stage 2 | Distance from 52w high | within 10% |
| Failed breakdown | New-low lookback | 20 bars |
| Failed breakdown | Reversal volume | > 1.5 × 20d avg |
| All detectors | Minimum history | 220 bars |

Tuning policy: leave as-is for v1, observe empirical hit counts after first run, then adjust.

## Error handling

- All detector failures (NaN, insufficient history, divide-by-zero) → return `None`. Never raise.
- Filter-stage fetch failures → log INFO, drop ticker from result dict.
- Empty `setups_by_ticker` (no setups across universe) → scan completes with zero candidates and a log line. Not an error.

## Testing

**Unit tests — `tests/domain/test_technical_signals.py`**
- 8 happy-path tests: each detector × each direction with synthetic OHLCV fixtures that hand-craft each condition's trigger.
- `len(history) < 220` returns `[]`.
- Flat / NaN history returns `[]`, no exception.
- Determinism: same fixture → identical setups list.

**Integration test — `tests/pipeline/test_technical_filter.py`**
- Mock 5 sources: 2 with bullish setups, 1 bearish, 2 with no signal.
- Assert: returned dict has 3 keys with the expected setups attached.
- One source raises `DataFetchError`: assert that ticker is dropped, others succeed.

**Regression test — `tests/pipeline/test_run_scan.py`**
- Add case with `use_technical_filter=False`: legacy behavior — same candidates as before this change.
- Add case with `use_technical_filter=True`: assert LongCall fires only on bullish-tagged tickers, LongPut only on bearish-tagged tickers, no candidates on no-signal tickers.

## File manifest

**New:**
- `domain/technical_signals.py`
- `pipeline/technical_filter.py`
- `tests/domain/test_technical_signals.py`
- `tests/pipeline/test_technical_filter.py`

**Modified:**
- `pipeline/run_scan.py` (filter stage insertion, strategy gate, ScanConfig field)
- `engine/scorer.py` (`ScoredCandidate` setup fields)
- `ui/exporter.py` (3 new columns)
- `ui/dashboard.py` (setup column + filter widget)
- `main.py` (`--no-technical-filter` flag)
- `tests/pipeline/test_run_scan.py` (2 new cases)

## Open questions

None at sign-off. All design decisions are locked. Threshold tuning is deferred to post-v1 empirical observation.
