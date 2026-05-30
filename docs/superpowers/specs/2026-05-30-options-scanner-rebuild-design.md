# Options Scanner Rebuild — Design Spec

**Date:** 2026-05-30
**Author:** Sarthak + Claude (brainstorming session)
**Status:** Draft — awaiting user review

---

## 1. Goal & Motivation

Rebuild the options scanner to:

1. **Remove Robinhood dependency** — `robin_stocks` is a reverse-engineered, ToS-violating, MFA-fragile scraper of Robinhood's private endpoints. Replace with free, no-auth alternatives.
2. **Add premium-buying mode** — current scanner only finds *overpriced* premium (seller's edge). Add the symmetric path: when realized vol > implied vol, options are underpriced — favor *buying*.
3. **Adopt a layered architecture** — current code couples data fetching, strategy logic, and scoring inside a few large functions. Untangle into clean layers so future expansion (swing-equity scanner, additional strategies) is additive, not a rewrite.
4. **Drop intraday cadence in favor of EOD** — both yahooquery and yfinance are ~15–20 min delayed; intraday scans on delayed data add complexity without value for 3–45 DTE position sizing.
5. **Dynamic universe** — auto-populate from S&P 500 daily, filtered by liquidity and options availability, instead of a hardcoded 12-ticker list.

## 2. Locked Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | **Migration + premium-buying mode** (sell when IV >> RV; buy when RV >> IV) |
| Cadence | **Single EOD scan** (~16:15 ET); manual or via OS cron |
| Strategy abstraction | **Strategy is a separate concept** — `Contract` is pure data; `Strategy` subclasses own per-direction math |
| Architecture | **Layered redesign** (`data/` → `domain/` → `engine/` → `pipeline/` → `ui/`) |
| Universe | **Dynamic S&P 500 screener** with liquidity + options filters, daily cache, hardcoded fallback |
| Universe defaults | top_n=50, min_avg_volume=1,000,000 shares/day, `--tickers` CLI override |

## 3. Library Stack

| Layer | Library | Purpose |
|---|---|---|
| Options chains (primary) | `yahooquery` | Stable Yahoo endpoints; multi-expiration option_chain |
| Options chains (fallback) | `yfinance` | Different scrape path → resilient when yahooquery fails |
| Prices (fallback²) | `stooq` via `pandas.read_csv` | Last-resort daily bars, no key |
| Greeks | `py_vollib` | Fast Black-Scholes Greeks from IV |
| Technical indicators | `pandas-ta` | Replaces hand-rolled RSI/SMA in current screener |
| Earnings | `yahooquery` | `Ticker.calendar_events` |
| Vol/stat | `arch`, `scipy`, `numpy`, `pandas` | Unchanged from current code |
| S&P 500 list | `pandas.read_html` + Wikipedia | Universe builder |
| UI | `streamlit` | Same, but no auto-refresh |
| Testing | `pytest`, `hypothesis` (selective), `pytest-mock` | Standard |

**Removed:** `robin_stocks`, `apscheduler`, `exchange_calendars` (no longer needed without intraday).

## 4. Module Layout

```
options-scanner/
├── data/                       # I/O only — fetch and adapt external data
│   ├── sources/
│   │   ├── base.py                   # DataSource ABC, RawContract dataclass
│   │   ├── yahooquery_source.py      # primary
│   │   ├── yfinance_source.py        # fallback
│   │   └── stooq_source.py           # fallback² (prices only)
│   ├── adapters.py             # RawContract → domain.Contract; computes Greeks via py_vollib
│   └── fallback.py             # fetch_with_fallback(sources, method, *args)
├── domain/                     # pure logic, no I/O
│   ├── contract.py             # Contract (frozen dataclass, no strategy field)
│   ├── strategy.py             # Strategy ABC + NakedPut / NakedCall / LongPut / LongCall
│   ├── signals.py              # compute_regime → Regime(rv_30, iv_atm, ratio, favored)
│   └── greeks.py               # py_vollib wrapper
├── engine/                     # operates on domain types
│   ├── pop_models.py           # 4 PoP models, all parameterized by Strategy
│   ├── stress.py               # compute_stress(contract, strategy, sigma) → StressResult
│   ├── risk_filters.py         # per-strategy FilterSet; apply_filters → FilterResult
│   └── scorer.py               # composite_score, label_from
├── pipeline/
│   ├── universe.py             # KNOWN_GOOD_FALLBACK list
│   ├── universe_builder.py     # S&P 500 fetch + filter + cache
│   ├── earnings.py             # has_earnings_within
│   └── run_scan.py             # orchestrator
├── ui/
│   ├── dashboard.py            # Streamlit (no auto-refresh, no scheduler hook)
│   └── exporter.py             # CSV + JSON writer
├── cache/                      # filesystem cache (gitignored)
│   └── universe/                     # daily universe JSON
├── output/scans/               # gitignored; scan_YYYYMMDD_HHMM.csv|json + latest.json
├── tests/                      # mirrors source tree
├── main.py                     # argparse entry: `scan` | `dashboard` | `universe`
└── requirements.txt
```

## 5. Domain Layer

### `Contract` (pure data)
```python
@dataclass(frozen=True)
class Contract:
    ticker: str
    expiration: date
    strike: float
    option_type: Literal['put', 'call']   # contract's nature, not strategy
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    dte: int
    spot_price: float
```

### `Strategy` abstraction
```python
class Strategy(ABC):
    name: str
    direction: Literal['sell', 'buy']
    option_type: Literal['put', 'call']

    @abstractmethod
    def applies_to(self, c: Contract) -> bool: ...
    @abstractmethod
    def breakeven(self, c: Contract) -> float: ...
    @abstractmethod
    def profit_condition(self, S_T: float, c: Contract) -> bool: ...
    @abstractmethod
    def expected_value(self, c: Contract, pop: float, stress: StressResult) -> float: ...
    @abstractmethod
    def margin_estimate(self, c: Contract) -> float: ...
```

Four concrete classes:
- `NakedPut`: sell put; profitable if `S_T > K`; delta range 0.05–0.35; breakeven `K - mid`
- `NakedCall`: sell call; profitable if `S_T < K`; delta range 0.05–0.35; breakeven `K + mid`
- `LongPut`: buy put; profitable if `S_T < K - mid`; delta range 0.30–0.55; breakeven `K - mid`
- `LongCall`: buy call; profitable if `S_T > K + mid`; delta range 0.30–0.55; breakeven `K + mid`

### `Regime` signal
```python
@dataclass
class Regime:
    ticker: str
    rv_30: float
    iv_atm: float
    rv_iv_ratio: float
    favored: list[Literal['sell', 'buy']]

# rv_iv_ratio < 0.8 → favor SELL
# rv_iv_ratio > 1.2 → favor BUY
# 0.8 ≤ ratio ≤ 1.2 → both, scoring sorts
```

## 6. Engine Layer

### PoP models (all parameterized by Strategy)
- `pop_delta(c, strategy)` — sellers: `1 - |Δ|`; buyers: `|Δ|`
- `pop_black_scholes(c, strategy, r)` — uses `strategy.breakeven` not just `K`
- `pop_historical(c, strategy, log_returns)` — rolling DTE-day terminal prices, count `strategy.profit_condition(S_T)`
- `pop_garch_mc(c, strategy, log_returns, n_paths)` — GARCH(1,1) + Student-t MC, count `strategy.profit_condition(S_T)`
- `blend_pop` — same weights as today: 20% Δ + 25% BS + 20% historical + 35% GARCH-MC

### Stress
- Sellers: intrinsic loss minus premium received at 1σ / 2σ adverse moves + 5th-percentile expiry
- Buyers: premium paid minus intrinsic gain, floored at `-mid` (limited risk)

### Expected value (per strategy)
- **Sellers** (existing formula): `ev = mid * p_blended - max_stress_loss * (1 - p_blended)` where `max_stress_loss = abs(min(stress_2sd, 0))`
- **Buyers**: `ev = expected_profit_if_itm * p_blended - mid * (1 - p_blended)` where `expected_profit_if_itm` is the mean payoff at expiry across GARCH-MC paths that satisfied `profit_condition`, minus `mid`. Loss side is capped at `-mid` (buyer's risk ceiling).

### Filter sets per strategy
```python
FILTER_SETS = {
    'naked_put':  FilterSet(max_spread=0.20, min_vol=100, min_oi=500, delta=(0.05,0.35), min_pop=0.70, min_ev=0.0,  max_stress_mult=3.0),
    'naked_call': FilterSet(max_spread=0.20, min_vol=100, min_oi=500, delta=(0.05,0.35), min_pop=0.70, min_ev=0.0,  max_stress_mult=3.0),
    'long_put':   FilterSet(max_spread=0.20, min_vol=100, min_oi=500, delta=(0.30,0.55), min_pop=0.40, min_ev=0.10, max_stress_mult=inf),
    'long_call':  FilterSet(max_spread=0.20, min_vol=100, min_oi=500, delta=(0.30,0.55), min_pop=0.40, min_ev=0.10, max_stress_mult=inf),
}
```

### Scorer
- **Composite (0–100)** = `50 * pop_blended + 30 * ev_score + 20 * regime_alignment_score`
  - `ev_score = clip(ev / abs(max_stress_loss), 0, 1)` for sellers; `clip(ev / mid, 0, 1)` for buyers (EV normalized by risk capital)
  - `regime_alignment_score = 1.0` if strategy direction is the *sole* favored direction in regime; `0.6` if direction is one of two favored; `0.3` if direction is not favored (only possible at boundary)
- Labels: TRADE (filter pass + score ≥65) / WATCHLIST (filter pass + 50–64) / NO_TRADE (any filter fail OR score <50)
- Carries `reason_for` and `reason_against` text for dashboard explainability

## 7. Data Layer

### `DataSource` interface
```python
class DataSource(ABC):
    def fetch_spot(self, ticker: str) -> float: ...
    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series: ...
    def fetch_option_chain(self, ticker: str) -> list[RawContract]: ...
```

### Fallback chain
```python
def fetch_with_fallback(sources, method, *args):
    for src in sources:
        try: return getattr(src, method)(*args)
        except Exception: continue
    raise DataFetchError(...)
```

Order:
- Spot + history: `[YahooQuery, Yfinance, Stooq]`
- Option chain: `[YahooQuery, Yfinance]`

### Greeks
Always computed by `data/adapters.to_contract` via `py_vollib`. Source-provided Greeks (if any) are ignored — eliminates nullness/staleness bugs.

### Caching
- In-process LRU on `fetch_price_history` keyed by `(ticker, today)`
- Filesystem daily cache on universe builds (see Section 8)
- No other persistent cache

## 8. Pipeline Layer

### Universe builder (dynamic S&P 500)
```python
@dataclass(frozen=True)
class UniverseFilters:
    min_avg_volume: int = 1_000_000
    require_options_chain: bool = True
    min_iv_rank: float = 0.0
    top_n: int = 50
```

Flow:
1. Read S&P 500 constituents from Wikipedia (`pandas.read_html`)
2. For each: fetch 30-day avg volume + option_chain availability
3. Filter by `UniverseFilters`
4. Rank by liquidity, keep top_n
5. Write to `cache/universe/universe_YYYY-MM-DD.json`

Fallback: if Wikipedia scrape fails *and* no cache exists, use `KNOWN_GOOD_FALLBACK` (the original 12 tickers). Scan logs a warning.

First scan/day: 5–10 min. Cached same-day runs: instant.

### Earnings blackout
`pipeline/earnings.has_earnings_within(ticker, days)` via `yahooquery.Ticker(t).calendar_events`. Tickers with earnings ≤ N days out are added to `skipped['earnings_blackout']`.

### Scan orchestrator
```python
def run_scan(config: ScanConfig) -> ScanResult:
    sources = [YahooQuerySource(), YfinanceSource(), StooqSource()]
    strategies = [NakedPut(), NakedCall(), LongPut(), LongCall()]
    universe = build_universe_cached(config.universe_filters, sources)

    candidates, skipped = [], {}
    for ticker in universe:
        try:
            if has_earnings_within(ticker, config.earnings_blackout_days):
                skipped[ticker] = 'earnings_blackout'; continue
            history = fetch_with_fallback(sources, 'fetch_price_history', ticker, 365)
            spot    = fetch_with_fallback(sources, 'fetch_spot', ticker)
            raw_chain = fetch_with_fallback(sources[:2], 'fetch_option_chain', ticker)
            log_returns = np.log(history / history.shift(1)).dropna()
            regime = compute_regime(log_returns, raw_chain)

            for strategy in strategies:
                if strategy.direction not in regime.favored: continue
                for raw in raw_chain:
                    if raw.option_type != strategy.option_type: continue
                    contract = to_contract(raw, spot, config.risk_free_rate, config.today)
                    if contract is None or not strategy.applies_to(contract): continue
                    pop = run_pop_models(contract, strategy, log_returns, config)
                    stress = compute_stress(contract, strategy, contract.implied_volatility)  # IV from yahooquery is already annualized
                    ev = strategy.expected_value(contract, pop.blended, stress)
                    fr = apply_filters(contract, strategy, pop.blended, ev, stress)
                    score = composite_score(pop.blended, ev, stress, regime)
                    candidates.append(ScoredCandidate(...))
        except DataFetchError as e:
            skipped[ticker] = f'fetch_failed: {e}'

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return ScanResult(datetime.now(), config, candidates, skipped)
```

Per-ticker failures land in `skipped`; never kill the scan.

## 9. UI Layer

### Exporter
Writes `output/scans/scan_{ts}.csv` (flat rows), `scan_{ts}.json` (full result), copies the JSON to `latest.json`.

### Dashboard
- **Header**: last scan timestamp; counts by label; regime summary across universe (e.g., 22 favoring sell / 18 favoring buy / 10 neutral)
- **Tabs**: All / Sell / Buy / TRADE only / WATCHLIST only
- **Table**: sortable by composite_score; color-coded by label; columns: ticker, strategy, strike, exp, bid/ask, mid, blended_pop, ev, score, label
- **Expand row**: all 5 PoP values, stress scenarios, margin, reason_for, reason_against, regime context
- **Manual "Refresh" button** — re-reads `latest.json`. No auto-refresh.

### `main.py` (entry point)
```
python -m main scan                          # default top-50 dynamic universe
python -m main scan --top 100
python -m main scan --min-volume 5000000
python -m main scan --tickers SPY,QQQ        # bypass screener
python -m main scan --then-dashboard
python -m main dashboard
python -m main universe rebuild              # force universe cache rebuild
```

## 10. Testing Strategy

| Layer | Mocks? | Tests | Notes |
|---|---|---|---|
| `domain/` | None | ~30 | Pure-fn tests; analytic answers + edge cases |
| `engine/` | None | ~25 | Synthetic Contract/Strategy; converge to BS on lognormal |
| `data/adapters` | None | ~10 | Pure fns |
| `data/sources/*` | HTTP mocks | ~15 | `responses` / `pytest-mock`; frozen JSON fixtures |
| `data/fallback` | DataSource stubs | ~5 | |
| `pipeline/*` | Mocked sources | ~10 | Inject fake sources; verify skipped/sorted/regime branching |
| `integration/` | None — real network | 3–5 | `@pytest.mark.slow`, opt-in via `pytest -m slow` |

**Total ~95–100 tests.** Conventions:
- No mocks inside `domain/` or `engine/` — if a test needs one, abstraction is wrong
- `np.random.seed(42)` preserved for GARCH-MC determinism
- Selective `hypothesis` for math invariants (`pop ∈ [0,1]`)
- Frozen yahooquery fixture: `tests/fixtures/spy_2026_05_29.json`

Coverage targets:
- `domain/`, `engine/`: 95%+
- `data/sources/`: 80%+
- `pipeline/`: 85%+
- `ui/`: smoke only

## 11. Demo Run Definition

Three independently verifiable steps.

### Step 1: Universe build
```bash
python -m main universe rebuild
```
**Pass**: `cache/universe/universe_2026-05-30.json` exists with ≥40 tickers; SPY/QQQ/AAPL/MSFT present.

### Step 2: Scan
```bash
python -m main scan
```
**Pass**: completes without exceptions; ≥1 TRADE candidate; `skipped` reasons are all expected (earnings_blackout / fetch_failed).

Summary print example:
```
Scanned 47 tickers (3 skipped). 12 TRADE / 38 WATCHLIST / 416 NO_TRADE.
Top: SPY long_call $620 2026-06-13 score 78.
```

### Step 3: Dashboard
```bash
python -m main dashboard
```
**Pass**: opens at `http://localhost:8501`; header + tabs + expand-row all functional; both Sell and Buy candidates visible.

### Demo acceptance checklist
- [ ] All unit tests pass (`pytest`)
- [ ] Slow tests pass (`pytest -m slow`)
- [ ] Universe builder produces a non-trivial list
- [ ] Scan produces both Sell and Buy candidates (proves new buyer path works)
- [ ] Dashboard renders the buyer column
- [ ] Output JSON round-trips through `ScanResult` dataclass
- [ ] README updated with new commands + new data-source disclaimer

## 12. Removed from Current Code

- `scanner/scheduler.py` — no scheduler in EOD model
- `scanner/clock.py` — no market-hours logic needed for EOD
- `scanner/chain_fetcher.py` — replaced by `data/sources/yahooquery_source.py` + `data/adapters.py`
- `Contract.strategy` field — strategy is no longer a property of the contract
- `robin_stocks` dependency, `ROBINHOOD_USERNAME` / `ROBINHOOD_PASSWORD` env vars
- `apscheduler` dependency
- `exchange_calendars` dependency
- Streamlit session-state scheduler initialization in `main.py`

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| yahooquery breaks | yfinance fallback; fixture-frozen tests guard schema regressions |
| Wikipedia S&P 500 page changes structure | One-line `pandas.read_html` parse; if it fails, hardcoded `KNOWN_GOOD_FALLBACK` |
| Scan exceeds reasonable runtime as universe grows | `top_n` cap; daily cache; ThreadPoolExecutor opportunity flagged for future |
| Greeks from py_vollib differ from market | Documented; acceptable for screening (not pricing) |
| Buyer path generates false positives in choppy markets | Regime gate (`rv_iv_ratio > 1.2`) + min_ev filter + min_pop filter — three layers |
| Tests become slow as count grows | Integration tests gated behind `-m slow`; unit tests stay pure-fn fast |

## 14. Open Questions

None at design time. Implementation may surface library-specific issues that get logged as the work proceeds.

## 15. Out of Scope (deferred follow-ups)

- **Mode C — Swing-equity scanner** for fair-value / underpriced underlyings. Distinct module; follows the same layered pattern as a sibling to options scanning.
- **Live trading / paper trading hookup** (Alpaca, Tradier). Read-only research scope preserved.
- **Multi-leg strategies** (vertical spreads, iron condors). The `Strategy` ABC supports them; concrete classes deferred.
- **Persistent metrics / historical performance tracking** of past scans.
- **Backtesting harness** (vectorbt) — deferred until we have hindsight on the live scanner's behavior.
