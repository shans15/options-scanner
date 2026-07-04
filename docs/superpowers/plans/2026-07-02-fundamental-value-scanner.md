# Fundamental Value Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stand-alone monthly stock mispricing scanner that ranks Russell 2000 constituents by fundamental value using two parallel methodologies (sector-relative multiples + fundamental momentum divergence), overlaid with volume and short-interest signals.

**Architecture:** New `domain/value/` module for all math; new `data/sources/yahooquery_fundamentals.py` for the single data source; two CLIs (`value_watchlist`, `value_check`) drive the pipeline. Monthly Parquet cache stores raw fundamentals; both Role 1 (tag) and Role 2 (composite) scoring modes are computed in parallel every run for A/B comparison. Zero changes to existing A+ or scanner code.

**Tech Stack:** Python 3.9, `yahooquery` (already in repo), `pandas`, `pyarrow` (for Parquet cache), `pytest`.

---

## File Structure

```
domain/value/
├── __init__.py            NEW
├── types.py               NEW  dataclasses (ValuationInputs, SectorRelativeResult, ...)
├── fundamentals.py        NEW  raw payload → structured ValuationInputs (normalization)
├── sector_relative.py     NEW  Methodology A: peer-multiple comparison
├── fundamental_divergence.py  NEW  Methodology B: revenue/EPS vs price trend
├── volume_overlay.py      NEW  5d/20d volume ratio + RISING/FLAT/DECLINING tag
├── short_overlay.py       NEW  direction-aware short-interest scoring + tags
├── scoring.py             NEW  Role 1 tags + Role 2 composite (both in one pass)
└── report.py              NEW  assemble MispricingReport, serialize to JSON

data/sources/
└── yahooquery_fundamentals.py  NEW  batch fetch fundamentals + short interest

data/universe/
└── russell2000_constituents.json  NEW  committed snapshot; refreshed via script

scripts/
├── refresh_russell2000.py  NEW  pull iShares IWM CSV, write JSON list
├── value_watchlist.py      NEW  full monthly scan CLI
└── value_check.py          NEW  on-demand single-ticker lookup

cache/value/                   ← created at first run (gitignored)
output/value/                  ← created at first run (gitignored)

tests/domain/value/
├── __init__.py               NEW
├── test_types.py             NEW
├── test_fundamentals.py      NEW
├── test_sector_relative.py   NEW
├── test_fundamental_divergence.py  NEW
├── test_volume_overlay.py    NEW
├── test_short_overlay.py     NEW
├── test_scoring.py           NEW
└── test_report.py            NEW

tests/scripts/
├── test_refresh_russell2000.py  NEW
├── test_value_watchlist.py      NEW
└── test_value_check.py          NEW
```

**No file exceeds ~200 lines. Each has one responsibility.**

---

## Pre-flight (one-time before Task 1)

- [ ] **Step P1: Confirm branch state**

```bash
cd ~/options-scanner
git status --short
git branch --show-current
```

Expected: on `main` (or a feature branch you're working on). Untracked files may exist from prior work — do not touch them.

- [ ] **Step P2: Baseline tests pass**

```bash
python3 -m pytest -q 2>&1 | tail -5
```

Expected: matches whatever the current baseline is (may have 3 pre-existing failures in `tests/pipeline/test_universe_builder.py` — those are unrelated, skip them for gate purposes).

Record the pass count for post-implementation comparison.

- [ ] **Step P3: Confirm yahooquery and pyarrow are installed**

```bash
python3 -c "import yahooquery, pyarrow; print('yahooquery', yahooquery.__version__, 'pyarrow', pyarrow.__version__)"
```

Expected: prints versions. If `pyarrow` is missing:

```bash
python3 -m pip install pyarrow
```

- [ ] **Step P4: Create empty output/cache dirs and gitignore entries**

```bash
mkdir -p cache/value output/value
```

Check `.gitignore` at repo root — ensure `cache/` and `output/` are already ignored (they should be, matching existing patterns). If not:

```bash
echo -e "\n# Value scanner runtime\ncache/value/\noutput/value/" >> .gitignore
```

Commit if `.gitignore` changed:

```bash
git add .gitignore
git commit -m "chore: ignore value scanner cache and output dirs"
```

---

## Task 1: Dataclasses in domain/value/types.py

**Files:**
- Create: `domain/value/__init__.py` (empty)
- Create: `domain/value/types.py`
- Create: `tests/domain/value/__init__.py` (empty)
- Create: `tests/domain/value/test_types.py`

- [ ] **Step 1.1: Create empty `__init__.py` files**

```bash
touch domain/value/__init__.py tests/domain/value/__init__.py
```

- [ ] **Step 1.2: Write failing test for the type module**

Create `tests/domain/value/test_types.py`:

```python
"""Verify types.py exposes the dataclasses used across the value module."""
from domain.value.types import (
    ValuationInputs,
    SectorRelativeResult,
    FundamentalDivergenceResult,
    VolumeOverlayResult,
    ShortOverlayResult,
    Role1Score,
    Role2Score,
    ValuationSnapshot,
    MispricingReport,
)


def test_valuation_inputs_holds_all_fields():
    v = ValuationInputs(
        ticker='INTC', sector='Technology', price=24.11,
        forward_eps=2.87, revenue_ttm=54_000_000_000,
        revenue_ttm_1y_ago=48_000_000_000,
        eps_ttm=1.4, eps_ttm_1y_ago=1.29,
        book_value_per_share=21.9, enterprise_value=100_000_000_000,
        ebitda_ttm=15_400_000_000,
        shares_short=180_000_000, float_shares=1_040_000_000,
        avg_daily_volume_30d=29_000_000,
        volume_5d_avg=45_000_000, volume_20d_avg=38_000_000,
        price_1y_ago=29.55,
    )
    assert v.ticker == 'INTC'
    assert v.price == 24.11


def test_valuation_snapshot_combines_all_partial_results():
    """A ValuationSnapshot is the per-ticker container of all sub-results."""
    v = ValuationInputs(
        ticker='X', sector='Technology', price=10.0,
        forward_eps=None, revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None, book_value_per_share=None,
        enterprise_value=None, ebitda_ttm=None,
        shares_short=None, float_shares=None, avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None, price_1y_ago=None,
    )
    snap = ValuationSnapshot(
        ticker='X', inputs=v,
        sector_relative=None, fundamental_divergence=None,
        volume=None, short_overlay=None,
        role1=None, role2=None,
        skip_reason='no_fundamentals',
    )
    assert snap.ticker == 'X'
    assert snap.skip_reason == 'no_fundamentals'


def test_mispricing_report_has_four_lists():
    r = MispricingReport(
        run_timestamp_utc='2026-07-02T18:22:00Z',
        universe='russell2000',
        universe_size=2000,
        graded_size=1847,
        skipped=153,
        skipped_reasons={'missing_fundamentals': 98},
        role1_ranked_longs=[],
        role1_ranked_shorts=[],
        role2_ranked_longs=[],
        role2_ranked_shorts=[],
    )
    assert r.universe == 'russell2000'
    assert isinstance(r.role1_ranked_longs, list)
```

- [ ] **Step 1.3: Run failing test**

```bash
python3 -m pytest tests/domain/value/test_types.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ValuationInputs' from 'domain.value.types'`.

- [ ] **Step 1.4: Implement `domain/value/types.py`**

```python
"""Dataclasses used across the Value scanner.

Each unit is small and has one responsibility.  Importing modules
should depend only on this file for type information.

Per spec docs/superpowers/specs/2026-07-02-fundamental-value-scanner-design.md
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


VolumeTag = Literal['RISING', 'FLAT', 'DECLINING']
Direction = Literal['long', 'short']
Role1Priority = Literal[
    'HIGH_PRIORITY_SQUEEZE',
    'HIGH_PRIORITY',
    'PATIENT',
    'STALE_REASSESS',
    'SHORTS_BUILDING',
    'CLEAN_SHORT',
    'PATIENT_SHORT',
    'CROWDED_SHORT_AVOID',
    'NONE',
]


@dataclass(frozen=True)
class ValuationInputs:
    """Structured, per-ticker fundamentals + price + volume + short data.

    Any field may be None if the underlying data source did not
    supply it — downstream methodologies decide how to handle missing
    fields (skip vs neutral score)."""
    ticker: str
    sector: Optional[str]
    price: Optional[float]
    forward_eps: Optional[float]
    revenue_ttm: Optional[float]
    revenue_ttm_1y_ago: Optional[float]
    eps_ttm: Optional[float]
    eps_ttm_1y_ago: Optional[float]
    book_value_per_share: Optional[float]
    enterprise_value: Optional[float]
    ebitda_ttm: Optional[float]
    shares_short: Optional[float]
    float_shares: Optional[float]
    avg_daily_volume_30d: Optional[float]
    volume_5d_avg: Optional[float]
    volume_20d_avg: Optional[float]
    price_1y_ago: Optional[float]


@dataclass(frozen=True)
class SectorRelativeResult:
    score: float                       # 0-10
    pe_fwd: Optional[float] = None
    pe_fwd_peer_median: Optional[float] = None
    pe_discount_pct: Optional[float] = None
    ps_ttm: Optional[float] = None
    ps_ttm_peer_median: Optional[float] = None
    ps_discount_pct: Optional[float] = None
    pb: Optional[float] = None
    pb_peer_median: Optional[float] = None
    pb_discount_pct: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_ebitda_peer_median: Optional[float] = None
    ratios_used: int = 0


@dataclass(frozen=True)
class FundamentalDivergenceResult:
    score: float                       # 0-10
    revenue_growth_yoy_pct: Optional[float] = None
    eps_growth_yoy_pct: Optional[float] = None
    price_growth_yoy_pct: Optional[float] = None
    divergence_pp: Optional[float] = None


@dataclass(frozen=True)
class VolumeOverlayResult:
    score: float                       # 0, 5, or 10
    tag: VolumeTag
    ratio: Optional[float] = None
    vol_5d_avg: Optional[float] = None
    vol_20d_avg: Optional[float] = None


@dataclass(frozen=True)
class ShortOverlayResult:
    score: float                       # 0-10 (direction-aware)
    short_interest_pct: Optional[float] = None
    days_to_cover: Optional[float] = None
    short_interest_delta_pp: Optional[float] = None


@dataclass(frozen=True)
class Role1Score:
    combined_rank_score: float         # 0-10 (mean of sector_rel + fund_div where available)
    priority: Role1Priority


@dataclass(frozen=True)
class Role2Score:
    composite_score: float             # 0-100
    component_breakdown: dict          # {'sector_relative': 8.5, ...}


@dataclass(frozen=True)
class ValuationSnapshot:
    """All computed results for one ticker in one monthly run."""
    ticker: str
    inputs: ValuationInputs
    sector_relative: Optional[SectorRelativeResult]
    fundamental_divergence: Optional[FundamentalDivergenceResult]
    volume: Optional[VolumeOverlayResult]
    short_overlay: Optional[ShortOverlayResult]
    role1: Optional[Role1Score]
    role2: Optional[Role2Score]
    skip_reason: Optional[str] = None      # populated iff ticker was skipped entirely


@dataclass(frozen=True)
class MispricingReport:
    """Final output for a single monthly run."""
    run_timestamp_utc: str
    universe: str
    universe_size: int
    graded_size: int
    skipped: int
    skipped_reasons: dict[str, int]
    role1_ranked_longs: list[dict]
    role1_ranked_shorts: list[dict]
    role2_ranked_longs: list[dict]
    role2_ranked_shorts: list[dict]
```

- [ ] **Step 1.5: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_types.py -v
```

Expected: 3/3 PASS.

- [ ] **Step 1.6: Commit**

```bash
git add domain/value/__init__.py domain/value/types.py tests/domain/value/__init__.py tests/domain/value/test_types.py
git commit -m "$(cat <<'EOF'
feat(value): dataclasses for the fundamental value scanner

Foundation types shared by all value-module methodologies.  No
behavior yet — just the type surface downstream modules depend on.

Per spec 2026-07-02-fundamental-value-scanner-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fundamentals normalizer

**Files:**
- Create: `domain/value/fundamentals.py`
- Create: `tests/domain/value/test_fundamentals.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/domain/value/test_fundamentals.py`:

```python
"""Normalize raw yahooquery payload → ValuationInputs."""
from domain.value.fundamentals import normalize_fundamentals


def test_normalize_full_payload_populates_all_fields():
    raw = {
        'ticker': 'INTC',
        'sector': 'Technology',
        'price': 24.11,
        'forward_eps': 2.87,
        'revenue_ttm': 54_000_000_000,
        'revenue_ttm_1y_ago': 48_000_000_000,
        'eps_ttm': 1.4,
        'eps_ttm_1y_ago': 1.29,
        'book_value_per_share': 21.9,
        'enterprise_value': 100_000_000_000,
        'ebitda_ttm': 15_400_000_000,
        'shares_short': 180_000_000,
        'float_shares': 1_040_000_000,
        'avg_daily_volume_30d': 29_000_000,
        'volume_5d_avg': 45_000_000,
        'volume_20d_avg': 38_000_000,
        'price_1y_ago': 29.55,
    }
    v = normalize_fundamentals(raw)
    assert v.ticker == 'INTC'
    assert v.sector == 'Technology'
    assert v.price == 24.11
    assert v.forward_eps == 2.87


def test_normalize_missing_optional_fields_returns_none():
    raw = {'ticker': 'X', 'sector': 'Technology'}
    v = normalize_fundamentals(raw)
    assert v.ticker == 'X'
    assert v.price is None
    assert v.forward_eps is None
    assert v.eps_ttm is None


def test_normalize_nan_treated_as_none():
    import math
    raw = {'ticker': 'X', 'sector': None, 'price': math.nan}
    v = normalize_fundamentals(raw)
    assert v.price is None
    assert v.sector is None


def test_normalize_missing_ticker_raises():
    import pytest
    with pytest.raises((KeyError, ValueError)):
        normalize_fundamentals({'sector': 'Technology'})


def test_normalize_zero_prices_kept_as_zero_not_none():
    """Zero is a valid value distinct from missing; only NaN/None get nulled."""
    raw = {'ticker': 'X', 'price': 0.0, 'revenue_ttm': 0}
    v = normalize_fundamentals(raw)
    assert v.price == 0.0
    assert v.revenue_ttm == 0
```

- [ ] **Step 2.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_fundamentals.py -v
```

Expected: `ModuleNotFoundError: No module named 'domain.value.fundamentals'`.

- [ ] **Step 2.3: Implement `domain/value/fundamentals.py`**

```python
"""Normalize raw fundamentals payload → ValuationInputs.

The raw payload is a plain dict pulled by yahooquery_fundamentals.py.
This module handles missing fields, NaN, and type coercion so downstream
methodologies see a clean, uniform shape.
"""
from __future__ import annotations
import math
from typing import Any, Optional

from domain.value.types import ValuationInputs


_FIELDS: tuple[str, ...] = (
    'sector', 'price', 'forward_eps', 'revenue_ttm', 'revenue_ttm_1y_ago',
    'eps_ttm', 'eps_ttm_1y_ago', 'book_value_per_share',
    'enterprise_value', 'ebitda_ttm', 'shares_short', 'float_shares',
    'avg_daily_volume_30d', 'volume_5d_avg', 'volume_20d_avg',
    'price_1y_ago',
)


def _clean(v: Any) -> Optional[Any]:
    """Coerce NaN and empty strings to None; pass through everything else."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.strip() == '':
        return None
    return v


def normalize_fundamentals(raw: dict) -> ValuationInputs:
    """Turn a raw yahooquery-style payload into a ValuationInputs.

    Requires 'ticker' key.  All other fields are optional and default
    to None if missing / NaN.
    """
    if 'ticker' not in raw or not raw['ticker']:
        raise ValueError("normalize_fundamentals: 'ticker' is required")
    kwargs: dict = {'ticker': str(raw['ticker'])}
    for f in _FIELDS:
        kwargs[f] = _clean(raw.get(f))
    return ValuationInputs(**kwargs)
```

- [ ] **Step 2.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_fundamentals.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add domain/value/fundamentals.py tests/domain/value/test_fundamentals.py
git commit -m "$(cat <<'EOF'
feat(value): normalize raw payload → ValuationInputs

Handles NaN, empty strings, and missing optional fields.  Ticker is
required; everything else is nullable and left for downstream
methodologies to interpret.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Sector-relative methodology (Methodology A)

**Files:**
- Create: `domain/value/sector_relative.py`
- Create: `tests/domain/value/test_sector_relative.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/domain/value/test_sector_relative.py`:

```python
"""Methodology A — sector-relative multiples."""
from domain.value.types import ValuationInputs
from domain.value.sector_relative import (
    compute_sector_medians,
    compute_sector_relative,
)


def _make(ticker: str, sector: str, pe: float, ps: float, pb: float, ev_ebitda: float):
    """Small helper.  Constructs a ValuationInputs whose derived ratios
    will be exactly pe, ps, pb, ev_ebitda when price=100 is passed downstream.

    Since the module computes ratios from raw fields, we set those fields
    directly to values that produce the target ratios at price=100.
    """
    return ValuationInputs(
        ticker=ticker, sector=sector, price=100.0,
        forward_eps=100.0 / pe,               # so price/eps = pe
        revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None,
        book_value_per_share=100.0 / pb,      # price/book = pb
        enterprise_value=ev_ebitda * 1_000_000_000,
        ebitda_ttm=1_000_000_000,             # so EV/EBITDA = ev_ebitda
        shares_short=None, float_shares=None,
        avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=None,
    )


def test_sector_medians_computed_across_sector():
    """Given 3 tickers in a sector, medians should equal the middle value."""
    inputs = [
        _make('A', 'Technology', pe=10, ps=1, pb=1, ev_ebitda=5),
        _make('B', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10),
        _make('C', 'Technology', pe=30, ps=3, pb=3, ev_ebitda=15),
    ]
    medians = compute_sector_medians(inputs)
    assert 'Technology' in medians
    assert medians['Technology']['pe_fwd'] == 20.0
    assert medians['Technology']['pb'] == 2.0


def test_sector_medians_require_min_10_peers_by_default():
    """Sectors with fewer than 10 members should not appear in medians."""
    inputs = [_make(f'T{i}', 'Utilities', pe=15, ps=2, pb=1, ev_ebitda=8)
              for i in range(5)]
    medians = compute_sector_medians(inputs, min_peers=10)
    assert 'Utilities' not in medians


def test_sector_relative_score_undervalued():
    """A ticker at 50% discount to peer median PE should score high."""
    inputs = [
        _make('A', 'Technology', pe=10, ps=1, pb=1, ev_ebitda=5),  # our ticker
        *[_make(f'B{i}', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10)
          for i in range(10)],
    ]
    medians = compute_sector_medians(inputs)
    result = compute_sector_relative(inputs[0], medians)
    assert result.score >= 8.0    # very undervalued
    assert result.pe_discount_pct is not None and result.pe_discount_pct > 40.0


def test_sector_relative_score_overvalued():
    inputs = [
        _make('A', 'Technology', pe=40, ps=4, pb=4, ev_ebitda=20),  # overvalued
        *[_make(f'B{i}', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10)
          for i in range(10)],
    ]
    medians = compute_sector_medians(inputs)
    result = compute_sector_relative(inputs[0], medians)
    assert result.score <= 3.0


def test_sector_relative_none_when_sector_missing_from_medians():
    """Ticker in a sector with too few peers → None."""
    solo = _make('Q', 'Utilities', pe=15, ps=2, pb=1, ev_ebitda=8)
    medians = compute_sector_medians([solo], min_peers=10)
    assert compute_sector_relative(solo, medians) is None


def test_sector_relative_requires_at_least_two_ratios():
    """A ticker with only one computable ratio should be skipped."""
    v = ValuationInputs(
        ticker='X', sector='Technology', price=100.0,
        forward_eps=5.0,                       # gives PE
        revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None,
        book_value_per_share=None,             # no PB
        enterprise_value=None,                  # no EV/EBITDA
        ebitda_ttm=None,
        shares_short=None, float_shares=None,
        avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=None,
    )
    peers = [_make(f'B{i}', 'Technology', pe=20, ps=2, pb=2, ev_ebitda=10)
             for i in range(10)]
    medians = compute_sector_medians([v] + peers)
    result = compute_sector_relative(v, medians)
    assert result is None   # only 1 ratio; skip
```

- [ ] **Step 3.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_sector_relative.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement `domain/value/sector_relative.py`**

```python
"""Methodology A — sector-relative valuation multiples.

Compare a ticker's core valuation ratios to the median of its GICS
sector peers.  Score in [0, 10], higher = more undervalued.

P/S uses `float_shares` as a proxy for shares outstanding.  This is a
small approximation for names where float ≠ shares outstanding, but
close enough for a screener.
"""
from __future__ import annotations
import statistics
from typing import Optional

from domain.value.types import ValuationInputs, SectorRelativeResult


_RATIO_KEYS = ('pe_fwd', 'ps_ttm', 'pb', 'ev_ebitda')


def _ratios(v: ValuationInputs) -> dict[str, Optional[float]]:
    """Compute the four ratios; None when input is missing or math undefined."""
    pe = ps = pb = ev_ebitda = None

    if v.price is not None and v.forward_eps and v.forward_eps > 0:
        pe = v.price / v.forward_eps

    if v.price is not None and v.book_value_per_share and v.book_value_per_share > 0:
        pb = v.price / v.book_value_per_share

    if v.enterprise_value and v.ebitda_ttm and v.ebitda_ttm > 0:
        ev_ebitda = v.enterprise_value / v.ebitda_ttm

    # P/S = price / (revenue_ttm / shares).  Use float_shares as a proxy for
    # shares outstanding — good enough for a value screener.
    if (v.price is not None and v.revenue_ttm and v.revenue_ttm > 0
            and v.float_shares and v.float_shares > 0):
        revenue_per_share = v.revenue_ttm / v.float_shares
        if revenue_per_share > 0:
            ps = v.price / revenue_per_share

    return {'pe_fwd': pe, 'ps_ttm': ps, 'pb': pb, 'ev_ebitda': ev_ebitda}


def compute_sector_medians(
    inputs: list[ValuationInputs],
    min_peers: int = 10,
) -> dict[str, dict[str, float]]:
    """Group inputs by sector, compute median of each ratio.

    Sectors with fewer than `min_peers` valid rows are dropped.
    """
    by_sector: dict[str, dict[str, list[float]]] = {}
    for v in inputs:
        if not v.sector:
            continue
        r = _ratios(v)
        bucket = by_sector.setdefault(v.sector, {k: [] for k in _RATIO_KEYS})
        for key in _RATIO_KEYS:
            val = r.get(key)
            if val is not None and val > 0:
                bucket[key].append(val)

    medians: dict[str, dict[str, float]] = {}
    for sector, ratios in by_sector.items():
        # A sector qualifies if ANY single ratio has >= min_peers.  In
        # practice we treat the sector as valid iff its dominant ratio
        # meets the floor.
        counts = {k: len(vs) for k, vs in ratios.items()}
        max_n = max(counts.values()) if counts else 0
        if max_n < min_peers:
            continue
        medians[sector] = {
            k: statistics.median(vs) for k, vs in ratios.items() if len(vs) >= min_peers
        }
    return medians


def _discount_pct(ticker_ratio: float, peer_median: float) -> float:
    """Return (peer_median - ticker_ratio) / peer_median as a percentage."""
    return (peer_median - ticker_ratio) / peer_median * 100.0


def _score_from_discount(discount_pct: float) -> float:
    """Map a single-ratio discount percentage to a 0-10 score."""
    if discount_pct >= 30.0:
        return 10.0
    if discount_pct >= 15.0:
        return 7.0 + (discount_pct - 15.0) / 15.0 * 3.0
    if discount_pct >= -15.0:
        return 5.0 + discount_pct / 15.0 * 2.0
    if discount_pct >= -30.0:
        return 2.0 + (discount_pct + 30.0) / 15.0 * 2.0
    return max(0.0, 1.0 + (discount_pct + 45.0) / 15.0)


def compute_sector_relative(
    v: ValuationInputs,
    sector_medians: dict[str, dict[str, float]],
) -> Optional[SectorRelativeResult]:
    """Score a single ticker against its sector's median ratios.

    Returns None if the ticker's sector is not in `sector_medians`
    (too few peers) OR the ticker has fewer than 2 computable ratios.
    """
    if not v.sector or v.sector not in sector_medians:
        return None
    peer_medians = sector_medians[v.sector]
    ticker_ratios = _ratios(v)

    per_ratio_scores: list[float] = []
    result_fields: dict = {}

    for key in _RATIO_KEYS:
        r = ticker_ratios.get(key)
        pm = peer_medians.get(key)
        if r is None or pm is None or pm <= 0:
            continue
        d = _discount_pct(r, pm)
        s = _score_from_discount(d)
        per_ratio_scores.append(s)
        result_fields[key] = r
        result_fields[f'{key}_peer_median'] = pm
        if key != 'ev_ebitda':
            result_fields[f'{key}_discount_pct'] = d

    if len(per_ratio_scores) < 2:
        return None

    return SectorRelativeResult(
        score=sum(per_ratio_scores) / len(per_ratio_scores),
        ratios_used=len(per_ratio_scores),
        **result_fields,
    )
```

- [ ] **Step 3.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_sector_relative.py -v
```

Expected: 6/6 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add domain/value/sector_relative.py tests/domain/value/test_sector_relative.py
git commit -m "$(cat <<'EOF'
feat(value): Methodology A — sector-relative multiples

Compares P/E, P/S, P/B, EV/EBITDA to GICS sector peer medians.  Score
0-10; higher = more undervalued.  Ticker skipped if sector lacks
min-10 peers or has fewer than 2 computable ratios.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Fundamental divergence methodology (Methodology B)

**Files:**
- Create: `domain/value/fundamental_divergence.py`
- Create: `tests/domain/value/test_fundamental_divergence.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/domain/value/test_fundamental_divergence.py`:

```python
"""Methodology B — fundamental momentum divergence."""
import pytest

from domain.value.types import ValuationInputs
from domain.value.fundamental_divergence import compute_fundamental_divergence


def _v(revenue: float, revenue_prior: float, eps: float, eps_prior: float,
       price: float, price_prior: float, ticker: str = 'X'):
    return ValuationInputs(
        ticker=ticker, sector='Technology', price=price,
        forward_eps=None,
        revenue_ttm=revenue, revenue_ttm_1y_ago=revenue_prior,
        eps_ttm=eps, eps_ttm_1y_ago=eps_prior,
        book_value_per_share=None, enterprise_value=None, ebitda_ttm=None,
        shares_short=None, float_shares=None, avg_daily_volume_30d=None,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=price_prior,
    )


def test_undervalued_when_fundamentals_up_price_down():
    v = _v(revenue=112, revenue_prior=100,          # +12% rev
           eps=1.087, eps_prior=1.0,                # +8.7% eps
           price=81.6, price_prior=100.0)           # -18.4% price
    r = compute_fundamental_divergence(v)
    assert r is not None
    assert r.divergence_pp > 25   # very undervalued
    assert r.score >= 8.0


def test_overvalued_when_price_ran_ahead():
    v = _v(revenue=101, revenue_prior=100,
           eps=1.01, eps_prior=1.0,
           price=150, price_prior=100)              # +50% price on +1% fundamentals
    r = compute_fundamental_divergence(v)
    assert r is not None
    assert r.divergence_pp < -25
    assert r.score <= 2.0


def test_aligned_when_growth_and_price_match():
    v = _v(revenue=110, revenue_prior=100,
           eps=1.10, eps_prior=1.0,
           price=110, price_prior=100)
    r = compute_fundamental_divergence(v)
    assert r is not None
    assert -10 <= r.divergence_pp <= 10
    assert 4.0 <= r.score <= 6.0


def test_skip_when_revenue_below_50m():
    v = _v(revenue=10_000_000, revenue_prior=8_000_000,  # too small
           eps=0.10, eps_prior=0.08,
           price=10, price_prior=8)
    assert compute_fundamental_divergence(v) is None


def test_skip_when_eps_crosses_zero():
    v = _v(revenue=100_000_000, revenue_prior=90_000_000,
           eps=0.5, eps_prior=-0.2,                 # sign flip
           price=10, price_prior=8)
    assert compute_fundamental_divergence(v) is None


def test_revenue_only_when_eps_negative_both_periods():
    """If both EPS values are negative, use revenue-only weighting."""
    v = _v(revenue=100_000_000, revenue_prior=100_000_000,
           eps=-0.5, eps_prior=-1.0,                # both negative
           price=90, price_prior=100)
    r = compute_fundamental_divergence(v)
    assert r is not None
    # revenue_growth = 0, price_growth = -10% → divergence = +10 pp
    assert 5 <= r.divergence_pp <= 15


def test_skip_when_missing_price_history():
    v = _v(revenue=100_000_000, revenue_prior=100_000_000,
           eps=1.0, eps_prior=1.0,
           price=100, price_prior=100)
    v_missing = ValuationInputs(**{**v.__dict__, 'price_1y_ago': None})
    assert compute_fundamental_divergence(v_missing) is None
```

- [ ] **Step 4.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_fundamental_divergence.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement `domain/value/fundamental_divergence.py`**

```python
"""Methodology B — fundamental momentum divergence.

Compare 12-month fundamental trends (revenue + EPS) to 12-month price
trends.  Divergence in favour of fundamentals = undervalued.
"""
from __future__ import annotations
from typing import Optional

from domain.value.types import ValuationInputs, FundamentalDivergenceResult


_MIN_REVENUE_USD = 50_000_000


def _score_from_divergence(divergence_pp: float) -> float:
    """Map divergence in percentage points to a 0-10 score."""
    if divergence_pp > 25:
        return 8.0 + min(2.0, (divergence_pp - 25.0) / 25.0 * 2.0)
    if divergence_pp > 10:
        return 6.0 + (divergence_pp - 10.0) / 15.0 * 2.0
    if divergence_pp >= -10:
        return 4.0 + (divergence_pp + 10.0) / 20.0 * 2.0
    if divergence_pp >= -25:
        return 2.0 + (divergence_pp + 25.0) / 15.0 * 2.0
    return max(0.0, 2.0 + (divergence_pp + 25.0) / 25.0 * 2.0)


def compute_fundamental_divergence(
    v: ValuationInputs,
) -> Optional[FundamentalDivergenceResult]:
    """Return a score + component fields, or None if skipped."""
    # Guardrail: revenue must exist and exceed floor
    if not v.revenue_ttm or v.revenue_ttm < _MIN_REVENUE_USD:
        return None
    if not v.revenue_ttm_1y_ago or v.revenue_ttm_1y_ago <= 0:
        return None
    if v.price is None or v.price_1y_ago is None or v.price_1y_ago <= 0:
        return None

    revenue_growth = (v.revenue_ttm - v.revenue_ttm_1y_ago) / v.revenue_ttm_1y_ago * 100.0

    eps_growth: Optional[float] = None
    if v.eps_ttm is not None and v.eps_ttm_1y_ago is not None:
        # Both negative → revenue-only weighting
        if v.eps_ttm < 0 and v.eps_ttm_1y_ago < 0:
            eps_growth = None
        elif v.eps_ttm * v.eps_ttm_1y_ago < 0:
            # Sign flip → growth rate undefined; skip ticker entirely
            return None
        elif v.eps_ttm_1y_ago == 0:
            return None
        else:
            eps_growth = (v.eps_ttm - v.eps_ttm_1y_ago) / abs(v.eps_ttm_1y_ago) * 100.0

    if eps_growth is None:
        fundamental_growth = revenue_growth       # revenue-only
    else:
        fundamental_growth = 0.6 * revenue_growth + 0.4 * eps_growth

    price_growth = (v.price - v.price_1y_ago) / v.price_1y_ago * 100.0
    divergence = fundamental_growth - price_growth

    return FundamentalDivergenceResult(
        score=_score_from_divergence(divergence),
        revenue_growth_yoy_pct=revenue_growth,
        eps_growth_yoy_pct=eps_growth,
        price_growth_yoy_pct=price_growth,
        divergence_pp=divergence,
    )
```

- [ ] **Step 4.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_fundamental_divergence.py -v
```

Expected: 7/7 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add domain/value/fundamental_divergence.py tests/domain/value/test_fundamental_divergence.py
git commit -m "$(cat <<'EOF'
feat(value): Methodology B — fundamental momentum divergence

Compares 12-month revenue/EPS growth to 12-month price growth.
Guardrails: revenue >= $50M, no EPS sign flips, both-negative EPS
falls back to revenue-only weighting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Volume overlay

**Files:**
- Create: `domain/value/volume_overlay.py`
- Create: `tests/domain/value/test_volume_overlay.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/domain/value/test_volume_overlay.py`:

```python
"""Volume overlay — 5d/20d ratio + tag."""
from domain.value.types import ValuationInputs
from domain.value.volume_overlay import compute_volume_overlay


def _v(v5: float, v20: float):
    return ValuationInputs(
        ticker='X', sector='Technology', price=100.0,
        forward_eps=None, revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None, book_value_per_share=None,
        enterprise_value=None, ebitda_ttm=None,
        shares_short=None, float_shares=None, avg_daily_volume_30d=None,
        volume_5d_avg=v5, volume_20d_avg=v20,
        price_1y_ago=None,
    )


def test_rising_when_ratio_at_or_above_1_15():
    r = compute_volume_overlay(_v(v5=115, v20=100))
    assert r.tag == 'RISING'
    assert r.score == 10.0
    assert abs(r.ratio - 1.15) < 1e-9


def test_flat_when_ratio_between_0_85_and_1_15():
    r = compute_volume_overlay(_v(v5=100, v20=100))
    assert r.tag == 'FLAT'
    assert r.score == 5.0


def test_declining_when_ratio_at_or_below_0_85():
    r = compute_volume_overlay(_v(v5=85, v20=100))
    assert r.tag == 'DECLINING'
    assert r.score == 0.0


def test_rising_just_at_boundary():
    r = compute_volume_overlay(_v(v5=1.151, v20=1.0))
    assert r.tag == 'RISING'


def test_flat_returned_when_data_missing():
    """Missing volume data → tag FLAT, score 5 (neutral, no signal)."""
    r = compute_volume_overlay(_v(v5=None, v20=None))
    assert r.tag == 'FLAT'
    assert r.score == 5.0
    assert r.ratio is None
```

- [ ] **Step 5.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_volume_overlay.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement `domain/value/volume_overlay.py`**

```python
"""Volume overlay — 5-day average / 20-day average ratio + tag.

Applies to both Role 1 (tag) and Role 2 (score component).  Missing
volume data → FLAT, score 5 (neutral).
"""
from __future__ import annotations

from domain.value.types import ValuationInputs, VolumeOverlayResult


def compute_volume_overlay(v: ValuationInputs) -> VolumeOverlayResult:
    if v.volume_5d_avg is None or v.volume_20d_avg is None or v.volume_20d_avg <= 0:
        return VolumeOverlayResult(score=5.0, tag='FLAT', ratio=None,
                                   vol_5d_avg=v.volume_5d_avg, vol_20d_avg=v.volume_20d_avg)
    ratio = v.volume_5d_avg / v.volume_20d_avg
    if ratio >= 1.15:
        return VolumeOverlayResult(score=10.0, tag='RISING', ratio=ratio,
                                   vol_5d_avg=v.volume_5d_avg,
                                   vol_20d_avg=v.volume_20d_avg)
    if ratio <= 0.85:
        return VolumeOverlayResult(score=0.0, tag='DECLINING', ratio=ratio,
                                   vol_5d_avg=v.volume_5d_avg,
                                   vol_20d_avg=v.volume_20d_avg)
    return VolumeOverlayResult(score=5.0, tag='FLAT', ratio=ratio,
                               vol_5d_avg=v.volume_5d_avg,
                               vol_20d_avg=v.volume_20d_avg)
```

- [ ] **Step 5.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_volume_overlay.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 5.5: Commit**

```bash
git add domain/value/volume_overlay.py tests/domain/value/test_volume_overlay.py
git commit -m "$(cat <<'EOF'
feat(value): volume overlay — 5d/20d ratio + RISING/FLAT/DECLINING tag

Boundaries: >=1.15 RISING, <=0.85 DECLINING, else FLAT.  Missing
data collapses to FLAT/5 (neutral).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Short interest overlay

**Files:**
- Create: `domain/value/short_overlay.py`
- Create: `tests/domain/value/test_short_overlay.py`

- [ ] **Step 6.1: Write failing tests**

Create `tests/domain/value/test_short_overlay.py`:

```python
"""Short interest overlay — direction-aware scoring."""
from domain.value.types import ValuationInputs
from domain.value.short_overlay import compute_short_overlay


def _v(shares_short, float_shares, avg_vol=1_000_000, prior_pct=None):
    return ValuationInputs(
        ticker='X', sector='Technology', price=100.0,
        forward_eps=None, revenue_ttm=None, revenue_ttm_1y_ago=None,
        eps_ttm=None, eps_ttm_1y_ago=None, book_value_per_share=None,
        enterprise_value=None, ebitda_ttm=None,
        shares_short=shares_short, float_shares=float_shares,
        avg_daily_volume_30d=avg_vol,
        volume_5d_avg=None, volume_20d_avg=None,
        price_1y_ago=None,
    )


def test_long_direction_high_si_scores_ten():
    """base_rank >= 5 (long lean), SI 20% → squeeze validation → 10."""
    v = _v(shares_short=200_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=None)
    assert r.short_interest_pct == 20.0
    assert r.score == 10.0


def test_short_direction_high_si_scores_zero():
    """base_rank < 5 (short lean), SI 20% → crowded short → 0."""
    v = _v(shares_short=200_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=2.0, prior_short_interest_pct=None)
    assert r.score == 0.0


def test_short_direction_low_si_scores_ten():
    """base_rank < 5 (short lean), SI 2% → fresh short → 10."""
    v = _v(shares_short=20_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=2.0, prior_short_interest_pct=None)
    assert r.score == 10.0


def test_missing_data_returns_neutral_five():
    v = _v(shares_short=None, float_shares=None)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=None)
    assert r.score == 5.0
    assert r.short_interest_pct is None


def test_days_to_cover_computed():
    v = _v(shares_short=10_000_000, float_shares=100_000_000, avg_vol=2_000_000)
    r = compute_short_overlay(v, base_rank=5.0, prior_short_interest_pct=None)
    assert r.days_to_cover == 5.0


def test_short_interest_delta_positive():
    v = _v(shares_short=180_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=15.0)
    # current SI 18%, prior 15% → delta +3 pp
    assert r.short_interest_delta_pp == 3.0


def test_short_interest_delta_none_when_no_prior():
    v = _v(shares_short=180_000_000, float_shares=1_000_000_000)
    r = compute_short_overlay(v, base_rank=8.0, prior_short_interest_pct=None)
    assert r.short_interest_delta_pp is None
```

- [ ] **Step 6.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_short_overlay.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 6.3: Implement `domain/value/short_overlay.py`**

```python
"""Short interest overlay — direction-aware scoring.

For LONG candidates (base_rank >= 5):
  - High SI (> 15%)  → squeeze validation, score 10
  - Medium (5-15%)  → normal, score 6
  - Low (< 5%)      → clean value, score 5 (neutral)

For SHORT candidates (base_rank < 5):
  - High SI (> 15%) → crowded short, score 0 (danger)
  - Medium (5-15%)  → some crowd, score 4
  - Low (< 5%)      → fresh short, score 10
"""
from __future__ import annotations
from typing import Optional

from domain.value.types import ValuationInputs, ShortOverlayResult


def compute_short_overlay(
    v: ValuationInputs,
    base_rank: float,
    prior_short_interest_pct: Optional[float],
) -> ShortOverlayResult:
    if (v.shares_short is None or v.float_shares is None
            or v.float_shares <= 0):
        return ShortOverlayResult(score=5.0)

    si_pct = v.shares_short / v.float_shares * 100.0

    days_to_cover: Optional[float] = None
    if v.avg_daily_volume_30d and v.avg_daily_volume_30d > 0:
        days_to_cover = v.shares_short / v.avg_daily_volume_30d

    delta_pp: Optional[float] = None
    if prior_short_interest_pct is not None:
        delta_pp = si_pct - prior_short_interest_pct

    # Direction-aware scoring
    if base_rank >= 5.0:  # long lean
        if si_pct > 15.0:
            score = 10.0
        elif si_pct >= 5.0:
            score = 6.0
        else:
            score = 5.0
    else:                  # short lean
        if si_pct > 15.0:
            score = 0.0
        elif si_pct >= 5.0:
            score = 4.0
        else:
            score = 10.0

    return ShortOverlayResult(
        score=score,
        short_interest_pct=si_pct,
        days_to_cover=days_to_cover,
        short_interest_delta_pp=delta_pp,
    )
```

- [ ] **Step 6.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_short_overlay.py -v
```

Expected: 7/7 PASS.

- [ ] **Step 6.5: Commit**

```bash
git add domain/value/short_overlay.py tests/domain/value/test_short_overlay.py
git commit -m "$(cat <<'EOF'
feat(value): direction-aware short interest overlay

Long candidates reward high SI (squeeze validation).  Short
candidates penalise high SI (crowded short).  Missing SI data →
neutral 5.  Days-to-cover and SI delta included when data present.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Scoring — Role 1 tags + Role 2 composite

**Files:**
- Create: `domain/value/scoring.py`
- Create: `tests/domain/value/test_scoring.py`

- [ ] **Step 7.1: Write failing tests**

Create `tests/domain/value/test_scoring.py`:

```python
"""Role 1 tags + Role 2 composite scoring."""
from domain.value.types import (
    SectorRelativeResult, FundamentalDivergenceResult,
    VolumeOverlayResult, ShortOverlayResult,
)
from domain.value.scoring import assign_role1, assign_role2


# ---- Role 1 -----------------------------------------------------------

def test_role1_high_priority_squeeze_when_undervalued_high_si_high_dtc():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=8.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0,
                                          short_interest_pct=18.0,
                                          days_to_cover=6.0),
    )
    assert r.priority == 'HIGH_PRIORITY_SQUEEZE'
    assert r.combined_rank_score == 8.0


def test_role1_high_priority_when_undervalued_clean():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=7.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=6.0, short_interest_pct=8.0),
    )
    assert r.priority == 'HIGH_PRIORITY'


def test_role1_crowded_short_when_overvalued_high_si():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=2.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=2.0),
        volume=VolumeOverlayResult(score=0.0, tag='DECLINING'),
        short_overlay=ShortOverlayResult(score=0.0, short_interest_pct=20.0),
    )
    assert r.priority == 'CROWDED_SHORT_AVOID'


def test_role1_clean_short_when_overvalued_low_si():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=2.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=2.0),
        volume=VolumeOverlayResult(score=0.0, tag='DECLINING'),
        short_overlay=ShortOverlayResult(score=10.0, short_interest_pct=3.0),
    )
    assert r.priority == 'CLEAN_SHORT'


def test_role1_none_priority_for_middle_ranks():
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=5.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=5.0),
        volume=VolumeOverlayResult(score=5.0, tag='FLAT'),
        short_overlay=ShortOverlayResult(score=5.0),
    )
    assert r.priority == 'NONE'


def test_role1_combined_rank_uses_only_available_scores():
    r = assign_role1(
        sector_relative=None,
        fundamental_divergence=FundamentalDivergenceResult(score=8.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=6.0),
    )
    assert r.combined_rank_score == 8.0


def test_role1_shorts_building_tag():
    """Undervalued + SI delta > +1 pp → SHORTS_BUILDING."""
    r = assign_role1(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=7.5),
        volume=VolumeOverlayResult(score=5.0, tag='FLAT'),
        short_overlay=ShortOverlayResult(score=10.0, short_interest_pct=17.0,
                                          days_to_cover=4.0,
                                          short_interest_delta_pp=1.5),
    )
    # 4.0 DTC does NOT satisfy the >5 required for SQUEEZE, so falls to SHORTS_BUILDING
    assert r.priority == 'SHORTS_BUILDING'


# ---- Role 2 -----------------------------------------------------------

def test_role2_composite_full_weights():
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=8.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0),
    )
    # 0.35*8 + 0.35*8 + 0.15*10 + 0.15*10 = 2.8 + 2.8 + 1.5 + 1.5 = 8.6 → *10 = 86.0
    assert abs(r.composite_score - 86.0) < 1e-9


def test_role2_composite_all_zero():
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=0.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=0.0),
        volume=VolumeOverlayResult(score=0.0, tag='DECLINING'),
        short_overlay=ShortOverlayResult(score=0.0),
    )
    assert r.composite_score == 0.0


def test_role2_weights_redistribute_when_component_missing():
    """If fundamental_divergence is None, remaining 3 components get its
    0.35 weight scaled proportionally into their existing weights."""
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=10.0, ratios_used=3),
        fundamental_divergence=None,
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0),
    )
    # Original weights 0.35 (SR) + 0.15 (Vol) + 0.15 (Short) = 0.65
    # Renormalised: 0.538 / 0.231 / 0.231
    # All scores 10 → composite = 10 * 10 = 100
    assert abs(r.composite_score - 100.0) < 1e-9


def test_role2_component_breakdown_present():
    r = assign_role2(
        sector_relative=SectorRelativeResult(score=8.0, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=7.0),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=6.0),
    )
    assert r.component_breakdown['sector_relative'] == 8.0
    assert r.component_breakdown['fund_divergence'] == 7.0
    assert r.component_breakdown['volume'] == 10.0
    assert r.component_breakdown['short_signal'] == 6.0
```

- [ ] **Step 7.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_scoring.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 7.3: Implement `domain/value/scoring.py`**

```python
"""Role 1 (priority tag) + Role 2 (composite) scoring.

Both are computed in parallel on every run so the archived output
supports A/B comparison over multiple monthly cycles.
"""
from __future__ import annotations
from typing import Optional

from domain.value.types import (
    SectorRelativeResult,
    FundamentalDivergenceResult,
    VolumeOverlayResult,
    ShortOverlayResult,
    Role1Score,
    Role2Score,
)


_ROLE2_WEIGHTS: dict[str, float] = {
    'sector_relative': 0.35,
    'fund_divergence': 0.35,
    'volume': 0.15,
    'short_signal': 0.15,
}


def _combined_rank(
    sr: Optional[SectorRelativeResult],
    fd: Optional[FundamentalDivergenceResult],
) -> float:
    """Mean of the two mispricing scores where available."""
    scores: list[float] = []
    if sr is not None:
        scores.append(sr.score)
    if fd is not None:
        scores.append(fd.score)
    if not scores:
        return 5.0                    # neutral default (shouldn't happen — grader skips such tickers)
    return sum(scores) / len(scores)


def _priority(rank: float, short: ShortOverlayResult, volume: VolumeOverlayResult) -> str:
    si = short.short_interest_pct
    dtc = short.days_to_cover
    delta = short.short_interest_delta_pp

    if rank >= 7.0:
        # Long lean
        if si is not None and si > 15.0 and dtc is not None and dtc > 5.0:
            return 'HIGH_PRIORITY_SQUEEZE'
        if delta is not None and delta > 1.0:
            return 'SHORTS_BUILDING'
        return 'HIGH_PRIORITY'
    if rank <= 3.0:
        # Short lean
        if si is not None and si > 15.0:
            return 'CROWDED_SHORT_AVOID'
        if si is not None and si >= 5.0:
            return 'PATIENT_SHORT'
        return 'CLEAN_SHORT'
    return 'NONE'


def assign_role1(
    sector_relative: Optional[SectorRelativeResult],
    fundamental_divergence: Optional[FundamentalDivergenceResult],
    volume: VolumeOverlayResult,
    short_overlay: ShortOverlayResult,
) -> Role1Score:
    rank = _combined_rank(sector_relative, fundamental_divergence)
    priority = _priority(rank, short_overlay, volume)
    return Role1Score(combined_rank_score=rank, priority=priority)  # type: ignore[arg-type]


def assign_role2(
    sector_relative: Optional[SectorRelativeResult],
    fundamental_divergence: Optional[FundamentalDivergenceResult],
    volume: VolumeOverlayResult,
    short_overlay: ShortOverlayResult,
) -> Role2Score:
    available: dict[str, float] = {}
    if sector_relative is not None:
        available['sector_relative'] = sector_relative.score
    if fundamental_divergence is not None:
        available['fund_divergence'] = fundamental_divergence.score
    available['volume'] = volume.score
    available['short_signal'] = short_overlay.score

    # Renormalise weights across available components
    total_available_weight = sum(_ROLE2_WEIGHTS[k] for k in available)
    weighted_sum = 0.0
    for k, score in available.items():
        weighted_sum += (_ROLE2_WEIGHTS[k] / total_available_weight) * score

    return Role2Score(
        composite_score=weighted_sum * 10.0,
        component_breakdown=available,
    )
```

- [ ] **Step 7.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_scoring.py -v
```

Expected: 11/11 PASS.

- [ ] **Step 7.5: Commit**

```bash
git add domain/value/scoring.py tests/domain/value/test_scoring.py
git commit -m "$(cat <<'EOF'
feat(value): Role 1 (tag) + Role 2 (composite) scoring in parallel

Role 1 assigns priority tags based on combined rank + short overlay.
Role 2 computes a 0-100 weighted composite with proportional weight
redistribution when a component is null.

Both are produced on every run for later A/B comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Report assembly

**Files:**
- Create: `domain/value/report.py`
- Create: `tests/domain/value/test_report.py`

- [ ] **Step 8.1: Write failing tests**

Create `tests/domain/value/test_report.py`:

```python
"""MispricingReport assembly + serialisation."""
from domain.value.types import (
    ValuationInputs, ValuationSnapshot,
    SectorRelativeResult, FundamentalDivergenceResult,
    VolumeOverlayResult, ShortOverlayResult,
    Role1Score, Role2Score,
)
from domain.value.report import build_report, snapshot_to_dict


def _snap(ticker: str, rank: float, composite: float, skip: str = None):
    v = ValuationInputs(
        ticker=ticker, sector='Technology', price=100.0,
        forward_eps=5.0, revenue_ttm=1e9, revenue_ttm_1y_ago=8e8,
        eps_ttm=5.0, eps_ttm_1y_ago=4.0, book_value_per_share=20.0,
        enterprise_value=1e10, ebitda_ttm=1e9,
        shares_short=100_000_000, float_shares=1_000_000_000,
        avg_daily_volume_30d=1e7, volume_5d_avg=1.2e7, volume_20d_avg=1e7,
        price_1y_ago=80.0,
    )
    if skip:
        return ValuationSnapshot(
            ticker=ticker, inputs=v,
            sector_relative=None, fundamental_divergence=None,
            volume=None, short_overlay=None, role1=None, role2=None,
            skip_reason=skip,
        )
    return ValuationSnapshot(
        ticker=ticker, inputs=v,
        sector_relative=SectorRelativeResult(score=rank, ratios_used=3),
        fundamental_divergence=FundamentalDivergenceResult(score=rank),
        volume=VolumeOverlayResult(score=10.0, tag='RISING'),
        short_overlay=ShortOverlayResult(score=10.0,
                                          short_interest_pct=15.0),
        role1=Role1Score(combined_rank_score=rank, priority='HIGH_PRIORITY'),
        role2=Role2Score(composite_score=composite,
                          component_breakdown={'sector_relative': rank,
                                                'fund_divergence': rank,
                                                'volume': 10.0,
                                                'short_signal': 10.0}),
        skip_reason=None,
    )


def test_report_ranks_top_longs_by_role1():
    snaps = [
        _snap('A', rank=9.0, composite=85.0),
        _snap('B', rank=7.5, composite=80.0),
        _snap('C', rank=5.0, composite=60.0),
    ]
    r = build_report(
        snapshots=snaps, universe_name='russell2000',
        universe_size=3, run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=2,
    )
    assert [x['ticker'] for x in r.role1_ranked_longs] == ['A', 'B']


def test_report_ranks_top_shorts_by_role1_ascending():
    snaps = [
        _snap('A', rank=9.0, composite=85.0),
        _snap('B', rank=1.5, composite=15.0),
        _snap('C', rank=2.5, composite=25.0),
    ]
    r = build_report(
        snapshots=snaps, universe_name='russell2000',
        universe_size=3, run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=2,
    )
    assert [x['ticker'] for x in r.role1_ranked_shorts] == ['B', 'C']


def test_report_skipped_counts():
    snaps = [
        _snap('A', rank=9.0, composite=85.0),
        _snap('X', rank=0, composite=0, skip='missing_fundamentals'),
        _snap('Y', rank=0, composite=0, skip='revenue_too_small'),
        _snap('Z', rank=0, composite=0, skip='missing_fundamentals'),
    ]
    r = build_report(
        snapshots=snaps, universe_name='russell2000',
        universe_size=4, run_timestamp_utc='2026-07-02T00:00:00Z',
        top_n=5,
    )
    assert r.graded_size == 1
    assert r.skipped == 3
    assert r.skipped_reasons == {'missing_fundamentals': 2, 'revenue_too_small': 1}


def test_snapshot_to_dict_has_all_keys():
    snap = _snap('A', rank=9.0, composite=85.0)
    d = snapshot_to_dict(snap)
    assert d['ticker'] == 'A'
    assert d['sector_relative_score'] == 9.0
    assert d['composite_score'] == 85.0
    assert 'priority' in d
    assert 'fundamental_trend' in d
    assert 'sector_multiples' in d
```

- [ ] **Step 8.2: Run tests, expect failure**

```bash
python3 -m pytest tests/domain/value/test_report.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 8.3: Implement `domain/value/report.py`**

```python
"""Assemble ValuationSnapshots into a MispricingReport with ranked lists."""
from __future__ import annotations
from collections import Counter
from typing import Iterable

from domain.value.types import ValuationSnapshot, MispricingReport


def snapshot_to_dict(snap: ValuationSnapshot) -> dict:
    """Serialize a ValuationSnapshot to the flat dict used in JSON output."""
    v = snap.inputs
    sr = snap.sector_relative
    fd = snap.fundamental_divergence
    vol = snap.volume
    short = snap.short_overlay
    r1 = snap.role1
    r2 = snap.role2

    d: dict = {
        'ticker': snap.ticker,
        'sector': v.sector,
        'spot': v.price,
    }
    if sr is not None:
        d['sector_relative_score'] = sr.score
        d['sector_multiples'] = {
            'pe_fwd': sr.pe_fwd, 'pe_fwd_peer_median': sr.pe_fwd_peer_median,
            'pe_discount_pct': sr.pe_discount_pct,
            'ps_ttm': sr.ps_ttm, 'ps_ttm_peer_median': sr.ps_ttm_peer_median,
            'ps_discount_pct': sr.ps_discount_pct,
            'pb': sr.pb, 'pb_peer_median': sr.pb_peer_median,
            'pb_discount_pct': sr.pb_discount_pct,
            'ev_ebitda': sr.ev_ebitda, 'ev_ebitda_peer_median': sr.ev_ebitda_peer_median,
        }
    if fd is not None:
        d['fundamental_divergence_score'] = fd.score
        d['fundamental_trend'] = {
            'revenue_growth_yoy_pct': fd.revenue_growth_yoy_pct,
            'eps_growth_yoy_pct': fd.eps_growth_yoy_pct,
            'price_growth_yoy_pct': fd.price_growth_yoy_pct,
            'divergence_pp': fd.divergence_pp,
        }
    if vol is not None:
        d['volume_tag'] = vol.tag
        d['volume_score'] = vol.score
        d['volume'] = {'vol_5d_avg': vol.vol_5d_avg,
                       'vol_20d_avg': vol.vol_20d_avg,
                       'ratio': vol.ratio}
    if short is not None:
        d['short_signal_score'] = short.score
        d['short_interest_pct'] = short.short_interest_pct
        d['days_to_cover'] = short.days_to_cover
        d['short_interest_delta_pp'] = short.short_interest_delta_pp
    if r1 is not None:
        d['combined_rank_score'] = r1.combined_rank_score
        d['priority'] = r1.priority
    if r2 is not None:
        d['composite_score'] = r2.composite_score
        d['component_breakdown'] = r2.component_breakdown
    return d


def build_report(
    snapshots: Iterable[ValuationSnapshot],
    universe_name: str,
    universe_size: int,
    run_timestamp_utc: str,
    top_n: int = 20,
) -> MispricingReport:
    all_snaps = list(snapshots)
    graded = [s for s in all_snaps if s.skip_reason is None and s.role1 is not None]
    skipped = [s for s in all_snaps if s.skip_reason is not None]

    skipped_reasons = Counter(s.skip_reason for s in skipped)

    # Role 1 rankings
    role1_longs = sorted(graded, key=lambda s: -s.role1.combined_rank_score)[:top_n]
    role1_shorts = sorted(graded, key=lambda s: s.role1.combined_rank_score)[:top_n]

    # Role 2 rankings
    role2_longs = sorted(graded, key=lambda s: -s.role2.composite_score)[:top_n]
    role2_shorts = sorted(graded, key=lambda s: s.role2.composite_score)[:top_n]

    return MispricingReport(
        run_timestamp_utc=run_timestamp_utc,
        universe=universe_name,
        universe_size=universe_size,
        graded_size=len(graded),
        skipped=len(skipped),
        skipped_reasons=dict(skipped_reasons),
        role1_ranked_longs=[snapshot_to_dict(s) for s in role1_longs],
        role1_ranked_shorts=[snapshot_to_dict(s) for s in role1_shorts],
        role2_ranked_longs=[snapshot_to_dict(s) for s in role2_longs],
        role2_ranked_shorts=[snapshot_to_dict(s) for s in role2_shorts],
    )
```

- [ ] **Step 8.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/domain/value/test_report.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 8.5: Run full value module test surface**

```bash
python3 -m pytest tests/domain/value/ -q
```

Expected: all pass (Tasks 1-8 accumulate).

- [ ] **Step 8.6: Commit**

```bash
git add domain/value/report.py tests/domain/value/test_report.py
git commit -m "$(cat <<'EOF'
feat(value): assemble ValuationSnapshots → MispricingReport

Top-N longs/shorts under both Role 1 and Role 2 rankings.  Skipped
tickers tracked with per-reason counts.  snapshot_to_dict is the
single JSON serialization entry point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: yahooquery fundamentals fetcher

**Files:**
- Create: `data/sources/yahooquery_fundamentals.py`
- Create: `tests/data/sources/test_yahooquery_fundamentals.py`

- [ ] **Step 9.1: Write failing tests (unit-only, no network)**

Create the test dirs if needed:

```bash
mkdir -p tests/data/sources
touch tests/data/sources/__init__.py
```

Create `tests/data/sources/test_yahooquery_fundamentals.py`:

```python
"""yahooquery_fundamentals — mocked network tests only."""
from unittest.mock import MagicMock, patch

from data.sources.yahooquery_fundamentals import (
    extract_ticker_row,
    fetch_fundamentals_batch,
)


def test_extract_ticker_row_maps_all_fields():
    """Given a mocked yahooquery Ticker payload for one ticker, extract
    the fields we need into a plain dict."""
    yq = MagicMock()
    yq.asset_profile = {'INTC': {'sector': 'Technology'}}
    yq.key_stats = {'INTC': {
        'forwardEps': 2.87, 'bookValue': 21.9, 'enterpriseValue': 100_000_000_000,
        'sharesShort': 180_000_000, 'floatShares': 1_040_000_000,
    }}
    yq.income_statement = {'INTC': [
        {'periodType': 'TTM', 'TotalRevenue': 54_000_000_000, 'DilutedEPS': 1.4, 'EBITDA': 15_400_000_000},
        {'periodType': '12M', 'TotalRevenue': 48_000_000_000, 'DilutedEPS': 1.29, 'EBITDA': 14_000_000_000},
    ]}
    yq.summary_detail = {'INTC': {
        'regularMarketPrice': 24.11, 'averageVolume': 29_000_000,
    }}
    yq.history = MagicMock(return_value=MagicMock())      # unused in unit test

    row = extract_ticker_row(yq, 'INTC', volume_5d_avg=45e6, volume_20d_avg=38e6,
                             price_1y_ago=29.55)
    assert row['ticker'] == 'INTC'
    assert row['sector'] == 'Technology'
    assert row['forward_eps'] == 2.87
    assert row['revenue_ttm'] == 54_000_000_000
    assert row['revenue_ttm_1y_ago'] == 48_000_000_000
    assert row['shares_short'] == 180_000_000
    assert row['price'] == 24.11
    assert row['volume_5d_avg'] == 45e6
    assert row['price_1y_ago'] == 29.55


def test_extract_ticker_row_returns_none_fields_on_missing():
    """Missing sub-dicts → all fields present but None."""
    yq = MagicMock()
    yq.asset_profile = {'X': None}
    yq.key_stats = {'X': None}
    yq.income_statement = {'X': None}
    yq.summary_detail = {'X': None}

    row = extract_ticker_row(yq, 'X', volume_5d_avg=None, volume_20d_avg=None,
                              price_1y_ago=None)
    assert row['ticker'] == 'X'
    assert row['sector'] is None
    assert row['forward_eps'] is None
    assert row['revenue_ttm'] is None


def test_fetch_fundamentals_batch_calls_yahooquery():
    """Wire-level test that fetch_fundamentals_batch splits into batches
    and calls yahooquery once per batch."""
    with patch('data.sources.yahooquery_fundamentals.Ticker') as MockTicker:
        yq_instance = MagicMock()
        MockTicker.return_value = yq_instance
        yq_instance.asset_profile = {}
        yq_instance.key_stats = {}
        yq_instance.income_statement = {}
        yq_instance.summary_detail = {}
        yq_instance.history = MagicMock(return_value=MagicMock(empty=True))

        result = fetch_fundamentals_batch(['A', 'B', 'C'], batch_size=2, sleep_seconds=0.0)

    # 2 batches expected: ['A','B'] and ['C']
    assert MockTicker.call_count == 2
    assert isinstance(result, dict)
```

- [ ] **Step 9.2: Run tests, expect failure**

```bash
python3 -m pytest tests/data/sources/test_yahooquery_fundamentals.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 9.3: Implement `data/sources/yahooquery_fundamentals.py`**

```python
"""Batch fundamentals fetcher over yahooquery.

Pulls the fields our value scanner needs into plain dicts ready for
domain.value.fundamentals.normalize_fundamentals.
"""
from __future__ import annotations
import time
import logging
from typing import Optional

from yahooquery import Ticker


log = logging.getLogger(__name__)


def _safe_get(container, key):
    """dict.get that survives None / non-dict containers."""
    if container is None or not isinstance(container, dict):
        return None
    return container.get(key)


def _first_ttm_row(income_statement_rows) -> Optional[dict]:
    """Return the TTM row if present, else the most recent 12M row."""
    if not income_statement_rows or not isinstance(income_statement_rows, list):
        return None
    for row in income_statement_rows:
        if row.get('periodType') == 'TTM':
            return row
    for row in income_statement_rows:
        if row.get('periodType') == '12M':
            return row
    return income_statement_rows[0]


def _second_12m_row(income_statement_rows) -> Optional[dict]:
    """Return the 12M row from 1 year ago (index 1 after TTM)."""
    if not income_statement_rows or not isinstance(income_statement_rows, list):
        return None
    twelve_months = [r for r in income_statement_rows if r.get('periodType') == '12M']
    if len(twelve_months) >= 2:
        return twelve_months[1]
    if len(twelve_months) == 1:
        return twelve_months[0]
    return None


def extract_ticker_row(
    yq: Ticker,
    ticker: str,
    volume_5d_avg: Optional[float],
    volume_20d_avg: Optional[float],
    price_1y_ago: Optional[float],
) -> dict:
    """Pull all fields for one ticker into a normalisation-ready dict."""
    ap = _safe_get(yq.asset_profile, ticker) or {}
    ks = _safe_get(yq.key_stats, ticker) or {}
    inc = _safe_get(yq.income_statement, ticker)
    sd = _safe_get(yq.summary_detail, ticker) or {}

    ttm = _first_ttm_row(inc) or {}
    prior = _second_12m_row(inc) or {}

    return {
        'ticker': ticker,
        'sector': ap.get('sector') if isinstance(ap, dict) else None,
        'price': sd.get('regularMarketPrice') if isinstance(sd, dict) else None,
        'forward_eps': ks.get('forwardEps') if isinstance(ks, dict) else None,
        'revenue_ttm': ttm.get('TotalRevenue'),
        'revenue_ttm_1y_ago': prior.get('TotalRevenue'),
        'eps_ttm': ttm.get('DilutedEPS'),
        'eps_ttm_1y_ago': prior.get('DilutedEPS'),
        'book_value_per_share': ks.get('bookValue') if isinstance(ks, dict) else None,
        'enterprise_value': ks.get('enterpriseValue') if isinstance(ks, dict) else None,
        'ebitda_ttm': ttm.get('EBITDA'),
        'shares_short': ks.get('sharesShort') if isinstance(ks, dict) else None,
        'float_shares': ks.get('floatShares') if isinstance(ks, dict) else None,
        'avg_daily_volume_30d': sd.get('averageVolume') if isinstance(sd, dict) else None,
        'volume_5d_avg': volume_5d_avg,
        'volume_20d_avg': volume_20d_avg,
        'price_1y_ago': price_1y_ago,
    }


def _compute_volume_and_price_history(yq: Ticker, tickers: list[str]) -> dict[str, tuple]:
    """Return {ticker: (volume_5d_avg, volume_20d_avg, price_1y_ago)}."""
    result: dict[str, tuple] = {}
    try:
        hist = yq.history(period='1y', interval='1d')
    except Exception as exc:                        # noqa: BLE001 - vendor lib
        log.warning("history fetch failed for batch: %s", exc)
        return {t: (None, None, None) for t in tickers}

    for t in tickers:
        try:
            df = hist.xs(t) if hasattr(hist, 'xs') else hist
            closes = df['close'].dropna()
            volumes = df['volume'].dropna()
            v5 = float(volumes.tail(5).mean()) if len(volumes) >= 5 else None
            v20 = float(volumes.tail(20).mean()) if len(volumes) >= 20 else None
            p1y = float(closes.iloc[0]) if len(closes) > 0 else None
            result[t] = (v5, v20, p1y)
        except Exception:
            result[t] = (None, None, None)
    return result


def fetch_fundamentals_batch(
    tickers: list[str],
    batch_size: int = 20,
    sleep_seconds: float = 1.0,
) -> dict[str, dict]:
    """Fetch fundamentals for `tickers`, one batch at a time.

    Returns {ticker: raw_dict}.  Missing tickers get a ticker-only stub
    dict so downstream normalisation can still record a skip reason.
    """
    all_rows: dict[str, dict] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            yq = Ticker(batch, asynchronous=True, progress=False)
            hist_map = _compute_volume_and_price_history(yq, batch)
            for t in batch:
                v5, v20, p1y = hist_map.get(t, (None, None, None))
                all_rows[t] = extract_ticker_row(yq, t, v5, v20, p1y)
        except Exception as exc:                # noqa: BLE001
            log.warning("batch %d-%d failed: %s", i, i + batch_size, exc)
            for t in batch:
                all_rows.setdefault(t, {'ticker': t})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return all_rows
```

- [ ] **Step 9.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/data/sources/test_yahooquery_fundamentals.py -v
```

Expected: 3/3 PASS.

- [ ] **Step 9.5: Commit**

```bash
git add data/sources/yahooquery_fundamentals.py tests/data/sources/__init__.py tests/data/sources/test_yahooquery_fundamentals.py
git commit -m "$(cat <<'EOF'
feat(value): yahooquery batch fundamentals fetcher

Pulls sector, ratios, income-statement TTM + 1y-ago, short interest,
and derives 5d/20d volume + price 1y ago from history.  Batches with
sleep to respect yahooquery rate limits.  Missing data becomes None
in the output; downstream normalisation handles it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Russell 2000 constituent refresh script

**Files:**
- Create: `scripts/refresh_russell2000.py`
- Create: `data/universe/russell2000_constituents.json` (via running the script)
- Create: `tests/scripts/test_refresh_russell2000.py`

- [ ] **Step 10.1: Write failing tests**

Create `tests/scripts/test_refresh_russell2000.py`:

```python
"""Test refresh_russell2000 — behavior only, mocked network."""
from unittest.mock import patch, MagicMock
import io
import json
import pandas as pd

from scripts.refresh_russell2000 import fetch_constituents


def test_fetch_constituents_normalises_dot_to_dash():
    csv = 'Ticker,Name\nAAPL,Apple\nBRK.B,Berkshire\nGOOG,Google\n'
    fake = pd.read_csv(io.StringIO(csv))
    with patch('scripts.refresh_russell2000.pd.read_csv', return_value=fake):
        result = fetch_constituents(url='https://fake')
    assert 'AAPL' in result
    assert 'BRK-B' in result
    assert 'BRK.B' not in result


def test_fetch_constituents_deduplicates():
    csv = 'Ticker,Name\nAAPL,Apple\nAAPL,Apple\nMSFT,Microsoft\n'
    fake = pd.read_csv(io.StringIO(csv))
    with patch('scripts.refresh_russell2000.pd.read_csv', return_value=fake):
        result = fetch_constituents(url='https://fake')
    assert result.count('AAPL') == 1
    assert len(result) == 2


def test_fetch_constituents_raises_when_no_ticker_column():
    csv = 'X,Y\n1,2\n'
    fake = pd.read_csv(io.StringIO(csv))
    with patch('scripts.refresh_russell2000.pd.read_csv', return_value=fake):
        try:
            fetch_constituents(url='https://fake')
        except RuntimeError:
            return
        raise AssertionError('expected RuntimeError')
```

- [ ] **Step 10.2: Run tests, expect failure**

```bash
python3 -m pytest tests/scripts/test_refresh_russell2000.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 10.3: Implement `scripts/refresh_russell2000.py`**

```python
"""Refresh the committed Russell 2000 constituents file.

Pulls from iShares IWM holdings CSV.  Writes a plain JSON list to
data/universe/russell2000_constituents.json.

Usage:
    python -m scripts.refresh_russell2000
    python -m scripts.refresh_russell2000 --dry-run
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SOURCE_URL = (
    'https://www.ishares.com/us/products/239710/'
    'ishares-russell-2000-etf/1467271812596.ajax'
    '?fileType=csv&fileName=IWM_holdings&dataType=fund'
)
OUTPUT_PATH = Path('data/universe/russell2000_constituents.json')

_TICKER_COLUMN_CANDIDATES = ('Ticker', 'Symbol', 'ticker', 'symbol')


def fetch_constituents(url: str = SOURCE_URL) -> list[str]:
    """Pull tickers from the iShares CSV, dedup, normalise dots to dashes."""
    # iShares CSV has a header block; skip it if pandas reads it in
    df = pd.read_csv(url)
    col = None
    for c in _TICKER_COLUMN_CANDIDATES:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise RuntimeError(f"No ticker column found; got {list(df.columns)}")

    syms = df[col].astype(str).str.strip()
    # Normalise dots to dashes (yfinance/Yahoo convention)
    syms = [s.replace('.', '-') for s in syms if s and s != 'nan']
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Refresh Russell 2000 list')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--url', default=SOURCE_URL)
    parser.add_argument('--out', default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)

    try:
        tickers = fetch_constituents(args.url)
    except Exception as exc:                        # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not tickers:
        print("ERROR: 0 tickers returned", file=sys.stderr)
        return 1

    print(f"Fetched {len(tickers)} tickers")
    print(f"  first: {tickers[:5]}")
    print(f"  last:  {tickers[-5:]}")

    if args.dry_run:
        print("Dry run — not writing.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tickers, indent=2) + '\n')
    print(f"Wrote {out} ({len(tickers)} tickers).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 10.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/scripts/test_refresh_russell2000.py -v
```

Expected: 3/3 PASS.

- [ ] **Step 10.5: Seed the committed file**

If iShares URL is available:

```bash
python3 -m scripts.refresh_russell2000
```

If it fails (network issue, iShares URL changed), fall back to a hardcoded stub for now:

```bash
python3 -c "
import json
from pathlib import Path
stub = ['AAOI', 'ABCM', 'ACAD', 'ACIW', 'ACLS']   # tiny stub — refresh later
Path('data/universe').mkdir(parents=True, exist_ok=True)
Path('data/universe/russell2000_constituents.json').write_text(json.dumps(stub, indent=2)+chr(10))
print('Stub written for now — refresh with python -m scripts.refresh_russell2000')
"
```

- [ ] **Step 10.6: Commit**

```bash
git add scripts/refresh_russell2000.py data/universe/russell2000_constituents.json tests/scripts/test_refresh_russell2000.py
git commit -m "$(cat <<'EOF'
feat(value): Russell 2000 constituent refresh script

Pulls the iShares IWM holdings CSV, normalises dots to dashes,
deduplicates.  Committed snapshot at data/universe/
russell2000_constituents.json refreshed via
python -m scripts.refresh_russell2000 (monthly cadence).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: value_watchlist CLI

**Files:**
- Create: `scripts/value_watchlist.py`
- Create: `tests/scripts/test_value_watchlist.py`

- [ ] **Step 11.1: Write failing tests using a small fixture**

Create the fixture file `tests/scripts/fixtures/value_mini_universe.json`:

```bash
mkdir -p tests/scripts/fixtures
cat > tests/scripts/fixtures/value_mini_universe.json <<'EOF'
[
  {"ticker":"A","sector":"Technology","price":10.0,"forward_eps":2.5,"revenue_ttm":1e9,"revenue_ttm_1y_ago":0.8e9,"eps_ttm":2.0,"eps_ttm_1y_ago":1.5,"book_value_per_share":5.0,"enterprise_value":2e9,"ebitda_ttm":2e8,"shares_short":50e6,"float_shares":500e6,"avg_daily_volume_30d":1e6,"volume_5d_avg":1.2e6,"volume_20d_avg":1e6,"price_1y_ago":8.0}
]
EOF
```

Create `tests/scripts/test_value_watchlist.py`:

```python
"""Integration test — end-to-end value_watchlist pipeline."""
import json
from pathlib import Path
from unittest.mock import patch

from scripts.value_watchlist import run_watchlist


def test_run_watchlist_with_mocked_fundamentals(tmp_path):
    fixture = json.loads(Path(
        'tests/scripts/fixtures/value_mini_universe.json'
    ).read_text())
    fake_rows = {r['ticker']: r for r in fixture}

    # Build a 10-name Technology sector so sector_relative can score
    for i in range(10):
        tk = f'PEER{i}'
        fake_rows[tk] = {
            'ticker': tk, 'sector': 'Technology', 'price': 20.0,
            'forward_eps': 2.0, 'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 1e9,
            'eps_ttm': 2.0, 'eps_ttm_1y_ago': 2.0,
            'book_value_per_share': 10.0, 'enterprise_value': 2e10,
            'ebitda_ttm': 2e9, 'shares_short': 5e7, 'float_shares': 1e9,
            'avg_daily_volume_30d': 1e6, 'volume_5d_avg': 1e6, 'volume_20d_avg': 1e6,
            'price_1y_ago': 20.0,
        }

    with patch('scripts.value_watchlist.fetch_fundamentals_batch',
                return_value=fake_rows):
        report = run_watchlist(
            constituents=list(fake_rows.keys()),
            out_dir=tmp_path, cache_dir=tmp_path / 'cache',
            universe_name='mini_universe', top_n=5, use_cache=False,
        )

    assert report.universe == 'mini_universe'
    assert report.graded_size >= 1
    assert (tmp_path / 'latest.json').exists()

    out_data = json.loads((tmp_path / 'latest.json').read_text())
    assert 'role1_ranked_longs' in out_data
    assert 'role2_ranked_longs' in out_data


def test_cache_hit_skips_fetch(tmp_path):
    """When a monthly cache Parquet exists, do NOT call fetch_fundamentals_batch."""
    import pandas as pd
    from datetime import datetime
    cache = tmp_path / 'cache'
    cache.mkdir()
    ym = datetime.utcnow().strftime('%Y-%m')
    fixture = json.loads(Path(
        'tests/scripts/fixtures/value_mini_universe.json'
    ).read_text())
    df = pd.DataFrame(fixture)
    df.to_parquet(cache / f'fundamentals_{ym}.parquet')

    with patch('scripts.value_watchlist.fetch_fundamentals_batch') as spy:
        run_watchlist(
            constituents=['A'], out_dir=tmp_path, cache_dir=cache,
            universe_name='mini', top_n=5, use_cache=True,
        )
        assert spy.call_count == 0        # cache hit
```

- [ ] **Step 11.2: Run tests, expect failure**

```bash
python3 -m pytest tests/scripts/test_value_watchlist.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 11.3: Implement `scripts/value_watchlist.py`**

```python
"""Value Watchlist — monthly full-universe scan.

Usage:
    python -m scripts.value_watchlist                    # russell2000 default
    python -m scripts.value_watchlist --universe sp500
    python -m scripts.value_watchlist --top 30
    python -m scripts.value_watchlist --no-cache
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from data.sources.yahooquery_fundamentals import fetch_fundamentals_batch
from domain.value.fundamentals import normalize_fundamentals
from domain.value.sector_relative import compute_sector_medians, compute_sector_relative
from domain.value.fundamental_divergence import compute_fundamental_divergence
from domain.value.volume_overlay import compute_volume_overlay
from domain.value.short_overlay import compute_short_overlay
from domain.value.scoring import assign_role1, assign_role2
from domain.value.report import build_report
from domain.value.types import ValuationSnapshot


_UNIVERSE_FILES: dict[str, Path] = {
    'russell2000': Path('data/universe/russell2000_constituents.json'),
    'sp500': Path('data/universe/sp500_constituents.json'),
}


def _load_universe(name: str) -> list[str]:
    path = _UNIVERSE_FILES.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Universe {name} not found at {path}")
    return json.loads(path.read_text())


def _cache_path(cache_dir: Path, universe: str) -> Path:
    ym = datetime.utcnow().strftime('%Y-%m')
    return cache_dir / f'fundamentals_{ym}.parquet'


def _load_or_fetch_fundamentals(
    constituents: list[str],
    cache_dir: Path,
    universe: str,
    use_cache: bool,
) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, universe)
    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return {row['ticker']: row.to_dict() for _, row in df.iterrows()}

    rows = fetch_fundamentals_batch(constituents)
    if rows:
        df = pd.DataFrame(list(rows.values()))
        df.to_parquet(cache_file, index=False)
    return rows


def _score_ticker(
    inputs, sector_medians, prior_si_pct: Optional[float]
) -> ValuationSnapshot:
    """Compute all sub-results + Role1/Role2 for a single ValuationInputs."""
    sr = compute_sector_relative(inputs, sector_medians)
    fd = compute_fundamental_divergence(inputs)

    if sr is None and fd is None:
        return ValuationSnapshot(
            ticker=inputs.ticker, inputs=inputs, sector_relative=None,
            fundamental_divergence=None, volume=None, short_overlay=None,
            role1=None, role2=None, skip_reason='insufficient_data',
        )

    vol = compute_volume_overlay(inputs)
    # Base rank for direction-aware short overlay
    scores = [x.score for x in (sr, fd) if x is not None]
    base_rank = sum(scores) / len(scores)
    short = compute_short_overlay(inputs, base_rank=base_rank,
                                    prior_short_interest_pct=prior_si_pct)

    role1 = assign_role1(sr, fd, vol, short)
    role2 = assign_role2(sr, fd, vol, short)
    return ValuationSnapshot(
        ticker=inputs.ticker, inputs=inputs,
        sector_relative=sr, fundamental_divergence=fd,
        volume=vol, short_overlay=short, role1=role1, role2=role2,
        skip_reason=None,
    )


def run_watchlist(
    constituents: list[str],
    out_dir: Path,
    cache_dir: Path,
    universe_name: str,
    top_n: int = 20,
    use_cache: bool = True,
    prior_report_path: Optional[Path] = None,
):
    """Execute the full pipeline.  Returns MispricingReport."""
    raw_rows = _load_or_fetch_fundamentals(constituents, cache_dir, universe_name, use_cache)

    # Load prior monthly SI if available
    prior_si: dict[str, float] = {}
    if prior_report_path and prior_report_path.exists():
        prior = json.loads(prior_report_path.read_text())
        for lst_key in ('role1_ranked_longs', 'role1_ranked_shorts',
                         'role2_ranked_longs', 'role2_ranked_shorts'):
            for row in prior.get(lst_key, []):
                if row.get('short_interest_pct') is not None:
                    prior_si[row['ticker']] = row['short_interest_pct']

    # Normalize + build inputs list
    inputs_list = []
    normalize_failures: list[str] = []
    for t in constituents:
        raw = raw_rows.get(t)
        if not raw:
            normalize_failures.append(t)
            continue
        try:
            inputs_list.append(normalize_fundamentals(raw))
        except Exception:
            normalize_failures.append(t)

    # Sector medians (across all tickers with sector + ratios)
    sector_medians = compute_sector_medians(inputs_list)

    # Score every ticker
    snapshots: list[ValuationSnapshot] = []
    for v in inputs_list:
        snap = _score_ticker(v, sector_medians,
                              prior_si_pct=prior_si.get(v.ticker))
        snapshots.append(snap)

    # Add explicit skip-rows for tickers that never made it into inputs
    for t in normalize_failures:
        snapshots.append(ValuationSnapshot(
            ticker=t,
            inputs=normalize_fundamentals({'ticker': t}),
            sector_relative=None, fundamental_divergence=None,
            volume=None, short_overlay=None, role1=None, role2=None,
            skip_reason='missing_fundamentals',
        ))

    ts_utc = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    report = build_report(
        snapshots=snapshots, universe_name=universe_name,
        universe_size=len(constituents), run_timestamp_utc=ts_utc,
        top_n=top_n,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_timestamp_utc': report.run_timestamp_utc,
        'universe': report.universe,
        'universe_size': report.universe_size,
        'graded_size': report.graded_size,
        'skipped': report.skipped,
        'skipped_reasons': report.skipped_reasons,
        'role1_ranked_longs': report.role1_ranked_longs,
        'role1_ranked_shorts': report.role1_ranked_shorts,
        'role2_ranked_longs': report.role2_ranked_longs,
        'role2_ranked_shorts': report.role2_ranked_shorts,
    }
    (out_dir / 'latest.json').write_text(json.dumps(payload, indent=2, default=str))
    ym = datetime.utcnow().strftime('%Y-%m')
    (out_dir / f'value_report_{ym}.json').write_text(json.dumps(payload, default=str))

    _print_summary(report)
    return report


def _print_summary(report):
    print(f"\n=== Value Watchlist — {report.universe} ===")
    print(f"Universe: {report.universe_size}  Graded: {report.graded_size}  Skipped: {report.skipped}")
    print(f"\nTOP 5 LONGS (Role 1 rank):")
    for i, row in enumerate(report.role1_ranked_longs[:5], 1):
        print(f"  {i} {row['ticker']:<7}  score {row.get('combined_rank_score',0):.1f}  "
              f"vol {row.get('volume_tag','?'):<10}  SI {row.get('short_interest_pct',0):.1f}%  "
              f"{row.get('priority','')}")
    print(f"\nTOP 5 LONGS (Role 2 composite):")
    for i, row in enumerate(report.role2_ranked_longs[:5], 1):
        print(f"  {i} {row['ticker']:<7}  composite {row.get('composite_score',0):.1f}")
    print(f"\nTOP 5 SHORTS (Role 1 rank):")
    for i, row in enumerate(report.role1_ranked_shorts[:5], 1):
        print(f"  {i} {row['ticker']:<7}  score {row.get('combined_rank_score',0):.1f}  "
              f"SI {row.get('short_interest_pct',0):.1f}%  {row.get('priority','')}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description='Value Watchlist — monthly stock mispricing scan')
    p.add_argument('--universe', choices=list(_UNIVERSE_FILES.keys()), default='russell2000')
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--out', type=Path, default=Path('output/value'))
    p.add_argument('--cache-dir', type=Path, default=Path('cache/value'))
    args = p.parse_args(argv)

    try:
        constituents = _load_universe(args.universe)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Look for prior month's archive so we can compute SI delta
    from datetime import date
    def _prior_month_yyyy_mm(today: date) -> str:
        y, m = today.year, today.month - 1
        if m == 0:
            y -= 1
            m = 12
        return f'{y:04d}-{m:02d}'
    prior_ym = _prior_month_yyyy_mm(date.today())
    prior_path = args.out / f'value_report_{prior_ym}.json'

    run_watchlist(
        constituents=constituents,
        out_dir=args.out, cache_dir=args.cache_dir,
        universe_name=args.universe, top_n=args.top,
        use_cache=not args.no_cache,
        prior_report_path=prior_path if prior_path.exists() else None,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 11.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/scripts/test_value_watchlist.py -v
```

Expected: 2/2 PASS.

- [ ] **Step 11.5: Full value + script suite**

```bash
python3 -m pytest tests/domain/value/ tests/data/sources/test_yahooquery_fundamentals.py tests/scripts/test_refresh_russell2000.py tests/scripts/test_value_watchlist.py -q
```

Expected: all pass.

- [ ] **Step 11.6: Commit**

```bash
git add scripts/value_watchlist.py tests/scripts/test_value_watchlist.py tests/scripts/fixtures/value_mini_universe.json
git commit -m "$(cat <<'EOF'
feat(value): value_watchlist CLI — monthly mispricing scan

Loads Russell 2000 (default) or S&P 500 constituents, fetches
fundamentals (Parquet cache per month), computes sector-relative +
fundamental-divergence + volume + short overlay results, produces
Role 1 tags and Role 2 composite in parallel, writes latest.json +
archived monthly snapshot.

Prior monthly archive is optionally loaded to compute SI delta.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: value_check CLI (on-demand single-ticker lookup)

**Files:**
- Create: `scripts/value_check.py`
- Create: `tests/scripts/test_value_check.py`

- [ ] **Step 12.1: Write failing tests**

Create `tests/scripts/test_value_check.py`:

```python
"""value_check — single-ticker deep dive against monthly cache."""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.value_check import check_ticker


def _write_cache(tmp_path: Path, ym: str, rows: list[dict]):
    cache = tmp_path / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(cache / f'fundamentals_{ym}.parquet')
    return cache


def test_check_ticker_prints_full_breakdown(tmp_path, capsys):
    """A cached ticker returns a dict with all breakdown fields."""
    from datetime import datetime
    ym = datetime.utcnow().strftime('%Y-%m')
    # Build a peer set for sector medians
    rows = [{
        'ticker': f'T{i}', 'sector': 'Technology', 'price': 20.0,
        'forward_eps': 2.0, 'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 1e9,
        'eps_ttm': 2.0, 'eps_ttm_1y_ago': 2.0,
        'book_value_per_share': 10.0, 'enterprise_value': 2e10,
        'ebitda_ttm': 2e9,
        'shares_short': 5e7, 'float_shares': 1e9, 'avg_daily_volume_30d': 1e6,
        'volume_5d_avg': 1e6, 'volume_20d_avg': 1e6, 'price_1y_ago': 20.0,
    } for i in range(10)]
    rows.append({
        'ticker': 'X', 'sector': 'Technology', 'price': 10.0,
        'forward_eps': 2.5, 'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 0.8e9,
        'eps_ttm': 2.0, 'eps_ttm_1y_ago': 1.5,
        'book_value_per_share': 5.0, 'enterprise_value': 2e9,
        'ebitda_ttm': 2e8,
        'shares_short': 5e7, 'float_shares': 5e8, 'avg_daily_volume_30d': 1e6,
        'volume_5d_avg': 1.2e6, 'volume_20d_avg': 1e6, 'price_1y_ago': 8.0,
    })
    cache = _write_cache(tmp_path, ym, rows)

    out = check_ticker('X', cache_dir=cache)
    assert out is not None
    assert out['ticker'] == 'X'
    assert 'sector_relative_score' in out
    assert 'fundamental_divergence_score' in out
    assert 'composite_score' in out


def test_check_ticker_missing_returns_none(tmp_path):
    from datetime import datetime
    ym = datetime.utcnow().strftime('%Y-%m')
    _write_cache(tmp_path, ym, [{'ticker': 'A', 'sector': 'Technology',
                                  'price': 20.0, 'forward_eps': 2.0,
                                  'revenue_ttm': 1e9, 'revenue_ttm_1y_ago': 1e9,
                                  'eps_ttm': 2.0, 'eps_ttm_1y_ago': 2.0,
                                  'book_value_per_share': 10.0,
                                  'enterprise_value': 2e10,
                                  'ebitda_ttm': 2e9,
                                  'shares_short': 5e7, 'float_shares': 1e9,
                                  'avg_daily_volume_30d': 1e6,
                                  'volume_5d_avg': 1e6, 'volume_20d_avg': 1e6,
                                  'price_1y_ago': 20.0}])
    cache = tmp_path / 'cache'
    out = check_ticker('DOES_NOT_EXIST', cache_dir=cache)
    assert out is None
```

- [ ] **Step 12.2: Run tests, expect failure**

```bash
python3 -m pytest tests/scripts/test_value_check.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 12.3: Implement `scripts/value_check.py`**

```python
"""value_check — on-demand single-ticker deep dive.

Loads the current month's fundamentals cache and prints a full
breakdown for one ticker.

Usage:
    python -m scripts.value_check INTC
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from domain.value.fundamentals import normalize_fundamentals
from domain.value.sector_relative import compute_sector_medians, compute_sector_relative
from domain.value.fundamental_divergence import compute_fundamental_divergence
from domain.value.volume_overlay import compute_volume_overlay
from domain.value.short_overlay import compute_short_overlay
from domain.value.scoring import assign_role1, assign_role2
from domain.value.report import snapshot_to_dict
from domain.value.types import ValuationSnapshot


def _current_cache_file(cache_dir: Path) -> Optional[Path]:
    ym = datetime.utcnow().strftime('%Y-%m')
    p = cache_dir / f'fundamentals_{ym}.parquet'
    return p if p.exists() else None


def check_ticker(ticker: str, cache_dir: Path = Path('cache/value')) -> Optional[dict]:
    """Return the flat dict for `ticker` if present in this month's cache."""
    cache_file = _current_cache_file(cache_dir)
    if cache_file is None:
        print(f"ERROR: no cached fundamentals for {datetime.utcnow().strftime('%Y-%m')} at {cache_dir}",
              file=sys.stderr)
        return None
    df = pd.read_parquet(cache_file)
    inputs_list = [normalize_fundamentals(r) for _, r in df.iterrows() if isinstance(r.to_dict(), dict)]
    hit = next((v for v in inputs_list if v.ticker == ticker), None)
    if hit is None:
        print(f"ERROR: {ticker} not in current cache", file=sys.stderr)
        return None

    sector_medians = compute_sector_medians(inputs_list)
    sr = compute_sector_relative(hit, sector_medians)
    fd = compute_fundamental_divergence(hit)
    vol = compute_volume_overlay(hit)
    base_rank = (sum(s.score for s in (sr, fd) if s is not None)
                  / max(1, sum(1 for s in (sr, fd) if s is not None)))
    short = compute_short_overlay(hit, base_rank=base_rank, prior_short_interest_pct=None)
    role1 = assign_role1(sr, fd, vol, short) if (sr or fd) else None
    role2 = assign_role2(sr, fd, vol, short) if (sr or fd) else None

    snap = ValuationSnapshot(
        ticker=ticker, inputs=hit,
        sector_relative=sr, fundamental_divergence=fd,
        volume=vol, short_overlay=short, role1=role1, role2=role2,
        skip_reason=None if role1 else 'insufficient_data',
    )
    return snapshot_to_dict(snap)


def _pretty_print(d: dict) -> None:
    print(f"\n=== {d['ticker']} — {d.get('sector', '?')} ===")
    print(f"Spot: ${d.get('spot', 0):.2f}")
    if 'sector_relative_score' in d:
        print(f"\nSector-relative score: {d['sector_relative_score']:.2f} / 10")
        m = d.get('sector_multiples', {})
        for k in ('pe_fwd', 'ps_ttm', 'pb', 'ev_ebitda'):
            if m.get(k) is not None:
                pm = m.get(f'{k}_peer_median')
                dp = m.get(f'{k}_discount_pct')
                dp_s = f"({dp:+.1f}% vs peer)" if dp is not None else ""
                print(f"  {k}: {m[k]:.2f}  peer median {pm:.2f}  {dp_s}")
    if 'fundamental_divergence_score' in d:
        print(f"\nFundamental divergence: {d['fundamental_divergence_score']:.2f} / 10")
        t = d.get('fundamental_trend', {})
        if t.get('revenue_growth_yoy_pct') is not None:
            print(f"  Revenue growth YoY: {t['revenue_growth_yoy_pct']:+.1f}%")
        if t.get('eps_growth_yoy_pct') is not None:
            print(f"  EPS growth YoY:     {t['eps_growth_yoy_pct']:+.1f}%")
        if t.get('price_growth_yoy_pct') is not None:
            print(f"  Price growth YoY:   {t['price_growth_yoy_pct']:+.1f}%")
        if t.get('divergence_pp') is not None:
            print(f"  Divergence:         {t['divergence_pp']:+.1f} pp")
    if 'volume_tag' in d:
        v = d.get('volume', {})
        r = v.get('ratio')
        r_s = f" (5d/20d = {r:.2f})" if r is not None else ""
        print(f"\nVolume: {d['volume_tag']}{r_s}")
    if 'short_interest_pct' in d and d.get('short_interest_pct') is not None:
        print(f"Short interest: {d['short_interest_pct']:.1f}%   "
              f"Days to cover: {d.get('days_to_cover') or 0:.1f}")
    if 'priority' in d:
        print(f"\nRole 1 priority: {d['priority']}")
    if 'composite_score' in d:
        print(f"Role 2 composite: {d['composite_score']:.1f} / 100")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description='Value scanner — single ticker check')
    p.add_argument('ticker')
    p.add_argument('--cache-dir', type=Path, default=Path('cache/value'))
    args = p.parse_args(argv)

    out = check_ticker(args.ticker, cache_dir=args.cache_dir)
    if out is None:
        return 1
    _pretty_print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 12.4: Run tests, expect all pass**

```bash
python3 -m pytest tests/scripts/test_value_check.py -v
```

Expected: 2/2 PASS.

- [ ] **Step 12.5: Commit**

```bash
git add scripts/value_check.py tests/scripts/test_value_check.py
git commit -m "$(cat <<'EOF'
feat(value): value_check CLI — single-ticker deep dive

Loads current month's fundamentals cache Parquet, recomputes all
per-methodology results for one ticker, prints a formatted
breakdown of sector-relative ratios, fundamental trend, volume, and
short interest.  No re-fetch — relies on the cache written by
value_watchlist.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: End-to-end verification

No code changes.  Runs the full pipeline against the small fixture universe.

- [ ] **Step 13.1: Confirm full value module test suite passes**

```bash
python3 -m pytest tests/domain/value/ tests/data/sources/test_yahooquery_fundamentals.py tests/scripts/test_refresh_russell2000.py tests/scripts/test_value_watchlist.py tests/scripts/test_value_check.py -q
```

Expected: all tests pass.

- [ ] **Step 13.2: Confirm no regression on existing suite**

```bash
python3 -m pytest -q 2>&1 | tail -10
```

Expected: pre-existing test count + all the new value tests, no new failures.

- [ ] **Step 13.3: Optional live smoke test**

If the Russell 2000 constituents file has been populated (Step 10.5), run the pipeline against a tiny sample:

```bash
python3 -c "
from pathlib import Path
from scripts.value_watchlist import run_watchlist
import json
mini = ['AAPL', 'MSFT', 'INTC', 'NVDA', 'AMD']
report = run_watchlist(
    constituents=mini,
    out_dir=Path('/tmp/value_smoke'), cache_dir=Path('/tmp/value_smoke_cache'),
    universe_name='smoke', top_n=5, use_cache=False,
)
print(json.dumps({
    'graded': report.graded_size,
    'longs_role1': [r['ticker'] for r in report.role1_ranked_longs],
    'longs_role2': [r['ticker'] for r in report.role2_ranked_longs],
}, indent=2))
"
```

Expected: prints a small report.  If this fails for network reasons, that's fine — the test suite is the acceptance gate.

- [ ] **Step 13.4: Print summary**

Print the file list and commit hashes for the record:

```bash
git log --oneline -13 -- domain/value/ data/sources/yahooquery_fundamentals.py scripts/refresh_russell2000.py scripts/value_watchlist.py scripts/value_check.py tests/domain/value/ tests/data/sources/ tests/scripts/test_refresh_russell2000.py tests/scripts/test_value_watchlist.py tests/scripts/test_value_check.py
```

## Done criteria (full plan)

- All 12 code tasks complete + Task 13 verification green.
- All new tests pass.  No regression to existing suite.
- `python -m scripts.value_watchlist` runs end-to-end on either universe (may take 30-45 min cold on Russell 2000).
- `python -m scripts.value_check TICKER` returns a formatted breakdown.
- Output JSON contains all 4 ranked lists (Role 1 longs/shorts, Role 2 longs/shorts).
- Monthly archive written alongside `latest.json`.

## Rollback

Every task commits independently, so rollback is per-commit `git revert`.  The safest low-risk revert order if the whole feature needs pulling: Task 12 → Task 11 → Task 10 → … → Task 1.
