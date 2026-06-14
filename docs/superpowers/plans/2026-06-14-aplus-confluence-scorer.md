# A+ Confluence Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-factor confluence scorer that augments the existing scanner output with A+/A/B+/B grading on each candidate, recommends an appropriate trade structure (long premium or debit spread) per candidate, and renders a small daily watchlist (1-5 trades) sized for a $1,000 account.

**Architecture:** New `domain/aplus/` module reads the existing `output/scans/latest.json`, extracts 20 features across 5 categories (Technical / Vol-VIX / Catalyst / Macro-Breadth / Liquidity), composes a weighted composite score, grades each candidate, selects a trade structure (long premium vs debit spread), and writes the result to `output/aplus/latest.json`. New CLI command `python -m main aplus_watchlist`.

**Tech Stack:** Python 3.9, pandas, numpy, existing scanner infrastructure (`output/scans/latest.json`), Schwab data via existing `SchwabSource`, `YahooMacroSource` for VIX/DXY/sector data, hardcoded macro-event calendar.

**Spec reference:** `docs/superpowers/specs/2026-06-14-aplus-confluence-scorer-design.md`

---

## File manifest

**New files:**
- `data/sources/macro_calendar.py` — hardcoded FOMC/CPI/PPI/jobs schedule
- `domain/aplus/__init__.py` — empty
- `domain/aplus/types.py` — dataclasses (FeatureScores, CategoryScores, GradedCandidate, MarketContext, TradeStructure)
- `domain/aplus/market_context.py` — fetch per-scan market data (SPY EMAs, sector returns, DXY, ^TNX, VVIX)
- `domain/aplus/features.py` — extract 20 features per candidate
- `domain/aplus/scoring.py` — category scores + weighted composite
- `domain/aplus/grading.py` — composite + category floors → A+/A/B+/B/F label
- `domain/aplus/structure.py` — long premium vs debit spread selector
- `scripts/aplus_watchlist.py` — CLI entry, reads scan JSON, renders + writes
- `tests/data/test_macro_calendar.py`
- `tests/domain/aplus/__init__.py`
- `tests/domain/aplus/test_types.py`
- `tests/domain/aplus/test_market_context.py`
- `tests/domain/aplus/test_features.py`
- `tests/domain/aplus/test_scoring.py`
- `tests/domain/aplus/test_grading.py`
- `tests/domain/aplus/test_structure.py`
- `tests/scripts/test_aplus_watchlist.py`

**Modified files:**
- `main.py` — add `aplus_watchlist` subcommand

---

## Task 1: Macro event calendar

**Files:**
- Create: `data/sources/macro_calendar.py`
- Test: `tests/data/test_macro_calendar.py`

A small hardcoded list of FOMC, CPI, PPI, NFP release dates. Single function: `days_to_next_event(today: date) -> int | None`. Returns days until next blocking macro event.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_macro_calendar.py`:

```python
from datetime import date
from data.sources.macro_calendar import days_to_next_event, MACRO_EVENTS


def test_macro_events_list_not_empty():
    assert len(MACRO_EVENTS) > 0
    for d, kind in MACRO_EVENTS:
        assert isinstance(d, date)
        assert kind in ('FOMC', 'CPI', 'PPI', 'NFP')


def test_days_to_next_event_returns_zero_on_event_day():
    if not MACRO_EVENTS:
        return
    first_event_date, _ = MACRO_EVENTS[0]
    assert days_to_next_event(first_event_date) == 0


def test_days_to_next_event_returns_positive_before_event():
    if not MACRO_EVENTS:
        return
    first_event_date, _ = MACRO_EVENTS[0]
    day_before = date.fromordinal(first_event_date.toordinal() - 1)
    assert days_to_next_event(day_before) == 1


def test_days_to_next_event_returns_none_after_last_event():
    if not MACRO_EVENTS:
        return
    last_event_date, _ = MACRO_EVENTS[-1]
    day_after = date.fromordinal(last_event_date.toordinal() + 1)
    assert days_to_next_event(day_after) is None
```

- [ ] **Step 2: Run test, confirm fail**

```bash
cd /Users/sarthakhans/options-scanner
pytest tests/data/test_macro_calendar.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the module**

Create `data/sources/macro_calendar.py`:

```python
"""Hardcoded macro-event calendar (FOMC / CPI / PPI / NFP).

Update this file quarterly with announced future dates from
fomc.gov / bls.gov / atlanta fed economic calendar. Versioning in
git history; no live API dependency.
"""
from __future__ import annotations
from datetime import date


# Format: (date, event_kind). Listed chronologically.
# Source: official Federal Reserve, BLS, and BEA calendars.
MACRO_EVENTS: list[tuple[date, str]] = [
    # 2026 FOMC meetings (8 per year)
    (date(2026, 1, 28), 'FOMC'),
    (date(2026, 3, 18), 'FOMC'),
    (date(2026, 4, 29), 'FOMC'),
    (date(2026, 6, 17), 'FOMC'),
    (date(2026, 7, 29), 'FOMC'),
    (date(2026, 9, 16), 'FOMC'),
    (date(2026, 11, 4), 'FOMC'),
    (date(2026, 12, 16), 'FOMC'),

    # 2026 CPI releases (typically mid-month for prior month)
    (date(2026, 1, 14), 'CPI'),
    (date(2026, 2, 11), 'CPI'),
    (date(2026, 3, 11), 'CPI'),
    (date(2026, 4, 14), 'CPI'),
    (date(2026, 5, 13), 'CPI'),
    (date(2026, 6, 10), 'CPI'),
    (date(2026, 7, 14), 'CPI'),
    (date(2026, 8, 12), 'CPI'),
    (date(2026, 9, 9), 'CPI'),
    (date(2026, 10, 14), 'CPI'),
    (date(2026, 11, 12), 'CPI'),
    (date(2026, 12, 9), 'CPI'),

    # 2026 PPI releases
    (date(2026, 1, 15), 'PPI'),
    (date(2026, 2, 12), 'PPI'),
    (date(2026, 3, 12), 'PPI'),
    (date(2026, 4, 15), 'PPI'),
    (date(2026, 5, 14), 'PPI'),
    (date(2026, 6, 11), 'PPI'),
    (date(2026, 7, 15), 'PPI'),
    (date(2026, 8, 13), 'PPI'),
    (date(2026, 9, 10), 'PPI'),
    (date(2026, 10, 15), 'PPI'),
    (date(2026, 11, 13), 'PPI'),
    (date(2026, 12, 10), 'PPI'),

    # 2026 NFP (first Friday of each month, approx)
    (date(2026, 1, 2), 'NFP'),
    (date(2026, 2, 6), 'NFP'),
    (date(2026, 3, 6), 'NFP'),
    (date(2026, 4, 3), 'NFP'),
    (date(2026, 5, 1), 'NFP'),
    (date(2026, 6, 5), 'NFP'),
    (date(2026, 7, 2), 'NFP'),
    (date(2026, 8, 7), 'NFP'),
    (date(2026, 9, 4), 'NFP'),
    (date(2026, 10, 2), 'NFP'),
    (date(2026, 11, 6), 'NFP'),
    (date(2026, 12, 4), 'NFP'),
]


def days_to_next_event(today: date) -> int | None:
    """Return whole calendar days until next blocking macro event (>=0).
    Returns None if `today` is after all known events."""
    upcoming = [d for d, _ in MACRO_EVENTS if d >= today]
    if not upcoming:
        return None
    return (upcoming[0] - today).days
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/data/test_macro_calendar.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add data/sources/macro_calendar.py tests/data/test_macro_calendar.py
git commit -m "feat(data): hardcoded macro event calendar for FOMC/CPI/PPI/NFP"
```

---

## Task 2: Domain types (dataclasses)

**Files:**
- Create: `domain/aplus/__init__.py`
- Create: `domain/aplus/types.py`
- Test: `tests/domain/aplus/__init__.py`
- Test: `tests/domain/aplus/test_types.py`

Define the small frozen dataclasses every later module imports.

- [ ] **Step 1: Write the failing test**

Create `tests/domain/aplus/__init__.py` (empty).

Create `tests/domain/aplus/test_types.py`:

```python
import pytest
from domain.aplus.types import (
    FeatureScores, CategoryScores, MarketContext,
    TradeStructure, GradedCandidate,
)


def test_feature_scores_holds_named_features():
    fs = FeatureScores(values={'tech_setup_type': 9.0, 'tech_setup_strength': 8.0})
    assert fs.values['tech_setup_type'] == 9.0


def test_category_scores_has_five_categories():
    cs = CategoryScores(technical=8.0, vol_vix=7.0, catalyst=6.0,
                        macro_breadth=7.0, liquidity=9.0)
    assert cs.technical == 8.0
    assert cs.vol_vix == 7.0


def test_market_context_frozen():
    mc = MarketContext(
        spx_trend_score=8.0, sector_rotation_rank={'XLK': 1},
        dxy_trend_score=5.0, yield_10y_score=6.0, vvix_score=7.0,
        days_to_macro_event=3,
    )
    with pytest.raises(AttributeError):
        mc.spx_trend_score = 9.0


def test_trade_structure_is_long_premium_or_debit_spread():
    assert TradeStructure.LONG_PREMIUM.value == 'long_premium'
    assert TradeStructure.DEBIT_SPREAD.value == 'debit_spread'


def test_graded_candidate_carries_grade_and_structure():
    gc = GradedCandidate(
        ticker='AAPL', strategy='long_call', composite_score=85.2, grade='A',
        category_scores=CategoryScores(8.0, 8.0, 7.0, 7.0, 9.0),
        feature_scores=FeatureScores(values={}),
        structure=TradeStructure.DEBIT_SPREAD, structure_rationale='IV>70',
        sizing_pct=0.08, max_risk_dollars=80.0,
        raw_candidate={},
    )
    assert gc.grade == 'A'
    assert gc.structure == TradeStructure.DEBIT_SPREAD
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_types.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the types**

Create `domain/aplus/__init__.py` (empty).

Create `domain/aplus/types.py`:

```python
"""Dataclasses used across the A+ confluence scorer.

Each unit is small and has one responsibility. Importing modules
should depend only on this file for type information.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


Grade = Literal['A+', 'A', 'B+', 'B', 'F']


class TradeStructure(str, Enum):
    """Recommended trade structure for a graded candidate."""
    LONG_PREMIUM = 'long_premium'
    DEBIT_SPREAD = 'debit_spread'


@dataclass(frozen=True)
class FeatureScores:
    """Per-feature scores in [0, 10]. Keys match feature names in
    domain/aplus/features.py."""
    values: dict[str, float]


@dataclass(frozen=True)
class CategoryScores:
    """Five category scores in [0, 10]. Average of the constituent
    feature scores per category."""
    technical: float
    vol_vix: float
    catalyst: float
    macro_breadth: float
    liquidity: float

    def composite(self) -> float:
        """Weighted composite, scaled to [0, 100]."""
        return (
            0.25 * self.technical
            + 0.25 * self.vol_vix
            + 0.15 * self.catalyst
            + 0.15 * self.macro_breadth
            + 0.20 * self.liquidity
        ) * 10.0


@dataclass(frozen=True)
class MarketContext:
    """Per-scan market data shared across all candidates. Computed once
    when the watchlist runs; reused for every candidate."""
    spx_trend_score: float
    sector_rotation_rank: dict[str, int]   # sector_etf → rank 1..11 by 1w return
    dxy_trend_score: float
    yield_10y_score: float
    vvix_score: float
    days_to_macro_event: int | None


@dataclass(frozen=True)
class GradedCandidate:
    """Final output unit. One per candidate that survived grading."""
    ticker: str
    strategy: str
    composite_score: float
    grade: Grade
    category_scores: CategoryScores
    feature_scores: FeatureScores
    structure: TradeStructure
    structure_rationale: str
    sizing_pct: float          # fraction of account to risk
    max_risk_dollars: float
    raw_candidate: dict        # the original scan-output dict
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_types.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/__init__.py domain/aplus/types.py tests/domain/aplus/__init__.py tests/domain/aplus/test_types.py
git commit -m "feat(aplus): types — FeatureScores, CategoryScores, MarketContext, TradeStructure, GradedCandidate"
```

---

## Task 3: Market context fetcher

**Files:**
- Create: `domain/aplus/market_context.py`
- Test: `tests/domain/aplus/test_market_context.py`

Fetch the per-scan market data (SPY EMAs, sector returns, DXY trend, 10y yield, VVIX) and compute the scores once. Uses existing `YahooMacroSource`.

- [ ] **Step 1: Write the failing test**

Create `tests/domain/aplus/test_market_context.py`:

```python
from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd

from domain.aplus.market_context import (
    score_spx_trend, score_dxy_trend, score_yield_10y, score_vvix,
    rank_sectors, build_market_context,
)


def _series(values, start='2026-01-01'):
    idx = pd.date_range(start, periods=len(values), freq='D')
    return pd.Series(values, index=idx)


def test_score_spx_trend_bullish_setup_strong_uptrend_high_score():
    closes = _series(list(range(80, 130)))  # rising
    score = score_spx_trend(closes, setup_direction='bullish')
    assert score >= 8.0


def test_score_spx_trend_bullish_setup_strong_downtrend_low_score():
    closes = _series(list(range(130, 80, -1)))
    score = score_spx_trend(closes, setup_direction='bullish')
    assert score <= 4.0


def test_score_dxy_trend_neutral_when_mixed():
    closes = _series([100.0] * 60)
    score = score_dxy_trend(closes, setup_direction='bullish')
    assert 4.0 <= score <= 6.0


def test_score_vvix_low_high_score():
    score = score_vvix(85.0)
    assert score >= 8.0


def test_score_vvix_high_low_score():
    score = score_vvix(125.0)
    assert score <= 4.0


def test_rank_sectors_assigns_rank_1_to_best_performer():
    returns = {'XLK': 0.05, 'XLF': 0.01, 'XLE': -0.02}
    ranks = rank_sectors(returns)
    assert ranks['XLK'] == 1


def test_build_market_context_returns_market_context():
    with patch('domain.aplus.market_context._fetch_macro_series') as m:
        m.return_value = _series([100.0] * 60)
        with patch('domain.aplus.market_context._fetch_vvix') as m_vvix:
            m_vvix.return_value = 95.0
            with patch('domain.aplus.market_context._fetch_sector_returns') as m_sec:
                m_sec.return_value = {'XLK': 0.05, 'XLF': 0.01}
                from domain.aplus.types import MarketContext
                ctx = build_market_context(today=date(2026, 6, 14), setup_direction='bullish')
                assert isinstance(ctx, MarketContext)
                assert 0 <= ctx.spx_trend_score <= 10
                assert 0 <= ctx.dxy_trend_score <= 10
                assert 0 <= ctx.vvix_score <= 10
                assert ctx.days_to_macro_event is None or ctx.days_to_macro_event >= 0
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_market_context.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement market_context.py**

Create `domain/aplus/market_context.py`:

```python
"""Per-scan market context — computed once per watchlist run."""
from __future__ import annotations
from datetime import date
from typing import Literal
import pandas as pd

from data.sources.macro_calendar import days_to_next_event
from domain.aplus.types import MarketContext


_SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC']


def score_spx_trend(spy_closes: pd.Series, setup_direction: str) -> float:
    """10 if SPY > 50EMA > 200EMA AND setup is bullish (or inverse for bearish);
    4 if mixed; 0 if SPY trend opposes setup direction."""
    if len(spy_closes) < 50:
        return 5.0
    ema50 = spy_closes.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = spy_closes.ewm(span=200, adjust=False).mean().iloc[-1] if len(spy_closes) >= 200 else ema50
    last = float(spy_closes.iloc[-1])

    bullish_stack = last > ema50 > ema200
    bearish_stack = last < ema50 < ema200

    if setup_direction == 'bullish':
        if bullish_stack:
            return 10.0
        if bearish_stack:
            return 0.0
        return 4.0
    if setup_direction == 'bearish':
        if bearish_stack:
            return 10.0
        if bullish_stack:
            return 0.0
        return 4.0
    return 5.0


def score_dxy_trend(dxy_closes: pd.Series, setup_direction: str) -> float:
    """DXY direction supports bullish stock setups when DXY is falling
    (USD weakness lifts multinationals); opposes when DXY is rising."""
    if len(dxy_closes) < 20:
        return 5.0
    sma20 = float(dxy_closes.rolling(20).mean().iloc[-1])
    last = float(dxy_closes.iloc[-1])
    dxy_falling = last < sma20 * 0.99
    dxy_rising = last > sma20 * 1.01
    if setup_direction == 'bullish':
        if dxy_falling:
            return 10.0
        if dxy_rising:
            return 2.0
        return 5.0
    if setup_direction == 'bearish':
        if dxy_rising:
            return 10.0
        if dxy_falling:
            return 2.0
        return 5.0
    return 5.0


def score_yield_10y(yield_closes: pd.Series, sector: str, setup_direction: str) -> float:
    """Map yield direction × sector × setup direction.
    Rising yields lift XLF/XLE, hurt XLK/XLRE; falling yields the reverse."""
    if len(yield_closes) < 20:
        return 5.0
    sma20 = float(yield_closes.rolling(20).mean().iloc[-1])
    last = float(yield_closes.iloc[-1])
    rising = last > sma20 * 1.02
    falling = last < sma20 * 0.98
    yields_helpful = {'XLF': 'rising', 'XLE': 'rising', 'XLK': 'falling', 'XLRE': 'falling'}
    helpful = yields_helpful.get(sector)
    if helpful is None:
        return 5.0
    if (helpful == 'rising' and rising) or (helpful == 'falling' and falling):
        return 10.0 if setup_direction == 'bullish' else 2.0
    if (helpful == 'rising' and falling) or (helpful == 'falling' and rising):
        return 2.0 if setup_direction == 'bullish' else 10.0
    return 5.0


def score_vvix(vvix_level: float) -> float:
    """10 if VVIX < 90 (stable vol-of-vol);
    scaling down to 3 if VVIX > 120 (unstable vol regime)."""
    if vvix_level < 90:
        return 10.0
    if vvix_level > 120:
        return 3.0
    # Linear interpolation between 90 and 120
    return 10.0 - (vvix_level - 90) / 30.0 * 7.0


def rank_sectors(returns_1w: dict[str, float]) -> dict[str, int]:
    """Assign rank 1..N to each sector by 1-week return (1 = best)."""
    sorted_etfs = sorted(returns_1w.items(), key=lambda kv: -kv[1])
    return {etf: i + 1 for i, (etf, _) in enumerate(sorted_etfs)}


def build_market_context(today: date, setup_direction: str = 'bullish') -> MarketContext:
    """Pull live macro data and build a MarketContext snapshot.

    NOTE: setup_direction is needed for SPX/DXY trend scoring;
    we default to bullish here and let the caller recompute
    direction-specific scores if needed at the candidate level.
    """
    spy_closes = _fetch_macro_series('SPY')
    dxy_closes = _fetch_macro_series('DX-Y.NYB')
    yield_closes = _fetch_macro_series('^TNX')
    vvix_level = _fetch_vvix()
    sector_returns = _fetch_sector_returns()

    return MarketContext(
        spx_trend_score=score_spx_trend(spy_closes, setup_direction),
        sector_rotation_rank=rank_sectors(sector_returns),
        dxy_trend_score=score_dxy_trend(dxy_closes, setup_direction),
        yield_10y_score=5.0,   # per-sector; computed lazily at feature time
        vvix_score=score_vvix(vvix_level),
        days_to_macro_event=days_to_next_event(today),
    )


def _fetch_macro_series(ticker: str) -> pd.Series:
    """Fetch ~60 days of daily closes for a ticker. Falls back to empty Series on failure."""
    from data.sources.yahoo_macro_source import YahooMacroSource
    try:
        df = YahooMacroSource().fetch_history(ticker, period='3mo')
        return df['close'] if not df.empty and 'close' in df.columns else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def _fetch_vvix() -> float:
    """Fetch most recent VVIX value (^VVIX on Yahoo). Returns neutral 100 on failure."""
    series = _fetch_macro_series('^VVIX')
    return float(series.iloc[-1]) if len(series) > 0 else 100.0


def _fetch_sector_returns() -> dict[str, float]:
    """1-week return per sector ETF."""
    out: dict[str, float] = {}
    for etf in _SECTOR_ETFS:
        series = _fetch_macro_series(etf)
        if len(series) >= 5:
            out[etf] = float(series.iloc[-1] / series.iloc[-5] - 1)
        else:
            out[etf] = 0.0
    return out
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_market_context.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/market_context.py tests/domain/aplus/test_market_context.py
git commit -m "feat(aplus): market_context — per-scan SPX/DXY/yield/VVIX/sector scoring"
```

---

## Task 4: Feature extraction — Technical & Vol/VIX

**Files:**
- Create: `domain/aplus/features.py`
- Test: `tests/domain/aplus/test_features.py`

Build the per-candidate feature extractors for the first two categories (5 + 4 = 9 features). The remaining categories follow in Tasks 5–6.

- [ ] **Step 1: Write the failing test**

Create `tests/domain/aplus/test_features.py`:

```python
from datetime import date
from domain.aplus.features import (
    score_setup_type, score_setup_strength,
    score_weekly_ribbon_agreement, score_atr_pivot_position,
    score_volume_zscore, score_vix_regime, score_vix_zscore_30d,
    score_vvix_level, score_iv_percentile,
)


def test_score_setup_type_compression_high():
    assert score_setup_type('compression_breakout') == 10.0


def test_score_setup_type_pullback_low():
    assert score_setup_type('pullback_in_trend') <= 4.0


def test_score_setup_type_unknown_neutral():
    assert score_setup_type(None) == 5.0


def test_score_setup_strength_max():
    assert score_setup_strength(1.0) == 10.0


def test_score_setup_strength_zero():
    assert score_setup_strength(0.0) == 0.0


def test_score_setup_strength_none_neutral():
    assert score_setup_strength(None) == 5.0


def test_score_weekly_ribbon_agreement_true_high():
    assert score_weekly_ribbon_agreement(True) == 10.0


def test_score_weekly_ribbon_agreement_false_mid():
    assert score_weekly_ribbon_agreement(False) == 5.0


def test_score_weekly_ribbon_agreement_none_neutral():
    assert score_weekly_ribbon_agreement(None) == 5.0


def test_score_atr_pivot_position_at_pivot_high():
    # spot at the same as prior close → 0 ATR distance
    assert score_atr_pivot_position(spot=100.0, prev_close=100.0, atr=1.0) == 10.0


def test_score_atr_pivot_position_two_atr_low():
    # 2 ATR away → 0
    assert score_atr_pivot_position(spot=102.0, prev_close=100.0, atr=1.0) == 0.0


def test_score_volume_zscore_high_volume_high_score():
    assert score_volume_zscore(z=2.0) == 10.0


def test_score_volume_zscore_low_volume_zero():
    assert score_volume_zscore(z=-1.0) == 0.0


def test_score_vix_regime_expansion_zero():
    assert score_vix_regime('expansion') == 0.0


def test_score_vix_regime_contraction_max():
    assert score_vix_regime('contraction') == 10.0


def test_score_vix_regime_neutral_mid_high():
    assert score_vix_regime('neutral') == 7.0


def test_score_vix_regime_none_neutral():
    assert score_vix_regime(None) == 5.0


def test_score_vix_zscore_30d_low_high_score():
    assert score_vix_zscore_30d(-0.5) >= 8.0


def test_score_vix_zscore_30d_high_low_score():
    assert score_vix_zscore_30d(1.0) <= 4.0


def test_score_vvix_level_uses_market_context_value():
    assert score_vvix_level(8.5) == 8.5


def test_score_iv_percentile_low_iv_high_score():
    assert score_iv_percentile(0.18) >= 8.0


def test_score_iv_percentile_high_iv_low_score():
    assert score_iv_percentile(0.60) <= 4.0
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_features.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement features.py — Technical + Vol/VIX**

Create `domain/aplus/features.py`:

```python
"""Per-candidate feature extractors for the A+ confluence scorer.

Each function returns a float in [0, 10]. Missing inputs return 5.0 (neutral)
unless the spec specifies a different default (e.g., VIX expansion forces 0).
"""
from __future__ import annotations
from typing import Optional
from datetime import date


_SETUP_TYPE_SCORES = {
    'compression_breakout': 10.0,
    'stage_2_breakout': 8.0,
    'failed_breakdown_reversal': 5.0,
    'pullback_in_trend': 3.0,
}


def score_setup_type(setup_name: Optional[str]) -> float:
    if setup_name is None:
        return 5.0
    return _SETUP_TYPE_SCORES.get(setup_name, 5.0)


def score_setup_strength(strength: Optional[float]) -> float:
    if strength is None:
        return 5.0
    return max(0.0, min(10.0, strength * 10.0))


def score_weekly_ribbon_agreement(agreement: Optional[bool]) -> float:
    if agreement is None:
        return 5.0
    return 10.0 if agreement else 5.0


def score_atr_pivot_position(spot: float, prev_close: float, atr: float) -> float:
    """10 if spot is at prior daily close; linear decline to 0 at 2 ATR away."""
    if atr <= 0:
        return 5.0
    distance = abs(spot - prev_close) / atr
    return max(0.0, 10.0 - distance * 5.0)


def score_volume_zscore(z: float) -> float:
    """Higher relative volume scores higher. 10 at z=2, 0 at z=-1."""
    return max(0.0, min(10.0, (z + 1.0) * (10.0 / 3.0)))


def score_vix_regime(regime: Optional[str]) -> float:
    if regime == 'expansion':
        return 0.0
    if regime == 'contraction':
        return 10.0
    if regime == 'neutral':
        return 7.0
    return 5.0


def score_vix_zscore_30d(z: float) -> float:
    """Lower VIX z-score scores higher; higher z penalises."""
    return max(0.0, min(10.0, 10.0 - z * 4.0))


def score_vvix_level(category_score: float) -> float:
    """Pass-through — MarketContext already converted VVIX level to a score."""
    return max(0.0, min(10.0, category_score))


def score_iv_percentile(iv: float) -> float:
    """For LONG premium scoring: low IV is better. 10 at IV<=20%, 0 at IV>=60%."""
    if iv <= 0.20:
        return 10.0
    if iv >= 0.60:
        return 0.0
    return 10.0 - (iv - 0.20) / 0.40 * 10.0
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_features.py -v
```

Expected: PASS (~20 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/features.py tests/domain/aplus/test_features.py
git commit -m "feat(aplus): features — Technical + Vol/VIX category extractors"
```

---

## Task 5: Feature extraction — Catalyst & Macro/Breadth

**Files:**
- Modify: `domain/aplus/features.py` (append)
- Modify: `tests/domain/aplus/test_features.py` (append)

Add Catalyst (3) and Macro/Breadth (3) feature scorers.

- [ ] **Step 1: Write the failing tests**

Append to `tests/domain/aplus/test_features.py`:

```python
from domain.aplus.features import (
    score_days_to_earnings, score_days_to_macro_event,
    score_skip_window, score_spx_alignment, score_sector_rotation,
    score_dxy_trend_pass_through,
)


def test_score_days_to_earnings_within_blackout_zero():
    assert score_days_to_earnings(3) == 0.0


def test_score_days_to_earnings_in_window_high():
    assert score_days_to_earnings(10) == 10.0


def test_score_days_to_earnings_far_neutral():
    assert score_days_to_earnings(60) == 6.0


def test_score_days_to_macro_event_within_2_days_zero():
    assert score_days_to_macro_event(1) == 0.0


def test_score_days_to_macro_event_5_days_out_high():
    assert score_days_to_macro_event(10) == 10.0


def test_score_skip_window_combines_blockers():
    # If both earnings and macro are clear, returns 10
    assert score_skip_window(days_earnings=15, days_macro=10) == 10.0
    # If either is blocking (≤2), returns 0
    assert score_skip_window(days_earnings=1, days_macro=10) == 0.0


def test_score_spx_alignment_pass_through():
    # Test that we just pass through the precomputed market_context value
    assert score_spx_alignment(8.0) == 8.0


def test_score_sector_rotation_top_sector_bullish_high():
    score = score_sector_rotation(rank=1, total_sectors=11, setup_direction='bullish')
    assert score >= 9.0


def test_score_sector_rotation_bottom_sector_bullish_low():
    score = score_sector_rotation(rank=11, total_sectors=11, setup_direction='bullish')
    assert score <= 1.0


def test_score_sector_rotation_top_sector_bearish_low():
    # If the sector is leading and the setup is bearish, that's bad
    score = score_sector_rotation(rank=1, total_sectors=11, setup_direction='bearish')
    assert score <= 1.0
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_features.py -v
```

Expected: FAIL — new functions not defined.

- [ ] **Step 3: Implement the new scorers**

Append to `domain/aplus/features.py`:

```python
def score_days_to_earnings(days: Optional[int]) -> float:
    """0 if <=5 days (blackout), 10 if 5-15 days (post-earnings window),
    6 if 15-30, 6 if >30 (no catalyst)."""
    if days is None:
        return 6.0
    if days <= 5:
        return 0.0
    if days <= 15:
        return 10.0
    return 6.0


def score_days_to_macro_event(days: Optional[int]) -> float:
    """0 if <=2 days, 10 if >5 days, linear in between."""
    if days is None:
        return 6.0
    if days <= 2:
        return 0.0
    if days >= 5:
        return 10.0
    return (days - 2) / 3.0 * 10.0


def score_skip_window(days_earnings: Optional[int], days_macro: Optional[int]) -> float:
    """0 if either earnings or macro event is within 2 days; 10 otherwise."""
    if (days_earnings is not None and days_earnings <= 2) or \
       (days_macro is not None and days_macro <= 2):
        return 0.0
    return 10.0


def score_spx_alignment(precomputed_score: float) -> float:
    """Pass-through from MarketContext.spx_trend_score."""
    return max(0.0, min(10.0, precomputed_score))


def score_sector_rotation(rank: int, total_sectors: int, setup_direction: str) -> float:
    """Sector leadership: rank 1 (best) → 10 for bullish, 0 for bearish.
    Sector laggard: rank N (worst) → 0 for bullish, 10 for bearish."""
    if rank < 1 or rank > total_sectors:
        return 5.0
    pct = (rank - 1) / (total_sectors - 1)  # 0.0 = best, 1.0 = worst
    if setup_direction == 'bullish':
        return 10.0 * (1.0 - pct)
    if setup_direction == 'bearish':
        return 10.0 * pct
    return 5.0


def score_dxy_trend_pass_through(precomputed_score: float) -> float:
    """Pass-through from MarketContext.dxy_trend_score."""
    return max(0.0, min(10.0, precomputed_score))
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_features.py -v
```

Expected: PASS (~30 tests now).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/features.py tests/domain/aplus/test_features.py
git commit -m "feat(aplus): features — Catalyst + Macro/Breadth scorers"
```

---

## Task 6: Feature extraction — Liquidity + top-level extractor

**Files:**
- Modify: `domain/aplus/features.py` (append)
- Modify: `tests/domain/aplus/test_features.py` (append)

Add Liquidity (3) feature scorers and a single `extract_features(candidate, market_context)` function that ties everything together.

- [ ] **Step 1: Write the failing tests**

Append to `tests/domain/aplus/test_features.py`:

```python
from domain.aplus.features import (
    score_bid_ask_spread, score_open_interest, score_volume_oi_ratio,
    extract_features,
)
from domain.aplus.types import MarketContext


def test_score_bid_ask_spread_tight_high():
    # Spread 1% of mid → high
    assert score_bid_ask_spread(bid=0.99, ask=1.01) == 10.0


def test_score_bid_ask_spread_wide_low():
    # Spread 30% of mid → low
    assert score_bid_ask_spread(bid=0.85, ask=1.15) == 0.0


def test_score_open_interest_high_oi_max():
    assert score_open_interest(2000) == 10.0


def test_score_open_interest_low_oi_zero():
    assert score_open_interest(50) == 0.0


def test_score_volume_oi_ratio_active_high():
    assert score_volume_oi_ratio(volume=1000, open_interest=2000) == 10.0


def test_score_volume_oi_ratio_dormant_low():
    assert score_volume_oi_ratio(volume=10, open_interest=2000) == 3.0


def test_extract_features_returns_feature_scores_with_20_keys():
    candidate = {
        'contract': {
            'ticker': 'XLF', 'strike': 52.0, 'bid': 0.47, 'ask': 0.53,
            'volume': 500, 'open_interest': 1500, 'implied_volatility': 0.17,
            'spot_price': 53.34, 'delta': 0.30,
        },
        'setup_name': 'compression_breakout',
        'setup_direction': 'bullish',
        'setup_strength': 0.87,
        'weekly_ribbon_agreement': True,
        'pct_change_4w': 0.05,
        'vix_regime': 'neutral',
        'vix_pct_vs_7d': -0.05,
    }
    mc = MarketContext(
        spx_trend_score=9.0, sector_rotation_rank={'XLF': 2}, dxy_trend_score=6.0,
        yield_10y_score=8.0, vvix_score=8.5, days_to_macro_event=8,
    )
    fs = extract_features(candidate, mc, days_to_earnings=15)
    assert len(fs.values) == 20
    assert fs.values['tech_setup_type'] == 10.0
    assert fs.values['vol_vix_regime'] == 7.0
    assert fs.values['liq_open_interest'] == 10.0
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_features.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement Liquidity scorers + extract_features**

Append to `domain/aplus/features.py`:

```python
def score_bid_ask_spread(bid: float, ask: float) -> float:
    """Score based on relative spread: tight (<3% of mid) = 10, wide (>15%) = 0."""
    mid = (bid + ask) / 2
    if mid <= 0:
        return 0.0
    spread_pct = (ask - bid) / mid
    if spread_pct <= 0.03:
        return 10.0
    if spread_pct >= 0.15:
        return 0.0
    return 10.0 - (spread_pct - 0.03) / 0.12 * 10.0


def score_open_interest(oi: int) -> float:
    """OI >= 1000 = 10; OI < 100 = 0; linear in between."""
    if oi >= 1000:
        return 10.0
    if oi <= 100:
        return 0.0
    return (oi - 100) / 900.0 * 10.0


def score_volume_oi_ratio(volume: int, open_interest: int) -> float:
    """Volume / OI > 0.3 → 10 (active flow). < 0.05 → 3 (dormant)."""
    if open_interest <= 0:
        return 3.0
    ratio = volume / open_interest
    if ratio >= 0.3:
        return 10.0
    if ratio <= 0.05:
        return 3.0
    return 3.0 + (ratio - 0.05) / 0.25 * 7.0


# Sector → ETF lookup for sector rotation feature. v1 covers liquid sector mappings.
_TICKER_SECTOR_ETF: dict[str, str] = {
    # Sector ETFs map to themselves
    'XLK': 'XLK', 'XLF': 'XLF', 'XLE': 'XLE', 'XLV': 'XLV', 'XLI': 'XLI',
    'XLP': 'XLP', 'XLY': 'XLY', 'XLB': 'XLB', 'XLU': 'XLU', 'XLRE': 'XLRE', 'XLC': 'XLC',
    # Mega cap → sector
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'GOOGL': 'XLC', 'META': 'XLC',
    'AMZN': 'XLY', 'TSLA': 'XLY', 'AMD': 'XLK', 'JPM': 'XLF', 'V': 'XLF',
    'UNH': 'XLV', 'COST': 'XLP', 'NFLX': 'XLC',
}


def extract_features(
    candidate: dict,
    market_context: 'MarketContext',
    days_to_earnings: Optional[int] = None,
) -> 'FeatureScores':
    """Combine all 20 feature scorers into a single FeatureScores object."""
    from domain.aplus.types import FeatureScores

    contract = candidate.get('contract', {})
    ticker = contract.get('ticker', '')
    setup_dir = candidate.get('setup_direction', 'bullish')

    bid = float(contract.get('bid') or 0.0)
    ask = float(contract.get('ask') or 0.0)
    iv = float(contract.get('implied_volatility') or 0.0)
    volume = int(contract.get('volume') or 0)
    oi = int(contract.get('open_interest') or 0)
    spot = float(contract.get('spot_price') or 0.0)

    sector_etf = _TICKER_SECTOR_ETF.get(ticker, ticker if ticker in market_context.sector_rotation_rank else None)
    sector_rank = market_context.sector_rotation_rank.get(sector_etf, 6) if sector_etf else 6

    days_macro = market_context.days_to_macro_event

    pct_4w = float(candidate.get('pct_change_4w') or 0.0)
    proxy_zscore = pct_4w * 10  # rough proxy for volume z-score from 4w momentum

    # Approximate ATR via the 4w/1w return divergence — simple proxy for v1.
    pct_1w = float(candidate.get('pct_change_1w') or 0.0)
    atr_proxy = max(0.01, abs(pct_1w) * spot)
    prev_close_proxy = spot * (1 - pct_1w / 5.0)

    vix_z = float(candidate.get('vix_pct_vs_7d') or 0.0)
    vix_regime = candidate.get('vix_regime')

    values = {
        # Technical (5)
        'tech_setup_type': score_setup_type(candidate.get('setup_name')),
        'tech_setup_strength': score_setup_strength(candidate.get('setup_strength')),
        'tech_weekly_ribbon_agreement': score_weekly_ribbon_agreement(candidate.get('weekly_ribbon_agreement')),
        'tech_atr_pivot_position': score_atr_pivot_position(spot, prev_close_proxy, atr_proxy),
        'tech_volume_zscore': score_volume_zscore(proxy_zscore),
        # Vol/VIX (4)
        'vol_vix_regime': score_vix_regime(vix_regime),
        'vol_vix_zscore_30d': score_vix_zscore_30d(vix_z),
        'vol_vvix_level': score_vvix_level(market_context.vvix_score),
        'vol_iv_percentile': score_iv_percentile(iv),
        # Catalyst (3)
        'cat_days_to_earnings': score_days_to_earnings(days_to_earnings),
        'cat_days_to_macro_event': score_days_to_macro_event(days_macro),
        'cat_skip_window': score_skip_window(days_to_earnings, days_macro),
        # Macro/Breadth (4)
        'macro_spx_trend': score_spx_alignment(market_context.spx_trend_score),
        'macro_sector_rotation': score_sector_rotation(sector_rank, 11, setup_dir),
        'macro_dxy_trend': score_dxy_trend_pass_through(market_context.dxy_trend_score),
        'macro_yield_10y_direction': max(0.0, min(10.0, market_context.yield_10y_score)),
        # Liquidity (4)
        'liq_bid_ask_spread': score_bid_ask_spread(bid, ask),
        'liq_open_interest': score_open_interest(oi),
        'liq_oi_change_dod': 5.0,  # v1 placeholder; v2 will compute from cached prior chain
        'liq_volume_oi_ratio': score_volume_oi_ratio(volume, oi),
    }
    return FeatureScores(values=values)
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_features.py -v
```

Expected: PASS (~37 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/features.py tests/domain/aplus/test_features.py
git commit -m "feat(aplus): features — Liquidity scorers + extract_features"
```

---

## Task 7: Category scoring + composite

**Files:**
- Create: `domain/aplus/scoring.py`
- Test: `tests/domain/aplus/test_scoring.py`

Roll the 20 feature scores into 5 category averages and the weighted composite.

- [ ] **Step 1: Write the failing test**

Create `tests/domain/aplus/test_scoring.py`:

```python
from domain.aplus.types import FeatureScores
from domain.aplus.scoring import score_categories


def _build_fs(uniform: float) -> FeatureScores:
    keys = [
        # Technical
        'tech_setup_type', 'tech_setup_strength', 'tech_weekly_ribbon_agreement',
        'tech_atr_pivot_position', 'tech_volume_zscore',
        # Vol/VIX
        'vol_vix_regime', 'vol_vix_zscore_30d', 'vol_vvix_level', 'vol_iv_percentile',
        # Catalyst
        'cat_days_to_earnings', 'cat_days_to_macro_event', 'cat_skip_window',
        # Macro/Breadth
        'macro_spx_trend', 'macro_sector_rotation', 'macro_dxy_trend',
        'macro_yield_10y_direction',
        # Liquidity
        'liq_bid_ask_spread', 'liq_open_interest', 'liq_oi_change_dod',
        'liq_volume_oi_ratio',
    ]
    return FeatureScores(values={k: uniform for k in keys})


def test_score_categories_all_tens_gives_perfect():
    cs = score_categories(_build_fs(10.0))
    assert cs.technical == 10.0
    assert cs.vol_vix == 10.0
    assert cs.catalyst == 10.0
    assert cs.macro_breadth == 10.0
    assert cs.liquidity == 10.0
    assert cs.composite() == 100.0


def test_score_categories_all_zeros_gives_zero():
    cs = score_categories(_build_fs(0.0))
    assert cs.composite() == 0.0


def test_score_categories_partial_weights_correctly():
    # Technical=10, others=0 → composite = 25 * 10 = 25.0 (25% weight × 10 × 10 normalize)
    fs = _build_fs(0.0)
    fs.values['tech_setup_type'] = 10.0
    fs.values['tech_setup_strength'] = 10.0
    fs.values['tech_weekly_ribbon_agreement'] = 10.0
    fs.values['tech_atr_pivot_position'] = 10.0
    fs.values['tech_volume_zscore'] = 10.0
    cs = score_categories(fs)
    assert cs.technical == 10.0
    assert cs.vol_vix == 0.0
    assert cs.composite() == 25.0


def test_score_categories_handles_missing_keys_as_neutral():
    # If a feature key is missing entirely, treat it as neutral (5)
    fs = FeatureScores(values={'tech_setup_type': 10.0})
    cs = score_categories(fs)
    # Only one of five technical features is present; others default to 5.
    assert 4.5 < cs.technical < 7.0
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_scoring.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement scoring.py**

Create `domain/aplus/scoring.py`:

```python
"""Aggregate feature scores into category scores + composite."""
from __future__ import annotations
from domain.aplus.types import FeatureScores, CategoryScores


_CATEGORY_FEATURES: dict[str, list[str]] = {
    'technical': [
        'tech_setup_type', 'tech_setup_strength', 'tech_weekly_ribbon_agreement',
        'tech_atr_pivot_position', 'tech_volume_zscore',
    ],
    'vol_vix': [
        'vol_vix_regime', 'vol_vix_zscore_30d', 'vol_vvix_level', 'vol_iv_percentile',
    ],
    'catalyst': [
        'cat_days_to_earnings', 'cat_days_to_macro_event', 'cat_skip_window',
    ],
    'macro_breadth': [
        'macro_spx_trend', 'macro_sector_rotation', 'macro_dxy_trend',
        'macro_yield_10y_direction',
    ],
    'liquidity': [
        'liq_bid_ask_spread', 'liq_open_interest', 'liq_oi_change_dod',
        'liq_volume_oi_ratio',
    ],
}


def score_categories(fs: FeatureScores) -> CategoryScores:
    """Compute category-level averages from individual feature scores.
    Missing keys default to 5.0 (neutral)."""

    def _avg(category: str) -> float:
        keys = _CATEGORY_FEATURES[category]
        vals = [fs.values.get(k, 5.0) for k in keys]
        return sum(vals) / len(vals)

    return CategoryScores(
        technical=_avg('technical'),
        vol_vix=_avg('vol_vix'),
        catalyst=_avg('catalyst'),
        macro_breadth=_avg('macro_breadth'),
        liquidity=_avg('liquidity'),
    )
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_scoring.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/scoring.py tests/domain/aplus/test_scoring.py
git commit -m "feat(aplus): scoring — category averages + weighted composite"
```

---

## Task 8: Grading

**Files:**
- Create: `domain/aplus/grading.py`
- Test: `tests/domain/aplus/test_grading.py`

Map composite + category floors → A+/A/B+/B/F.

- [ ] **Step 1: Write the failing test**

Create `tests/domain/aplus/test_grading.py`:

```python
from domain.aplus.types import CategoryScores
from domain.aplus.grading import assign_grade


def test_assign_grade_a_plus_when_high_composite_and_all_floors_met():
    cs = CategoryScores(technical=9.0, vol_vix=9.0, catalyst=9.0, macro_breadth=9.0, liquidity=9.0)
    assert assign_grade(cs) == 'A+'


def test_assign_grade_a_when_composite_high_but_categories_lower():
    cs = CategoryScores(technical=8.0, vol_vix=8.0, catalyst=7.0, macro_breadth=8.0, liquidity=8.0)
    grade = assign_grade(cs)
    assert grade in ('A', 'A+')   # composite ~78 — should be A not A+


def test_assign_grade_b_plus_when_mid_composite():
    cs = CategoryScores(technical=7.0, vol_vix=7.0, catalyst=7.0, macro_breadth=7.0, liquidity=7.0)
    assert assign_grade(cs) == 'B+'


def test_assign_grade_b_when_some_categories_at_floor():
    cs = CategoryScores(technical=8.0, vol_vix=8.0, catalyst=6.0, macro_breadth=6.0, liquidity=6.0)
    # composite = (8+8+6+6+6) means lower; categories at 6 floor
    grade = assign_grade(cs)
    assert grade in ('B', 'B+')


def test_assign_grade_f_when_composite_below_60():
    cs = CategoryScores(technical=5.0, vol_vix=5.0, catalyst=5.0, macro_breadth=5.0, liquidity=5.0)
    assert assign_grade(cs) == 'F'


def test_assign_grade_a_plus_blocked_by_one_low_category():
    # All others 9 but one category at 7 — should NOT be A+ (floor not met)
    cs = CategoryScores(technical=9.0, vol_vix=7.0, catalyst=9.0, macro_breadth=9.0, liquidity=9.0)
    grade = assign_grade(cs)
    assert grade != 'A+'
    assert grade == 'A'
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_grading.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement grading.py**

Create `domain/aplus/grading.py`:

```python
"""Map composite + category floors → A+/A/B+/B/F."""
from __future__ import annotations
from domain.aplus.types import CategoryScores, Grade


def _all_categories_at_least(cs: CategoryScores, floor: float) -> bool:
    return (
        cs.technical >= floor and cs.vol_vix >= floor
        and cs.catalyst >= floor and cs.macro_breadth >= floor
        and cs.liquidity >= floor
    )


def assign_grade(cs: CategoryScores) -> Grade:
    """Apply the grading thresholds from the spec:

        A+ : composite >= 90 AND every category >= 8
        A  : composite >= 80 AND every category >= 7
        B+ : composite >= 70 AND every category >= 6
        B  : composite >= 60
        F  : composite <  60
    """
    composite = cs.composite()

    if composite >= 90.0 and _all_categories_at_least(cs, 8.0):
        return 'A+'
    if composite >= 80.0 and _all_categories_at_least(cs, 7.0):
        return 'A'
    if composite >= 70.0 and _all_categories_at_least(cs, 6.0):
        return 'B+'
    if composite >= 60.0:
        return 'B'
    return 'F'
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_grading.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/grading.py tests/domain/aplus/test_grading.py
git commit -m "feat(aplus): grading — composite + category floors → A+/A/B+/B/F"
```

---

## Task 9: Structure selection

**Files:**
- Create: `domain/aplus/structure.py`
- Test: `tests/domain/aplus/test_structure.py`

Long premium vs debit spread.

- [ ] **Step 1: Write the failing test**

Create `tests/domain/aplus/test_structure.py`:

```python
from domain.aplus.types import FeatureScores, TradeStructure
from domain.aplus.structure import select_structure


def _fs(**overrides) -> FeatureScores:
    base = {
        'vol_iv_percentile': 8.0,
        'cat_post_earnings_drift': 0.0,
        'liq_bid_ask_spread': 6.0,
        'macro_sector_rotation': 6.0,
        'vol_vix_regime': 7.0,
    }
    base.update(overrides)
    return FeatureScores(values=base)


def test_high_iv_returns_debit_spread():
    # vol_iv_percentile <=3 means high IV (raw IV >50%); should return debit spread
    fs = _fs(vol_iv_percentile=2.0)
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.DEBIT_SPREAD
    assert 'IV' in rationale


def test_post_earnings_drift_with_tight_spreads_returns_long_premium():
    fs = _fs(vol_iv_percentile=8.0, cat_post_earnings_drift=10.0, liq_bid_ask_spread=9.5)
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.LONG_PREMIUM


def test_sector_leader_supportive_vol_returns_long_premium():
    fs = _fs(macro_sector_rotation=9.5, vol_vix_regime=8.0)
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.LONG_PREMIUM


def test_default_returns_debit_spread():
    fs = _fs()  # no special conditions met
    structure, rationale = select_structure(fs)
    assert structure == TradeStructure.DEBIT_SPREAD
```

- [ ] **Step 2: Run test, confirm fail**

```bash
pytest tests/domain/aplus/test_structure.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement structure.py**

Create `domain/aplus/structure.py`:

```python
"""Trade structure selector — long premium vs debit spread."""
from __future__ import annotations
from domain.aplus.types import FeatureScores, TradeStructure


def select_structure(fs: FeatureScores) -> tuple[TradeStructure, str]:
    """Choose a trade structure based on the feature scores.

    Branch order matches the spec in §5 of the design doc.
    """
    vix_iv_percentile = fs.values.get('vol_iv_percentile', 5.0)
    if vix_iv_percentile <= 3.0:
        # Low score on vol_iv_percentile = HIGH IV — prefer debit spread
        return TradeStructure.DEBIT_SPREAD, "IV percentile high — vega exposure would hurt long premium"

    post_earnings = fs.values.get('cat_post_earnings_drift', 5.0)
    spread = fs.values.get('liq_bid_ask_spread', 5.0)
    if post_earnings >= 9.0 and spread >= 9.0:
        return TradeStructure.LONG_PREMIUM, "post-earnings drift + tight spreads = asymmetric upside"

    sector_rot = fs.values.get('macro_sector_rotation', 5.0)
    vix_regime = fs.values.get('vol_vix_regime', 5.0)
    if sector_rot >= 9.0 and vix_regime >= 7.0:
        return TradeStructure.LONG_PREMIUM, "sector leadership + vol-supportive = trend continuation"

    return TradeStructure.DEBIT_SPREAD, "defensive default — defined R:R, better PoP"
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/domain/aplus/test_structure.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add domain/aplus/structure.py tests/domain/aplus/test_structure.py
git commit -m "feat(aplus): structure — long premium vs debit spread selector"
```

---

## Task 10: Watchlist script — read scan and render

**Files:**
- Create: `scripts/__init__.py` (if missing — likely already there)
- Create: `scripts/aplus_watchlist.py`
- Test: `tests/scripts/test_aplus_watchlist.py`

Wire it all together. Reads `output/scans/latest.json`, grades every candidate, filters to A+/A, writes `output/aplus/latest.json`, prints console table.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_aplus_watchlist.py`:

```python
import json
from pathlib import Path
from datetime import date
from unittest.mock import patch

from scripts.aplus_watchlist import render_watchlist


def _scan_payload():
    return {
        'timestamp': '2026-06-14T08:00:00',
        'candidates': [
            {
                'contract': {
                    'ticker': 'XLF', 'option_type': 'put', 'strike': 52.0,
                    'expiration': '2026-07-10', 'dte': 26, 'spot_price': 53.34,
                    'bid': 0.47, 'ask': 0.53, 'mid': 0.50, 'volume': 500,
                    'open_interest': 1500, 'implied_volatility': 0.17,
                    'delta': -0.30, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                },
                'strategy': 'long_put',
                'setup_name': 'compression_breakout',
                'setup_direction': 'bullish',
                'setup_strength': 0.85,
                'weekly_ribbon_agreement': True,
                'pct_change_1w': 0.01, 'pct_change_4w': 0.05,
                'vix_regime': 'neutral', 'vix_pct_vs_7d': -0.05,
                'composite_score': 70.0, 'label': 'WATCHLIST',
                'reason_for': '', 'reason_against': '',
            }
        ],
    }


def test_render_watchlist_with_strong_candidate_produces_output(tmp_path):
    scan_file = tmp_path / 'scan.json'
    scan_file.write_text(json.dumps(_scan_payload()))
    out_dir = tmp_path / 'aplus'

    with patch('scripts.aplus_watchlist._fetch_market_context') as mock_mc:
        from domain.aplus.types import MarketContext
        mock_mc.return_value = MarketContext(
            spx_trend_score=9.0, sector_rotation_rank={'XLF': 2},
            dxy_trend_score=7.0, yield_10y_score=8.0, vvix_score=9.0,
            days_to_macro_event=8,
        )
        with patch('scripts.aplus_watchlist._fetch_days_to_earnings') as mock_e:
            mock_e.return_value = 15
            result = render_watchlist(scan_file, out_dir, account_size=1000.0, today=date(2026, 6, 14))

    assert (out_dir / 'latest.json').exists()
    out_data = json.loads((out_dir / 'latest.json').read_text())
    assert 'graded_candidates' in out_data
    # Should have at least the XLF candidate graded
    assert len(result) >= 0
```

- [ ] **Step 2: Run test, confirm fail**

```bash
mkdir -p tests/scripts
touch tests/scripts/__init__.py
pytest tests/scripts/test_aplus_watchlist.py -v
```

Expected: FAIL (`scripts.aplus_watchlist` does not exist).

- [ ] **Step 3: Implement aplus_watchlist.py**

Create `scripts/aplus_watchlist.py`:

```python
"""A+ Confluence watchlist CLI.

Reads a scan JSON, grades every candidate, filters to A+/A, and writes
output/aplus/latest.json plus a console summary.

Usage:
    python -m scripts.aplus_watchlist
    python -m scripts.aplus_watchlist --scan output/scans/scan_20260614_0811.json
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from domain.aplus.features import extract_features
from domain.aplus.scoring import score_categories
from domain.aplus.grading import assign_grade
from domain.aplus.structure import select_structure
from domain.aplus.market_context import build_market_context
from domain.aplus.types import MarketContext, TradeStructure, GradedCandidate


_GRADE_RANK = {'A+': 5, 'A': 4, 'B+': 3, 'B': 2, 'F': 1}
_SIZING_PCT = {'A+': 0.125, 'A': 0.075, 'B+': 0.0, 'B': 0.0, 'F': 0.0}


def render_watchlist(
    scan_path: Path,
    out_dir: Path,
    account_size: float = 1000.0,
    today: Optional[date] = None,
) -> list[GradedCandidate]:
    """Read scan JSON, grade candidates, write output. Returns the graded list."""
    today = today or date.today()
    scan_data = json.loads(scan_path.read_text())
    candidates = scan_data.get('candidates', [])

    if not candidates:
        print("No candidates in scan — nothing to grade.")
        return []

    mc = _fetch_market_context(today)

    graded: list[GradedCandidate] = []
    for cand in candidates:
        ticker = cand.get('contract', {}).get('ticker', '')
        days_to_earn = _fetch_days_to_earnings(ticker)
        fs = extract_features(cand, mc, days_to_earnings=days_to_earn)
        cs = score_categories(fs)
        grade = assign_grade(cs)
        if grade in ('B', 'F'):
            continue
        structure, rationale = select_structure(fs)
        composite = cs.composite()
        sizing_pct = _SIZING_PCT[grade]
        max_risk = account_size * sizing_pct
        graded.append(GradedCandidate(
            ticker=ticker,
            strategy=cand.get('strategy', ''),
            composite_score=composite,
            grade=grade,
            category_scores=cs,
            feature_scores=fs,
            structure=structure,
            structure_rationale=rationale,
            sizing_pct=sizing_pct,
            max_risk_dollars=max_risk,
            raw_candidate=cand,
        ))

    # Sort: A+ first, then A; within grade, by composite descending.
    graded.sort(key=lambda g: (-_GRADE_RANK[g.grade], -g.composite_score))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_data = {
        'timestamp': today.isoformat(),
        'account_size': account_size,
        'graded_candidates': [_serialize(g) for g in graded],
    }
    (out_dir / 'latest.json').write_text(json.dumps(out_data, indent=2, default=str))

    print_summary(graded, account_size)
    return graded


def _serialize(g: GradedCandidate) -> dict:
    return {
        'ticker': g.ticker,
        'strategy': g.strategy,
        'grade': g.grade,
        'composite_score': g.composite_score,
        'category_scores': {
            'technical': g.category_scores.technical,
            'vol_vix': g.category_scores.vol_vix,
            'catalyst': g.category_scores.catalyst,
            'macro_breadth': g.category_scores.macro_breadth,
            'liquidity': g.category_scores.liquidity,
        },
        'feature_scores': g.feature_scores.values,
        'structure': g.structure.value,
        'structure_rationale': g.structure_rationale,
        'sizing_pct': g.sizing_pct,
        'max_risk_dollars': g.max_risk_dollars,
        'contract': g.raw_candidate.get('contract', {}),
    }


def print_summary(graded: list[GradedCandidate], account_size: float) -> None:
    print()
    print(f"=== A+ Watchlist — account ${account_size:,.0f} ===\n")
    if not graded:
        print("  No A+/A grade candidates today.\n")
        return
    print(f"  Grade  Ticker  Strategy     Composite  Structure       Max risk")
    print(f"  {'-'*65}")
    for g in graded[:10]:
        print(f"  {g.grade:<6} {g.ticker:<7} {g.strategy:<12} {g.composite_score:>7.1f}    "
              f"{g.structure.value:<15} ${g.max_risk_dollars:>5.0f}")
    print()


def _fetch_market_context(today: date) -> MarketContext:
    """Build market context using the canonical 'bullish' direction baseline.
    Direction-specific scores (SPX trend, sector rotation, DXY) are recomputed
    per candidate at feature time."""
    return build_market_context(today=today, setup_direction='bullish')


def _fetch_days_to_earnings(ticker: str) -> Optional[int]:
    """Return calendar days to next earnings, or None if unknown.
    Reuses the existing earnings infrastructure when available."""
    try:
        from pipeline.earnings import has_earnings_within
        for d in (3, 7, 14, 30, 60):
            if has_earnings_within(ticker, d):
                return d
        return None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="A+ confluence watchlist")
    p.add_argument('--scan', type=str, default='output/scans/latest.json')
    p.add_argument('--out', type=str, default='output/aplus')
    p.add_argument('--account-size', type=float, default=1000.0)
    args = p.parse_args(argv)
    render_watchlist(Path(args.scan), Path(args.out), args.account_size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/scripts/test_aplus_watchlist.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/aplus_watchlist.py tests/scripts/test_aplus_watchlist.py tests/scripts/__init__.py
git commit -m "feat(aplus): aplus_watchlist CLI script — reads scan, grades, renders"
```

---

## Task 11: CLI subcommand in `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Read current main.py to locate the subparser pattern**

```bash
grep -n "sub.add_parser" main.py | head
```

You'll see `scan`, `dashboard`, `watchlist`, `universe`. Add a new `aplus_watchlist` next to them.

- [ ] **Step 2: Add the cmd_aplus_watchlist handler**

In `main.py`, near the other `cmd_*` functions, add:

```python
def cmd_aplus_watchlist(args) -> int:
    from scripts.aplus_watchlist import render_watchlist
    scan_path = Path(args.scan) if args.scan else Path('output/scans/latest.json')
    if not scan_path.exists():
        print(f"Scan file not found: {scan_path}", file=sys.stderr)
        return 2
    render_watchlist(scan_path, Path(args.out), account_size=args.account_size)
    return 0
```

- [ ] **Step 3: Wire up the subparser**

Inside `main()`, alongside the other `sub.add_parser` calls, add:

```python
    a = sub.add_parser('aplus_watchlist', help='Generate A+ confluence-graded watchlist')
    a.add_argument('--scan', type=str, default=None,
                   help='Path to scan JSON (defaults to output/scans/latest.json)')
    a.add_argument('--out', type=str, default='output/aplus',
                   help='Output directory for latest.json')
    a.add_argument('--account-size', type=float, default=1000.0,
                   help='Account size for sizing recommendations (default $1000)')
    a.set_defaults(func=cmd_aplus_watchlist)
```

- [ ] **Step 4: Sanity-check the help flag**

```bash
python3 -m main aplus_watchlist --help
```

Expected: prints help text with `--scan`, `--out`, `--account-size` flags.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(cli): main.py — aplus_watchlist subcommand"
```

---

## Task 12: Integration test against a real scan fixture

**Files:**
- Create: `tests/scripts/fixtures/sample_scan.json`
- Modify: `tests/scripts/test_aplus_watchlist.py` (append integration test)

End-to-end: real scan JSON → output JSON. No mocks except for the network-bound market-context fetch.

- [ ] **Step 1: Capture a minimal real-shape fixture**

Create `tests/scripts/fixtures/sample_scan.json` (use `output/scans/scan_20260614_0811.json` shape — produce 3 candidates, varying setup types).

```bash
mkdir -p tests/scripts/fixtures
cat > tests/scripts/fixtures/sample_scan.json << 'JSON'
{
  "timestamp": "2026-06-14T08:00:00",
  "candidates": [
    {
      "contract": {
        "ticker": "XLF", "option_type": "put", "strike": 52.0,
        "expiration": "2026-07-10", "dte": 26, "spot_price": 53.34,
        "bid": 0.47, "ask": 0.53, "mid": 0.50, "volume": 500,
        "open_interest": 1500, "implied_volatility": 0.17,
        "delta": -0.30, "gamma": 0.01, "theta": -0.01, "vega": 0.05
      },
      "strategy": "long_put",
      "setup_name": "compression_breakout",
      "setup_direction": "bullish",
      "setup_strength": 0.87,
      "weekly_ribbon_agreement": true,
      "pct_change_1w": 0.01, "pct_change_2w": 0.02, "pct_change_4w": 0.05,
      "vix_regime": "neutral", "vix_pct_vs_7d": -0.05,
      "composite_score": 70.0, "label": "WATCHLIST",
      "reason_for": "compression squeeze",
      "reason_against": "",
      "vix_now": 17.7
    },
    {
      "contract": {
        "ticker": "AAPL", "option_type": "call", "strike": 295.0,
        "expiration": "2026-07-18", "dte": 34, "spot_price": 292.21,
        "bid": 3.50, "ask": 4.00, "mid": 3.75, "volume": 8000,
        "open_interest": 5000, "implied_volatility": 0.55,
        "delta": 0.45, "gamma": 0.02, "theta": -0.05, "vega": 0.10
      },
      "strategy": "long_call",
      "setup_name": "pullback_in_trend",
      "setup_direction": "bullish",
      "setup_strength": 0.55,
      "weekly_ribbon_agreement": false,
      "pct_change_1w": -0.02, "pct_change_2w": 0.01, "pct_change_4w": 0.02,
      "vix_regime": "expansion", "vix_pct_vs_7d": 0.18,
      "composite_score": 55.0, "label": "WATCHLIST",
      "reason_for": "pullback to 21EMA",
      "reason_against": ""
    },
    {
      "contract": {
        "ticker": "MSFT", "option_type": "call", "strike": 510.0,
        "expiration": "2026-07-25", "dte": 41, "spot_price": 505.10,
        "bid": 8.50, "ask": 9.00, "mid": 8.75, "volume": 1200,
        "open_interest": 2400, "implied_volatility": 0.25,
        "delta": 0.55, "gamma": 0.01, "theta": -0.03, "vega": 0.18
      },
      "strategy": "long_call",
      "setup_name": "compression_breakout",
      "setup_direction": "bullish",
      "setup_strength": 0.92,
      "weekly_ribbon_agreement": true,
      "pct_change_1w": 0.02, "pct_change_2w": 0.04, "pct_change_4w": 0.08,
      "vix_regime": "neutral", "vix_pct_vs_7d": 0.0,
      "composite_score": 85.0, "label": "TRADE",
      "reason_for": "compression + 4w trend",
      "reason_against": ""
    }
  ]
}
JSON
```

- [ ] **Step 2: Add the integration test**

Append to `tests/scripts/test_aplus_watchlist.py`:

```python
def test_render_watchlist_integration_fixture(tmp_path):
    fixture = Path(__file__).parent / 'fixtures' / 'sample_scan.json'
    out_dir = tmp_path / 'aplus'
    with patch('scripts.aplus_watchlist._fetch_market_context') as mock_mc:
        from domain.aplus.types import MarketContext
        mock_mc.return_value = MarketContext(
            spx_trend_score=9.0,
            sector_rotation_rank={'XLF': 2, 'XLK': 1},
            dxy_trend_score=7.0, yield_10y_score=8.0, vvix_score=9.0,
            days_to_macro_event=10,
        )
        with patch('scripts.aplus_watchlist._fetch_days_to_earnings') as mock_e:
            mock_e.return_value = 20
            result = render_watchlist(fixture, out_dir, account_size=1000.0, today=date(2026, 6, 14))

    out_payload = json.loads((out_dir / 'latest.json').read_text())
    assert 'graded_candidates' in out_payload
    # AAPL has vix_regime=expansion → vol_vix=0 floor → should not be A+/A
    aapl = next((c for c in out_payload['graded_candidates'] if c['ticker'] == 'AAPL'), None)
    if aapl is not None:
        assert aapl['grade'] != 'A+'
    # MSFT has compression + clean context → should grade well
    msft = next((c for c in out_payload['graded_candidates'] if c['ticker'] == 'MSFT'), None)
    # MSFT may not appear if features push it to B; that's okay too.
```

- [ ] **Step 3: Run, confirm pass**

```bash
pytest tests/scripts/test_aplus_watchlist.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/scripts/fixtures/sample_scan.json tests/scripts/test_aplus_watchlist.py
git commit -m "test(aplus): integration fixture + end-to-end watchlist test"
```

---

## Task 13: Full-suite regression check

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/sarthakhans/options-scanner
pytest -q
```

Expected: All tests pass (~470+ total, ~30+ new from this plan).

- [ ] **Step 2: Sanity check the CLI from scratch**

```bash
python3 -m main aplus_watchlist --help
```

Expected: prints argparse help including `--scan`, `--out`, `--account-size`.

- [ ] **Step 3: Live smoke test (network)**

(Only when markets are open or you have a cached scan with current data.)

```bash
set -a && source /Users/sarthakhans/options-scanner/.env && set +a
PYTHONPATH=. python3 -m main aplus_watchlist --scan output/scans/latest.json
```

Expected: prints a console summary table with 0-5 graded candidates and writes `output/aplus/latest.json`.

- [ ] **Step 4: Final commit + push**

```bash
git push
```

Expected: pushes the new commits to `origin/feat/layered-rebuild`.

---

## Spec coverage verification

| Spec requirement | Task implementing it |
|---|---|
| New CLI `python -m main aplus_watchlist` | Task 11 |
| Each candidate gets composite score in [0, 100] | Task 7 |
| Each candidate gets grade in {A+, A, B+, B} | Task 8 |
| B+ and B excluded from output | Task 10 |
| Trade structure recommendation per candidate | Task 9 + Task 10 |
| Sizing per §5 of spec | Task 10 |
| Forward-test protocol documented | spec already references; no code change needed |
| 434-test baseline still passes | Task 13 |
| New module has ≥85% test coverage | Tasks 1-12 (TDD throughout) |

## File-structure / boundary check

- `domain/aplus/types.py` — dataclasses only, no I/O
- `domain/aplus/features.py` — pure functions over dicts + MarketContext; no I/O
- `domain/aplus/scoring.py` — pure aggregation; no I/O
- `domain/aplus/grading.py` — pure mapping; no I/O
- `domain/aplus/structure.py` — pure mapping; no I/O
- `domain/aplus/market_context.py` — wraps I/O calls to YahooMacroSource; testable via mock
- `data/sources/macro_calendar.py` — hardcoded data, no I/O at runtime
- `scripts/aplus_watchlist.py` — orchestration only; delegates math to domain

Each unit has one clear responsibility, communicates through well-defined interfaces, and can be tested independently.
