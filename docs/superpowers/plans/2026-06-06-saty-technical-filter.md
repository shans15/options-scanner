# Saty Technical Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an EOD pre-filter stage that narrows the options-scan universe to tickers showing one of four Saty-style directional setups (compression breakout, pullback in trend, stage 2 breakout, failed breakdown reversal), and gates which long strategy is evaluated per surviving ticker based on the setup's direction.

**Architecture:** A new `domain/technical_signals.py` module provides four pure detectors operating on daily OHLCV. A new `pipeline/technical_filter.py` stage fetches history (cheap) before the option-chain pull (expensive) and drops tickers without signals. `pipeline/run_scan.py` is modified to call the filter and to map strategy direction → setup direction. `ScoredCandidate` gains three optional setup fields. The composite score weights are unchanged — filter membership encodes conviction, scoring still ranks on PoP / EV / regime.

**Tech Stack:** Python 3.11+, pandas, numpy, pytest, pytest-mock.

**Spec reference:** `docs/superpowers/specs/2026-06-06-saty-technical-filter-design.md`

---

## File manifest

**New files:**
- `domain/technical_signals.py` — TechnicalSetup, indicator helpers, 4 detectors, `detect_setups`
- `pipeline/technical_filter.py` — `filter_by_technicals` stage
- `tests/domain/test_technical_signals.py`
- `tests/pipeline/test_technical_filter.py`

**Modified files:**
- `data/sources/base.py` — add `fetch_price_history_ohlcv` abstract method
- `data/sources/yahooquery_source.py` — implement `fetch_price_history_ohlcv`
- `data/sources/yfinance_source.py` — implement `fetch_price_history_ohlcv`
- `data/sources/stooq_source.py` — implement `fetch_price_history_ohlcv`
- `pipeline/run_scan.py` — insert filter stage, replace direction gate, populate setup fields
- `engine/scorer.py` — extend `ScoredCandidate` with 3 optional fields
- `main.py` — `--no-technical-filter` flag
- `ui/exporter.py` — 3 new columns in CSV/JSON
- `ui/dashboard.py` — Setup column + sidebar filter widget
- `tests/pipeline/test_run_scan.py` — 2 regression cases
- `tests/data/test_sources.py` (if exists, else create) — tests for new OHLCV method

---

## Task 1: Extend DataSource with OHLCV fetch

**Files:**
- Modify: `data/sources/base.py`
- Modify: `data/sources/yahooquery_source.py`
- Modify: `data/sources/yfinance_source.py`
- Modify: `data/sources/stooq_source.py`
- Test: `tests/data/test_sources_ohlcv.py` (new)

### Why

Current `fetch_price_history` returns a `pd.Series` (close only). Detectors need a `pd.DataFrame` with columns `['open', 'high', 'low', 'close', 'volume']`. Lowercase, standardized across sources.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_sources_ohlcv.py`:

```python
import pandas as pd
import pytest

from data.sources.base import DataSource


class _StubSource(DataSource):
    def fetch_spot(self, ticker): return 0.0
    def fetch_price_history(self, ticker, lookback_days): return pd.Series(dtype=float)
    def fetch_option_chain(self, ticker): return []
    # Does not override fetch_price_history_ohlcv → should raise NotImplementedError


def test_datasource_requires_fetch_price_history_ohlcv():
    # Sources missing OHLCV impl should fail to instantiate (abstract)
    with pytest.raises(TypeError):
        _StubSource()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /Users/sarthakhans/options-scanner
pytest tests/data/test_sources_ohlcv.py::test_datasource_requires_fetch_price_history_ohlcv -v
```

Expected: FAIL — `_StubSource()` instantiates without error because no abstract method exists yet.

- [ ] **Step 3: Add abstract method to DataSource base**

Edit `data/sources/base.py`, append to the `DataSource` class:

```python
    @abstractmethod
    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Return DataFrame with lowercase columns: open, high, low, close, volume.
        Index is the date (ascending). Length up to lookback_days; may be less
        if the ticker has shorter history."""
```

- [ ] **Step 4: Re-run the test, confirm it passes**

```bash
pytest tests/data/test_sources_ohlcv.py::test_datasource_requires_fetch_price_history_ohlcv -v
```

Expected: PASS.

- [ ] **Step 5: Add concrete test for YahooQuery OHLCV (mocked)**

Append to `tests/data/test_sources_ohlcv.py`:

```python
from unittest.mock import patch, MagicMock
import numpy as np
from data.sources.yahooquery_source import YahooQuerySource


def test_yahooquery_ohlcv_returns_dataframe_with_required_columns():
    fake_df = pd.DataFrame({
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [99.0, 100.0],
        'close': [101.0, 102.0],
        'volume': [1_000_000, 1_100_000],
    }, index=pd.to_datetime(['2026-01-02', '2026-01-03']))

    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        instance = MagicMock()
        instance.history.return_value = fake_df
        MockTicker.return_value = instance
        out = YahooQuerySource().fetch_price_history_ohlcv('AAPL', 365)

    assert set(out.columns) >= {'open', 'high', 'low', 'close', 'volume'}
    assert len(out) == 2
    assert out['close'].iloc[-1] == 102.0
```

- [ ] **Step 6: Run new test, confirm it fails**

```bash
pytest tests/data/test_sources_ohlcv.py::test_yahooquery_ohlcv_returns_dataframe_with_required_columns -v
```

Expected: FAIL — `fetch_price_history_ohlcv` doesn't exist on YahooQuerySource.

- [ ] **Step 7: Implement YahooQuerySource.fetch_price_history_ohlcv**

Edit `data/sources/yahooquery_source.py`, append method to the class:

```python
    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        period = '1y' if lookback_days > 180 else '6mo'
        df = Ticker(ticker).history(period=period)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol', drop_level=True)
        cols = {c.lower(): c for c in df.columns}
        keep = ['open', 'high', 'low', 'close', 'volume']
        out = pd.DataFrame({k: df[cols[k]].astype(float) for k in keep if k in cols})
        return out.dropna()
```

- [ ] **Step 8: Re-run, confirm passes**

```bash
pytest tests/data/test_sources_ohlcv.py -v
```

Expected: both tests PASS.

- [ ] **Step 9: Implement yfinance OHLCV (Title-case → lowercase)**

Edit `data/sources/yfinance_source.py`, append method:

```python
    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        period = '1y' if lookback_days > 180 else '6mo'
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename)
        keep = ['open', 'high', 'low', 'close', 'volume']
        return df[[c for c in keep if c in df.columns]].astype(float).dropna()
```

- [ ] **Step 10: Implement stooq OHLCV**

Edit `data/sources/stooq_source.py`, append method:

```python
    def fetch_price_history_ohlcv(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        df = pd.read_csv(self._url(ticker))
        if df.empty:
            raise RuntimeError(f"stooq returned empty for {ticker}")
        rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename)
        keep = ['open', 'high', 'low', 'close', 'volume']
        out = df[[c for c in keep if c in df.columns]].astype(float).tail(lookback_days)
        return out.reset_index(drop=True).dropna()
```

- [ ] **Step 11: Run full test suite to ensure no regression**

```bash
pytest -q
```

Expected: all existing tests still pass, plus the 2 new ones.

- [ ] **Step 12: Commit**

```bash
git add data/sources/base.py data/sources/yahooquery_source.py data/sources/yfinance_source.py data/sources/stooq_source.py tests/data/test_sources_ohlcv.py
git commit -m "feat(data): add fetch_price_history_ohlcv to DataSource interface

Returns a DataFrame with lowercase open/high/low/close/volume columns.
Needed by the upcoming Saty technical-signal detectors."
```

---

## Task 2: TechnicalSetup dataclass + indicator helpers

**Files:**
- Create: `domain/technical_signals.py`
- Create: `tests/domain/test_technical_signals.py`

### Why

Shared math used by all 4 detectors. EMA, ATR(14) Wilder's, Phase Oscillator, and PO compression percentile rank. Building these first makes detector code trivial.

- [ ] **Step 1: Write the failing test for TechnicalSetup dataclass**

Create `tests/domain/test_technical_signals.py`:

```python
import numpy as np
import pandas as pd
import pytest

from domain.technical_signals import TechnicalSetup


def test_technical_setup_is_frozen_with_required_fields():
    s = TechnicalSetup(
        setup_name='compression_breakout',
        direction='bullish',
        strength=0.85,
        notes='PO squeeze rank 0.08, ribbon stacked',
    )
    assert s.setup_name == 'compression_breakout'
    assert s.direction == 'bullish'
    assert 0.0 <= s.strength <= 1.0
    with pytest.raises((AttributeError, Exception)):
        s.strength = 0.5  # frozen → cannot mutate
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
pytest tests/domain/test_technical_signals.py::test_technical_setup_is_frozen_with_required_fields -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Create domain/technical_signals.py with the dataclass**

Create `domain/technical_signals.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd


SetupName = Literal[
    'compression_breakout',
    'pullback_in_trend',
    'stage_2_breakout',
    'failed_breakdown_reversal',
]
Direction = Literal['bullish', 'bearish']


@dataclass(frozen=True)
class TechnicalSetup:
    setup_name: SetupName
    direction: Direction
    strength: float
    notes: str
```

- [ ] **Step 4: Re-run, confirm passes**

```bash
pytest tests/domain/test_technical_signals.py -v
```

Expected: PASS.

- [ ] **Step 5: Write tests for the helpers**

Append to `tests/domain/test_technical_signals.py`:

```python
from domain.technical_signals import (
    _emas, _atr14, _phase_oscillator, _po_bandwidth_percentile,
)


def _synth_ohlcv(close_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        'open':   close_series.shift(1).fillna(close_series.iloc[0]),
        'high':   close_series * 1.005,
        'low':    close_series * 0.995,
        'close':  close_series,
        'volume': pd.Series(1_000_000, index=close_series.index),
    })


def test_emas_returns_dict_with_expected_lengths():
    close = pd.Series(np.linspace(100, 110, 250))
    out = _emas(close)
    assert set(out.keys()) == {8, 13, 21, 48, 200}
    for n, ema in out.items():
        assert len(ema) == len(close)
        # last EMA value should be near the last close for a linear series
        assert abs(ema.iloc[-1] - close.iloc[-1]) < 5.0


def test_atr14_is_positive_and_smoothed():
    close = pd.Series(np.linspace(100, 120, 250))
    df = _synth_ohlcv(close)
    atr = _atr14(df['high'], df['low'], df['close'])
    assert (atr.dropna() > 0).all()
    assert len(atr) == len(close)


def test_phase_oscillator_zero_when_close_equals_ema21():
    close = pd.Series([100.0] * 250)
    df = _synth_ohlcv(close)
    emas = _emas(df['close'])
    atr = _atr14(df['high'], df['low'], df['close'])
    po = _phase_oscillator(df['close'], emas[21], atr)
    # On a flat series with positive ATR (from 0.5% wicks), PO should hover near 0
    assert abs(po.iloc[-1]) < 0.5


def test_po_bandwidth_percentile_is_in_unit_range():
    np.random.seed(1)
    po = pd.Series(np.random.normal(0, 1, 300))
    pct = _po_bandwidth_percentile(po)
    # After enough warmup, values should be in [0, 1]
    tail = pct.dropna().tail(50)
    assert ((tail >= 0) & (tail <= 1)).all()
```

- [ ] **Step 6: Run helper tests, confirm they fail**

```bash
pytest tests/domain/test_technical_signals.py -v
```

Expected: FAIL — helpers not implemented.

- [ ] **Step 7: Implement the helpers**

Append to `domain/technical_signals.py`:

```python
_EMA_LENGTHS = (8, 13, 21, 48, 200)
_ATR_PERIOD = 14
_PO_BBW_INNER = 20         # rolling std window on PO
_PO_BBW_OUTER = 126        # percentile rank window (~6 months)


def _emas(close: pd.Series) -> dict[int, pd.Series]:
    return {n: close.ewm(span=n, adjust=False).mean() for n in _EMA_LENGTHS}


def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing = EMA with alpha = 1/period
    return tr.ewm(alpha=1 / _ATR_PERIOD, adjust=False).mean()


def _phase_oscillator(close: pd.Series, ema21: pd.Series, atr14: pd.Series) -> pd.Series:
    # Guard against zero ATR (would produce inf). Replace with NaN.
    safe_atr = atr14.where(atr14 > 0)
    return (close - ema21) / safe_atr


def _po_bandwidth_percentile(po: pd.Series) -> pd.Series:
    """Compression metric: rolling stdev of PO, then percentile-rank over 6 months.
    Lower percentile = tighter PO = stronger squeeze."""
    po_std = po.rolling(_PO_BBW_INNER).std()
    return po_std.rolling(_PO_BBW_OUTER).rank(pct=True)
```

- [ ] **Step 8: Re-run, confirm passes**

```bash
pytest tests/domain/test_technical_signals.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add domain/technical_signals.py tests/domain/test_technical_signals.py
git commit -m "feat(domain): TechnicalSetup + indicator helpers (EMA/ATR/PO/PO-BBW pct)"
```

---

## Task 3: Detector 1 — Compression breakout

**Files:**
- Modify: `domain/technical_signals.py`
- Modify: `tests/domain/test_technical_signals.py`

- [ ] **Step 1: Write the failing test for bullish compression**

Append to `tests/domain/test_technical_signals.py`:

```python
from domain.technical_signals import _detect_compression_breakout


def _ohlcv_from_close(close_vals: list[float]) -> pd.DataFrame:
    close = pd.Series(close_vals)
    return pd.DataFrame({
        'open':   close.shift(1).fillna(close.iloc[0]),
        'high':   close * 1.003,
        'low':    close * 0.997,
        'close':  close,
        'volume': pd.Series(1_000_000, index=close.index),
    })


def test_compression_breakout_bullish_fires_on_squeeze_with_stacked_ribbon():
    # Strategy: build 250-bar series whose last 30 bars are very flat (squeeze),
    # whose ribbon is stacked bullish, and whose last bar is not extended.
    np.random.seed(7)
    base_up = np.linspace(80, 100, 220)
    base_flat = np.full(30, 100.0) + np.random.normal(0, 0.05, 30)  # very low vol
    close = np.concatenate([base_up, base_flat])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_compression_breakout(df)

    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'compression_breakout'
    assert 0.0 < setup.strength <= 1.0


def test_compression_breakout_returns_none_when_not_squeezed():
    # Volatile, non-squeezed series → no signal
    np.random.seed(11)
    close = 100 + np.cumsum(np.random.normal(0, 1.0, 250))
    df = _ohlcv_from_close(close.tolist())
    assert _detect_compression_breakout(df) is None
```

- [ ] **Step 2: Run, confirm tests fail**

```bash
pytest tests/domain/test_technical_signals.py::test_compression_breakout_bullish_fires_on_squeeze_with_stacked_ribbon -v
```

Expected: FAIL — `_detect_compression_breakout` not defined.

- [ ] **Step 3: Implement the detector**

Append to `domain/technical_signals.py`:

```python
_COMPRESSION_PCT_CUTOFF = 0.20
_NOT_EXTENDED_ATR_MULT = 0.5


def _detect_compression_breakout(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    emas = _emas(close)
    atr = _atr14(df['high'], df['low'], close)
    po = _phase_oscillator(close, emas[21], atr)
    bbw_pct = _po_bandwidth_percentile(po)

    last_bbw = bbw_pct.iloc[-1]
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    last_atr = atr.iloc[-1]

    if pd.isna(last_bbw) or pd.isna(last_atr) or last_atr <= 0:
        return None
    if last_bbw >= _COMPRESSION_PCT_CUTOFF:
        return None
    if abs(last_close - prev_close) >= _NOT_EXTENDED_ATR_MULT * last_atr:
        return None

    e8, e13, e21, e48 = emas[8].iloc[-1], emas[13].iloc[-1], emas[21].iloc[-1], emas[48].iloc[-1]
    strength = float(1.0 - last_bbw)

    if e8 > e13 > e21 and last_close > e48:
        return TechnicalSetup(
            setup_name='compression_breakout',
            direction='bullish',
            strength=strength,
            notes=f'PO squeeze pct={last_bbw:.2f}, ribbon stacked bullish',
        )
    if e8 < e13 < e21 and last_close < e48:
        return TechnicalSetup(
            setup_name='compression_breakout',
            direction='bearish',
            strength=strength,
            notes=f'PO squeeze pct={last_bbw:.2f}, ribbon stacked bearish',
        )
    return None
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/domain/test_technical_signals.py::test_compression_breakout_bullish_fires_on_squeeze_with_stacked_ribbon tests/domain/test_technical_signals.py::test_compression_breakout_returns_none_when_not_squeezed -v
```

Expected: PASS.

- [ ] **Step 5: Write the bearish-direction test**

Append to `tests/domain/test_technical_signals.py`:

```python
def test_compression_breakout_bearish_fires_on_inverse_setup():
    np.random.seed(7)
    base_down = np.linspace(120, 100, 220)
    base_flat = np.full(30, 100.0) + np.random.normal(0, 0.05, 30)
    close = np.concatenate([base_down, base_flat])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_compression_breakout(df)
    assert setup is not None
    assert setup.direction == 'bearish'
```

- [ ] **Step 6: Run, confirm passes**

```bash
pytest tests/domain/test_technical_signals.py::test_compression_breakout_bearish_fires_on_inverse_setup -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add domain/technical_signals.py tests/domain/test_technical_signals.py
git commit -m "feat(domain): compression breakout detector (bullish + bearish)"
```

---

## Task 4: Detector 2 — Pullback in trend

**Files:**
- Modify: `domain/technical_signals.py`
- Modify: `tests/domain/test_technical_signals.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/domain/test_technical_signals.py`:

```python
from domain.technical_signals import _detect_pullback_in_trend


def test_pullback_in_trend_bullish_fires_on_uptrend_pullback_to_21ema():
    # Build a 250-bar steady uptrend, then a 1-bar pullback whose low touches the
    # 13-21 EMA zone, with PO near zero and volume below avg.
    base = np.linspace(80, 130, 240)
    pullback_close = base[-1] * 0.97
    close = np.concatenate([base, [base[-1] * 0.99, pullback_close]])
    df = pd.DataFrame({
        'open':   pd.Series(close).shift(1).fillna(close[0]),
        'high':   pd.Series(close) * 1.003,
        'low':    pd.Series(close) * 0.985,   # deep wick to touch the EMAs
        'close':  pd.Series(close),
        'volume': pd.Series([1_000_000] * (len(close) - 1) + [600_000]),  # last bar low vol
    })

    setup = _detect_pullback_in_trend(df)
    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'pullback_in_trend'


def test_pullback_in_trend_returns_none_on_flat_market():
    close = pd.Series(np.full(250, 100.0) + np.random.RandomState(3).normal(0, 0.5, 250))
    df = _ohlcv_from_close(close.tolist())
    assert _detect_pullback_in_trend(df) is None
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/domain/test_technical_signals.py::test_pullback_in_trend_bullish_fires_on_uptrend_pullback_to_21ema -v
```

Expected: FAIL — function does not exist.

- [ ] **Step 3: Implement detector**

Append to `domain/technical_signals.py`:

```python
_PO_LAUNCH_BAND = 23.6
_PULLBACK_VOL_MULT = 1.1


def _detect_pullback_in_trend(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    low = df['low']
    volume = df['volume']
    emas = _emas(close)
    atr = _atr14(df['high'], low, close)
    po = _phase_oscillator(close, emas[21], atr)

    e8, e13, e21, e48 = emas[8].iloc[-1], emas[13].iloc[-1], emas[21].iloc[-1], emas[48].iloc[-1]
    e200 = emas[200].iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    last_po = po.iloc[-1]
    last_vol = volume.iloc[-1]
    avg_vol_prev = volume.iloc[-21:-1].mean()

    if pd.isna(last_po) or pd.isna(avg_vol_prev) or avg_vol_prev <= 0:
        return None
    if not (-_PO_LAUNCH_BAND <= last_po <= _PO_LAUNCH_BAND):
        return None
    if last_vol >= _PULLBACK_VOL_MULT * avg_vol_prev:
        return None

    strength = float(1.0 - abs(last_po) / _PO_LAUNCH_BAND)
    strength = max(0.0, min(1.0, strength))

    bullish_stack = e8 > e13 > e21 > e48 and last_close > e200
    bearish_stack = e8 < e13 < e21 < e48 and last_close < e200
    bullish_touch = e21 * 0.99 <= last_low <= e13 * 1.01
    bearish_touch = e13 * 0.99 <= df['high'].iloc[-1] <= e21 * 1.01

    if bullish_stack and bullish_touch:
        return TechnicalSetup(
            setup_name='pullback_in_trend',
            direction='bullish',
            strength=strength,
            notes=f'Ribbon stacked bullish, low tagged 13-21 EMA, PO={last_po:.1f}',
        )
    if bearish_stack and bearish_touch:
        return TechnicalSetup(
            setup_name='pullback_in_trend',
            direction='bearish',
            strength=strength,
            notes=f'Ribbon stacked bearish, high tagged 21-13 EMA, PO={last_po:.1f}',
        )
    return None
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/domain/test_technical_signals.py::test_pullback_in_trend_bullish_fires_on_uptrend_pullback_to_21ema tests/domain/test_technical_signals.py::test_pullback_in_trend_returns_none_on_flat_market -v
```

Expected: PASS.

- [ ] **Step 5: Write bearish test**

Append:

```python
def test_pullback_in_trend_bearish_fires_on_downtrend_rally_to_21ema():
    base = np.linspace(130, 80, 240)
    rally_close = base[-1] * 1.03
    close = np.concatenate([base, [base[-1] * 1.01, rally_close]])
    df = pd.DataFrame({
        'open':   pd.Series(close).shift(1).fillna(close[0]),
        'high':   pd.Series(close) * 1.015,
        'low':    pd.Series(close) * 0.997,
        'close':  pd.Series(close),
        'volume': pd.Series([1_000_000] * (len(close) - 1) + [600_000]),
    })
    setup = _detect_pullback_in_trend(df)
    assert setup is not None
    assert setup.direction == 'bearish'
```

- [ ] **Step 6: Run, confirm pass**

```bash
pytest tests/domain/test_technical_signals.py::test_pullback_in_trend_bearish_fires_on_downtrend_rally_to_21ema -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add domain/technical_signals.py tests/domain/test_technical_signals.py
git commit -m "feat(domain): pullback-in-trend detector (bullish + bearish)"
```

---

## Task 5: Detector 3 — Stage 2 breakout

**Files:**
- Modify: `domain/technical_signals.py`
- Modify: `tests/domain/test_technical_signals.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from domain.technical_signals import _detect_stage_2_breakout


def test_stage_2_breakout_bullish_fires_on_fresh_ema48_cross_with_prior_squeeze():
    # 200 bars of base, 10 bars of squeeze, then a 5-bar breakout above EMA48 with PO > 0
    np.random.seed(13)
    base = np.linspace(80, 95, 200)
    squeeze = np.full(15, 95.0) + np.random.normal(0, 0.05, 15)
    breakout = np.linspace(95.0, 105.0, 35)  # extends above the EMAs and crosses fresh
    close = np.concatenate([base, squeeze, breakout])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_stage_2_breakout(df)
    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'stage_2_breakout'


def test_stage_2_breakout_returns_none_when_far_from_52w_high():
    # Trending up but very far below recent high (we simulate by having an earlier higher peak)
    close = np.concatenate([np.linspace(80, 200, 100), np.linspace(200, 110, 150)])
    df = _ohlcv_from_close(close.tolist())
    assert _detect_stage_2_breakout(df) is None
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/domain/test_technical_signals.py::test_stage_2_breakout_bullish_fires_on_fresh_ema48_cross_with_prior_squeeze -v
```

Expected: FAIL.

- [ ] **Step 3: Implement detector**

Append to `domain/technical_signals.py`:

```python
_FRESH_CROSS_BARS = 5
_NEAR_52W_PCT = 0.10
_PRIOR_COMPRESSION_LOOKBACK = 20
_PRIOR_COMPRESSION_PCT = 0.25
_EMA200_SLOPE_LOOKBACK = 30


def _crossed_above_within(series: pd.Series, reference: pd.Series, bars: int) -> bool:
    window_s = series.iloc[-(bars + 1):]
    window_r = reference.iloc[-(bars + 1):]
    diff = window_s.values - window_r.values
    return bool((diff[:-1] <= 0).any() and diff[-1] > 0)


def _crossed_below_within(series: pd.Series, reference: pd.Series, bars: int) -> bool:
    window_s = series.iloc[-(bars + 1):]
    window_r = reference.iloc[-(bars + 1):]
    diff = window_s.values - window_r.values
    return bool((diff[:-1] >= 0).any() and diff[-1] < 0)


def _crossed_zero_above(series: pd.Series, bars: int) -> bool:
    window = series.iloc[-(bars + 1):]
    return bool((window.iloc[:-1] <= 0).any() and window.iloc[-1] > 0)


def _crossed_zero_below(series: pd.Series, bars: int) -> bool:
    window = series.iloc[-(bars + 1):]
    return bool((window.iloc[:-1] >= 0).any() and window.iloc[-1] < 0)


def _detect_stage_2_breakout(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    high = df['high']
    low = df['low']
    emas = _emas(close)
    atr = _atr14(high, low, close)
    po = _phase_oscillator(close, emas[21], atr)
    bbw_pct = _po_bandwidth_percentile(po)

    e48 = emas[48]
    e200 = emas[200]
    last_close = close.iloc[-1]
    last_atr = atr.iloc[-1]

    if pd.isna(last_atr) or last_atr <= 0:
        return None
    if pd.isna(e200.iloc[-1]) or pd.isna(e200.iloc[-_EMA200_SLOPE_LOOKBACK - 1]):
        return None

    ema200_rising = e200.iloc[-1] > e200.iloc[-_EMA200_SLOPE_LOOKBACK - 1]
    ema200_falling = e200.iloc[-1] < e200.iloc[-_EMA200_SLOPE_LOOKBACK - 1]

    recent_compression = bbw_pct.iloc[-_PRIOR_COMPRESSION_LOOKBACK:].min()
    if pd.isna(recent_compression) or recent_compression >= _PRIOR_COMPRESSION_PCT:
        return None

    # Bullish branch
    high_52w = high.iloc[-252:].max() if len(high) >= 252 else high.max()
    near_high = last_close >= (1 - _NEAR_52W_PCT) * high_52w

    if (ema200_rising
        and _crossed_above_within(close, e48, _FRESH_CROSS_BARS)
        and _crossed_zero_above(po, _FRESH_CROSS_BARS)
        and near_high):
        strength = float(np.clip((last_close - e48.iloc[-1]) / last_atr, 0.0, 2.0) / 2.0)
        return TechnicalSetup(
            setup_name='stage_2_breakout',
            direction='bullish',
            strength=strength,
            notes='EMA200 rising, fresh EMA48 + PO zero cross, near 52w high',
        )

    # Bearish branch (Stage 4 breakdown)
    low_52w = low.iloc[-252:].min() if len(low) >= 252 else low.min()
    near_low = last_close <= (1 + _NEAR_52W_PCT) * low_52w

    if (ema200_falling
        and _crossed_below_within(close, e48, _FRESH_CROSS_BARS)
        and _crossed_zero_below(po, _FRESH_CROSS_BARS)
        and near_low):
        strength = float(np.clip((e48.iloc[-1] - last_close) / last_atr, 0.0, 2.0) / 2.0)
        return TechnicalSetup(
            setup_name='stage_2_breakout',
            direction='bearish',
            strength=strength,
            notes='EMA200 falling, fresh EMA48 + PO zero cross, near 52w low',
        )

    return None
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest tests/domain/test_technical_signals.py::test_stage_2_breakout_bullish_fires_on_fresh_ema48_cross_with_prior_squeeze tests/domain/test_technical_signals.py::test_stage_2_breakout_returns_none_when_far_from_52w_high -v
```

Expected: PASS.

- [ ] **Step 5: Bearish test**

Append:

```python
def test_stage_2_breakout_bearish_fires_on_stage_4_breakdown():
    np.random.seed(17)
    base = np.linspace(120, 105, 200)
    squeeze = np.full(15, 105.0) + np.random.normal(0, 0.05, 15)
    breakdown = np.linspace(105.0, 90.0, 35)
    close = np.concatenate([base, squeeze, breakdown])
    df = _ohlcv_from_close(close.tolist())

    setup = _detect_stage_2_breakout(df)
    assert setup is not None
    assert setup.direction == 'bearish'
```

- [ ] **Step 6: Run, confirm pass**

```bash
pytest tests/domain/test_technical_signals.py::test_stage_2_breakout_bearish_fires_on_stage_4_breakdown -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add domain/technical_signals.py tests/domain/test_technical_signals.py
git commit -m "feat(domain): stage 2 breakout / stage 4 breakdown detector"
```

---

## Task 6: Detector 4 — Failed breakdown reversal

**Files:**
- Modify: `domain/technical_signals.py`
- Modify: `tests/domain/test_technical_signals.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from domain.technical_signals import _detect_failed_breakdown_reversal


def test_failed_breakdown_reversal_bullish_fires_on_reclaim_with_divergence():
    # 250 bars: build a downtrend to a low at bar 240, then 5 bars that briefly
    # break the prior 20-day low (low of bar 245), then reverse closing above
    # the prior support with high volume.
    base = np.linspace(120, 100, 240).tolist()
    breakdown_lows = [99.0, 97.5, 95.0, 97.0, 100.5]  # 5 bars: 3 lower lows then reversal
    close = np.array(base + breakdown_lows)
    high = np.array(base + [c + 1.0 for c in breakdown_lows])
    low = np.array(base + [c - 0.5 for c in breakdown_lows])
    low[-3] = 93.0   # the new 20-day low
    volume = np.concatenate([
        np.full(245, 1_000_000),
        np.array([1_800_000]),   # reversal bar with conviction
    ])
    df = pd.DataFrame({
        'open': pd.Series(close).shift(1).fillna(close[0]),
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })

    setup = _detect_failed_breakdown_reversal(df)
    assert setup is not None
    assert setup.direction == 'bullish'
    assert setup.setup_name == 'failed_breakdown_reversal'


def test_failed_breakdown_reversal_returns_none_on_continuous_decline():
    close = np.linspace(120, 80, 250)
    df = _ohlcv_from_close(close.tolist())
    assert _detect_failed_breakdown_reversal(df) is None
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/domain/test_technical_signals.py::test_failed_breakdown_reversal_bullish_fires_on_reclaim_with_divergence -v
```

Expected: FAIL.

- [ ] **Step 3: Implement detector**

Append:

```python
_REVERSAL_LOOKBACK = 25
_REVERSAL_RECENT_BARS = 5
_REVERSAL_VOL_MULT = 1.5
_DIVERGENCE_WINDOW = 20


def _has_bullish_divergence(low: pd.Series, po: pd.Series) -> bool:
    window_low = low.iloc[-_DIVERGENCE_WINDOW:]
    window_po = po.iloc[-_DIVERGENCE_WINDOW:]
    if window_low.isna().any() or window_po.isna().any():
        return False
    # Compare the two halves: did the second-half low go lower while PO low went higher?
    first_low = window_low.iloc[: _DIVERGENCE_WINDOW // 2].min()
    second_low = window_low.iloc[_DIVERGENCE_WINDOW // 2 :].min()
    first_po_low = window_po.iloc[: _DIVERGENCE_WINDOW // 2].min()
    second_po_low = window_po.iloc[_DIVERGENCE_WINDOW // 2 :].min()
    return bool(second_low < first_low and second_po_low > first_po_low)


def _has_bearish_divergence(high: pd.Series, po: pd.Series) -> bool:
    window_high = high.iloc[-_DIVERGENCE_WINDOW:]
    window_po = po.iloc[-_DIVERGENCE_WINDOW:]
    if window_high.isna().any() or window_po.isna().any():
        return False
    first_high = window_high.iloc[: _DIVERGENCE_WINDOW // 2].max()
    second_high = window_high.iloc[_DIVERGENCE_WINDOW // 2 :].max()
    first_po_high = window_po.iloc[: _DIVERGENCE_WINDOW // 2].max()
    second_po_high = window_po.iloc[_DIVERGENCE_WINDOW // 2 :].max()
    return bool(second_high > first_high and second_po_high < first_po_high)


def _detect_failed_breakdown_reversal(df: pd.DataFrame) -> Optional[TechnicalSetup]:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    emas = _emas(close)
    atr = _atr14(high, low, close)
    po = _phase_oscillator(close, emas[21], atr)

    last_close = close.iloc[-1]
    last_vol = volume.iloc[-1]
    avg_vol_prev = volume.iloc[-21:-1].mean()
    if pd.isna(avg_vol_prev) or avg_vol_prev <= 0:
        return None
    if last_vol < _REVERSAL_VOL_MULT * avg_vol_prev:
        return None

    prior_window_low = low.iloc[-_REVERSAL_LOOKBACK:-_REVERSAL_RECENT_BARS]
    recent_window_low = low.iloc[-_REVERSAL_RECENT_BARS:]
    prior_min_low = prior_window_low.min()
    recent_min_low = recent_window_low.min()

    prior_window_high = high.iloc[-_REVERSAL_LOOKBACK:-_REVERSAL_RECENT_BARS]
    recent_window_high = high.iloc[-_REVERSAL_RECENT_BARS:]
    prior_max_high = prior_window_high.max()
    recent_max_high = recent_window_high.max()

    strength = float(min(1.0, last_vol / (_REVERSAL_VOL_MULT * avg_vol_prev)))

    if (recent_min_low < prior_min_low
        and last_close > prior_min_low
        and _has_bullish_divergence(low, po)):
        return TechnicalSetup(
            setup_name='failed_breakdown_reversal',
            direction='bullish',
            strength=strength,
            notes='Reclaimed prior 20d low with bullish PO divergence on high volume',
        )

    if (recent_max_high > prior_max_high
        and last_close < prior_max_high
        and _has_bearish_divergence(high, po)):
        return TechnicalSetup(
            setup_name='failed_breakdown_reversal',
            direction='bearish',
            strength=strength,
            notes='Rejected prior 20d high with bearish PO divergence on high volume',
        )

    return None
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest tests/domain/test_technical_signals.py::test_failed_breakdown_reversal_bullish_fires_on_reclaim_with_divergence tests/domain/test_technical_signals.py::test_failed_breakdown_reversal_returns_none_on_continuous_decline -v
```

Expected: PASS.

- [ ] **Step 5: Bearish test**

Append:

```python
def test_failed_breakdown_reversal_bearish_fires_on_failed_breakout():
    base = np.linspace(80, 100, 240).tolist()
    breakout_highs = [101.0, 102.5, 105.0, 102.0, 99.0]
    close = np.array(base + breakout_highs)
    high = np.array(base + [c + 1.0 for c in breakout_highs])
    high[-3] = 107.0   # the new 20-day high
    low = np.array(base + [c - 0.5 for c in breakout_highs])
    volume = np.concatenate([np.full(245, 1_000_000), np.array([1_800_000])])
    df = pd.DataFrame({
        'open': pd.Series(close).shift(1).fillna(close[0]),
        'high': high, 'low': low, 'close': close, 'volume': volume,
    })

    setup = _detect_failed_breakdown_reversal(df)
    assert setup is not None
    assert setup.direction == 'bearish'
```

- [ ] **Step 6: Run, confirm pass**

```bash
pytest tests/domain/test_technical_signals.py::test_failed_breakdown_reversal_bearish_fires_on_failed_breakout -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add domain/technical_signals.py tests/domain/test_technical_signals.py
git commit -m "feat(domain): failed-breakdown / failed-breakout reversal detector"
```

---

## Task 7: detect_setups aggregator + edge cases

**Files:**
- Modify: `domain/technical_signals.py`
- Modify: `tests/domain/test_technical_signals.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from domain.technical_signals import detect_setups


def test_detect_setups_returns_empty_when_history_too_short():
    close = pd.Series(np.linspace(100, 110, 100))  # only 100 bars
    df = _ohlcv_from_close(close.tolist())
    assert detect_setups(df) == []


def test_detect_setups_returns_empty_for_flat_nan_history():
    df = pd.DataFrame({
        'open':   pd.Series([np.nan] * 250),
        'high':   pd.Series([np.nan] * 250),
        'low':    pd.Series([np.nan] * 250),
        'close':  pd.Series([np.nan] * 250),
        'volume': pd.Series([np.nan] * 250),
    })
    assert detect_setups(df) == []


def test_detect_setups_returns_list_when_setup_matches():
    np.random.seed(7)
    base_up = np.linspace(80, 100, 220)
    base_flat = np.full(30, 100.0) + np.random.normal(0, 0.05, 30)
    close = np.concatenate([base_up, base_flat])
    df = _ohlcv_from_close(close.tolist())
    out = detect_setups(df)
    assert len(out) >= 1
    assert all(isinstance(s, TechnicalSetup) for s in out)


def test_detect_setups_is_deterministic():
    np.random.seed(7)
    base_up = np.linspace(80, 100, 220)
    base_flat = np.full(30, 100.0) + np.random.normal(0, 0.05, 30)
    close = np.concatenate([base_up, base_flat])
    df = _ohlcv_from_close(close.tolist())
    first = detect_setups(df)
    second = detect_setups(df)
    assert first == second
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/domain/test_technical_signals.py::test_detect_setups_returns_empty_when_history_too_short -v
```

Expected: FAIL — `detect_setups` not defined.

- [ ] **Step 3: Implement the aggregator**

Append to `domain/technical_signals.py`:

```python
_MIN_HISTORY = 220
_REQUIRED_COLS = ('open', 'high', 'low', 'close', 'volume')


def detect_setups(history: pd.DataFrame) -> list[TechnicalSetup]:
    """Run all detectors against a daily OHLCV history. Returns non-None setups.

    Guarantees:
    - len(history) < 220 → []
    - All-NaN / flat history → []
    - Never raises.
    """
    if history is None or len(history) < _MIN_HISTORY:
        return []
    if not all(col in history.columns for col in _REQUIRED_COLS):
        return []
    if history['close'].isna().all() or history['close'].dropna().nunique() <= 1:
        return []

    detectors = (
        _detect_compression_breakout,
        _detect_pullback_in_trend,
        _detect_stage_2_breakout,
        _detect_failed_breakdown_reversal,
    )
    out: list[TechnicalSetup] = []
    for fn in detectors:
        try:
            setup = fn(history)
        except Exception:
            setup = None
        if setup is not None:
            out.append(setup)
    return out
```

- [ ] **Step 4: Run, confirm all 4 tests pass**

```bash
pytest tests/domain/test_technical_signals.py -v
```

Expected: full file passes.

- [ ] **Step 5: Commit**

```bash
git add domain/technical_signals.py tests/domain/test_technical_signals.py
git commit -m "feat(domain): detect_setups aggregator with edge-case guards"
```

---

## Task 8: pipeline/technical_filter.py

**Files:**
- Create: `pipeline/technical_filter.py`
- Create: `tests/pipeline/test_technical_filter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_technical_filter.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from data.fallback import DataFetchError
from data.sources.base import DataSource
from pipeline.technical_filter import filter_by_technicals


def _ohlcv_uptrend_with_squeeze(seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    base = np.linspace(80, 100, 220)
    flat = np.full(30, 100.0) + np.random.normal(0, 0.05, 30)
    close = np.concatenate([base, flat])
    s = pd.Series(close)
    return pd.DataFrame({
        'open':   s.shift(1).fillna(s.iloc[0]),
        'high':   s * 1.003,
        'low':    s * 0.997,
        'close':  s,
        'volume': pd.Series(1_000_000, index=s.index),
    })


def _ohlcv_flat() -> pd.DataFrame:
    close = pd.Series(np.full(250, 100.0) + np.random.RandomState(0).normal(0, 0.3, 250))
    return pd.DataFrame({
        'open':   close.shift(1).fillna(close.iloc[0]),
        'high':   close * 1.003,
        'low':    close * 0.997,
        'close':  close,
        'volume': pd.Series(1_000_000, index=close.index),
    })


class _FakeSource(DataSource):
    def __init__(self, histories: dict[str, pd.DataFrame]):
        self._histories = histories
    def fetch_spot(self, ticker): return 100.0
    def fetch_price_history(self, ticker, lookback_days): return self._histories[ticker]['close']
    def fetch_price_history_ohlcv(self, ticker, lookback_days):
        if ticker not in self._histories:
            raise RuntimeError(f"unknown ticker {ticker}")
        return self._histories[ticker]
    def fetch_option_chain(self, ticker): return []


def test_filter_drops_tickers_without_setups():
    src = _FakeSource({
        'AAA': _ohlcv_uptrend_with_squeeze(),    # bullish compression
        'BBB': _ohlcv_flat(),                    # no setup
    })
    out = filter_by_technicals(['AAA', 'BBB'], [src])
    assert 'AAA' in out
    assert 'BBB' not in out
    assert len(out['AAA']) >= 1


def test_filter_handles_fetch_failure_gracefully():
    src = _FakeSource({'AAA': _ohlcv_uptrend_with_squeeze()})
    # CCC will raise because it's not in the dict
    out = filter_by_technicals(['AAA', 'CCC'], [src])
    assert 'AAA' in out
    assert 'CCC' not in out
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/pipeline/test_technical_filter.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the filter stage**

Create `pipeline/technical_filter.py`:

```python
from __future__ import annotations
import logging

from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.base import DataSource
from domain.technical_signals import TechnicalSetup, detect_setups


log = logging.getLogger(__name__)

_OHLCV_LOOKBACK_DAYS = 365


def filter_by_technicals(
    tickers: list[str],
    sources: list[DataSource],
) -> dict[str, list[TechnicalSetup]]:
    """For each ticker, fetch OHLCV history and run detect_setups.
    Returns dict containing only tickers with at least one setup matched.
    Tickers whose fetch raises are logged and dropped."""
    results: dict[str, list[TechnicalSetup]] = {}
    for ticker in tickers:
        try:
            history = fetch_with_fallback(
                sources, 'fetch_price_history_ohlcv', ticker, _OHLCV_LOOKBACK_DAYS,
            )
        except DataFetchError as e:
            log.info("technical_filter: dropping %s — fetch failed (%s)", ticker, e)
            continue
        setups = detect_setups(history)
        if setups:
            results[ticker] = setups
    return results
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest tests/pipeline/test_technical_filter.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/technical_filter.py tests/pipeline/test_technical_filter.py
git commit -m "feat(pipeline): filter_by_technicals stage"
```

---

## Task 9: Extend ScoredCandidate + ScanConfig

**Files:**
- Modify: `engine/scorer.py`
- Modify: `pipeline/run_scan.py`
- Modify: `tests/engine/test_scorer.py`

- [ ] **Step 1: Write a failing test for ScoredCandidate setup fields**

Edit `tests/engine/test_scorer.py`, append:

```python
def test_scored_candidate_supports_optional_setup_fields():
    from engine.scorer import ScoredCandidate
    # Build a minimal instance using only the fields needed for this test —
    # other required fields are stubbed with simple values.
    fields_present = ScoredCandidate.__dataclass_fields__
    assert 'setup_name' in fields_present
    assert 'setup_direction' in fields_present
    assert 'setup_strength' in fields_present
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/engine/test_scorer.py::test_scored_candidate_supports_optional_setup_fields -v
```

Expected: FAIL — fields not present.

- [ ] **Step 3: Add the three optional fields to ScoredCandidate**

Edit `engine/scorer.py`. Update the `ScoredCandidate` dataclass:

```python
from typing import Literal, Optional
import numpy as np

from domain.contract import Contract
from domain.strategy import Strategy
from domain.signals import Regime
from engine.stress import StressResult
from engine.risk_filters import FilterResult


Label = Literal['TRADE', 'WATCHLIST', 'NO_TRADE']


@dataclass
class ScoredCandidate:
    contract: Contract
    strategy: Strategy
    pop_blended: float
    pop_delta: float
    pop_bs: float
    pop_historical: float
    pop_garch_mc: float
    stress: StressResult
    ev: float
    max_adverse_loss: float
    margin_estimate: float
    filter_result: FilterResult
    composite_score: float
    label: Label
    reason_for: str
    reason_against: str
    setup_name: Optional[str] = None
    setup_direction: Optional[str] = None
    setup_strength: Optional[float] = None
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest tests/engine/test_scorer.py -v
```

Expected: PASS (no other tests break — defaults preserve compatibility).

- [ ] **Step 5: Add `use_technical_filter` to ScanConfig**

Edit `pipeline/run_scan.py`. Update `ScanConfig`:

```python
@dataclass
class ScanConfig:
    risk_free_rate: float = 0.053
    earnings_blackout_days: int = 5
    n_monte_carlo_paths: int = 10_000
    today: date = field(default_factory=date.today)
    universe_filters: UniverseFilters = field(default_factory=UniverseFilters)
    use_technical_filter: bool = True
```

- [ ] **Step 6: Run full suite to confirm nothing broke**

```bash
pytest -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add engine/scorer.py pipeline/run_scan.py tests/engine/test_scorer.py
git commit -m "feat(engine): add optional setup fields to ScoredCandidate; ScanConfig.use_technical_filter"
```

---

## Task 10: Integrate filter + strategy gate into run_scan

**Files:**
- Modify: `pipeline/run_scan.py`

### Why

This is the central integration. We insert the filter stage, switch the per-ticker loop to iterate filter results, replace the direction gate, and populate setup fields on each candidate.

- [ ] **Step 1: Add the helper for strategy ↔ setup direction mapping**

Edit `pipeline/run_scan.py`. After the imports and before `run_scan`, add:

```python
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from domain.technical_signals import TechnicalSetup
from pipeline.technical_filter import filter_by_technicals


def _strategy_matches_setup(strategy, setup_dirs: set[str]) -> bool:
    """Map strategy → required setup direction.
    - LongCall, NakedPut are bullish-direction bets.
    - LongPut, NakedCall are bearish-direction bets.
    """
    if isinstance(strategy, (LongCall, NakedPut)):
        return 'bullish' in setup_dirs
    if isinstance(strategy, (LongPut, NakedCall)):
        return 'bearish' in setup_dirs
    return False


def _strongest_matching_setup(setups: list[TechnicalSetup], strategy) -> TechnicalSetup | None:
    target_dir = 'bullish' if isinstance(strategy, (LongCall, NakedPut)) else 'bearish'
    matches = [s for s in setups if s.direction == target_dir]
    if not matches:
        return None
    return max(matches, key=lambda s: s.strength)
```

- [ ] **Step 2: Insert filter stage in `run_scan`**

Edit `pipeline/run_scan.py`. Replace the body of `run_scan` from `universe = ...` through the candidate loop with:

```python
def run_scan(config: ScanConfig, sources_override: Optional[list[DataSource]] = None) -> ScanResult:
    sources = sources_override if sources_override is not None else [
        YahooQuerySource(), YfinanceSource(), StooqSource()
    ]
    universe = build_universe_cached(config.universe_filters, sources)
    candidates: list[ScoredCandidate] = []
    skipped: dict[str, str] = {}

    if config.use_technical_filter:
        setups_by_ticker = filter_by_technicals(universe, sources)
        ticker_iter = list(setups_by_ticker.keys())
    else:
        setups_by_ticker = {t: [] for t in universe}
        ticker_iter = universe

    for ticker in ticker_iter:
        try:
            if has_earnings_within(ticker, config.earnings_blackout_days):
                skipped[ticker] = 'earnings_blackout'
                continue
            history = fetch_with_fallback(sources, 'fetch_price_history', ticker, 365)
            spot = fetch_with_fallback(sources, 'fetch_spot', ticker)
            raw_chain = fetch_with_fallback(sources[:2], 'fetch_option_chain', ticker)
        except DataFetchError as e:
            skipped[ticker] = f'fetch_failed: {e}'
            continue

        if history is None or len(history) < 30 or spot <= 0:
            skipped[ticker] = 'insufficient_history'
            continue

        log_returns = np.log(history / history.shift(1)).dropna()
        regime = compute_regime(ticker, log_returns, raw_chain)
        setups = setups_by_ticker.get(ticker, [])
        setup_dirs = {s.direction for s in setups}

        for strategy in ALL_STRATEGIES:
            if config.use_technical_filter:
                if not _strategy_matches_setup(strategy, setup_dirs):
                    continue
            else:
                if strategy.direction not in regime.favored:
                    continue
            for raw in raw_chain:
                if raw.option_type != strategy.option_type:
                    continue
                contract = to_contract(raw, spot, config.risk_free_rate, config.today)
                if contract is None or not strategy.applies_to(contract):
                    continue

                p_d = pop_delta(contract, strategy)
                p_bs = pop_black_scholes(contract, strategy, config.risk_free_rate)
                p_hist = pop_historical(contract, strategy, log_returns)
                p_garch = pop_garch_mc(contract, strategy, log_returns, n_paths=config.n_monte_carlo_paths)
                p_blend = blend_pop(p_d, p_bs, p_hist, p_garch)

                stress = compute_stress(contract, strategy, contract.implied_volatility)
                max_adverse = abs(min(stress.stress_2sd, 0.0))

                if strategy.direction == 'sell':
                    ev = strategy.expected_value(contract, p_blend, max_adverse)
                else:
                    np.random.seed(42)
                    daily_vol = float(log_returns.std())
                    z = np.random.standard_normal((2000, contract.dte))
                    cumulative = (daily_vol * z).sum(axis=1)
                    terminal = contract.spot_price * np.exp(cumulative)
                    expected_profit_itm = _expected_profit_when_itm_long(terminal, contract, strategy)
                    ev = strategy.expected_value(contract, p_blend, max_adverse, expected_profit_when_itm=expected_profit_itm)

                fr = apply_filters(contract, strategy, p_blend, ev, stress)
                score = composite_score(
                    pop_blended=p_blend, ev=ev, max_adverse_loss=max_adverse,
                    strategy=strategy, regime=regime, mid=contract.mid,
                )
                label = label_from(score, fr.passed)

                chosen = _strongest_matching_setup(setups, strategy) if setups else None

                reason_for = (
                    f"PoP={p_blend:.0%}, EV=${ev:.2f}, regime ratio={regime.rv_iv_ratio:.2f} "
                    f"favors {strategy.direction}"
                    + (f", setup={chosen.setup_name} ({chosen.direction})" if chosen else "")
                )
                reason_against = (
                    f"Filters failed: {', '.join(fr.failed_filters)}" if fr.failed_filters else ""
                )

                candidates.append(ScoredCandidate(
                    contract=contract, strategy=strategy,
                    pop_blended=p_blend, pop_delta=p_d, pop_bs=p_bs,
                    pop_historical=p_hist, pop_garch_mc=p_garch,
                    stress=stress, ev=ev, max_adverse_loss=max_adverse,
                    margin_estimate=strategy.margin_estimate(contract),
                    filter_result=fr, composite_score=score, label=label,
                    reason_for=reason_for, reason_against=reason_against,
                    setup_name=chosen.setup_name if chosen else None,
                    setup_direction=chosen.direction if chosen else None,
                    setup_strength=chosen.strength if chosen else None,
                ))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return ScanResult(timestamp=datetime.now(), config=config,
                      candidates=candidates, skipped=skipped)
```

- [ ] **Step 3: Run the existing run_scan test to confirm no regression**

```bash
pytest tests/pipeline/test_run_scan.py -v
```

Expected: existing tests still PASS (they patch `build_universe_cached` and use `use_technical_filter=True` default, but with no setups they'll return zero candidates — which is fine for the existing assertions that check the result shape, not specific candidates). If any fails, ensure the existing test sets `use_technical_filter=False` to preserve legacy behavior; or relax the candidate assertions to allow empty.

- [ ] **Step 4: If existing tests fail, set `use_technical_filter=False` on legacy fixtures**

In `tests/pipeline/test_run_scan.py`, find existing `ScanConfig(...)` constructions and update to:

```python
ScanConfig(today=date(2026, 5, 30), use_technical_filter=False)
```

- [ ] **Step 5: Re-run to confirm pass**

```bash
pytest tests/pipeline/test_run_scan.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/run_scan.py tests/pipeline/test_run_scan.py
git commit -m "feat(pipeline): integrate technical filter + strategy gate into run_scan"
```

---

## Task 11: --no-technical-filter CLI flag

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the flag and wire it to ScanConfig**

Edit `main.py`. Update the `scan` subparser and `cmd_scan`:

In `main()`, after `s.add_argument('--fast', ...)`:

```python
    s.add_argument('--no-technical-filter', action='store_true',
                   help='Bypass Saty technical filter; use legacy RV/IV regime-only gating')
```

In `cmd_scan`, after `config = ScanConfig(universe_filters=filters)`:

```python
    config.use_technical_filter = not args.no_technical_filter
```

- [ ] **Step 2: Smoke-test CLI parses the flag**

```bash
python -m main scan --help | grep technical
```

Expected output: `--no-technical-filter  Bypass Saty technical filter...`

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(cli): --no-technical-filter flag"
```

---

## Task 12: Output additions — exporter + dashboard

**Files:**
- Modify: `ui/exporter.py`
- Modify: `ui/dashboard.py`
- Modify: `tests/ui/test_exporter.py` (if exists; else create)

- [ ] **Step 1: Add CSV columns**

Edit `ui/exporter.py`. In the `columns` list (currently `ui/exporter.py:30-35`), append the three new fields just before `'reason_for'`:

```python
    columns = ['ticker', 'strategy', 'option_type', 'strike', 'expiration', 'dte',
               'spot_price', 'bid', 'ask', 'mid', 'volume', 'open_interest',
               'implied_volatility', 'delta', 'gamma', 'theta', 'vega',
               'pop_blended', 'pop_delta', 'pop_bs', 'pop_historical', 'pop_garch_mc',
               'ev', 'max_adverse_loss', 'margin_estimate', 'composite_score', 'label',
               'setup_name', 'setup_direction', 'setup_strength',
               'reason_for', 'reason_against', 'failed_filters']
```

Then in the CSV row write (currently `ui/exporter.py:41-49`), insert the three values at the matching positions:

```python
            w.writerow([
                con.ticker, c.strategy.name, con.option_type, con.strike,
                con.expiration.isoformat(), con.dte, con.spot_price,
                con.bid, con.ask, con.mid, con.volume, con.open_interest,
                con.implied_volatility, con.delta, con.gamma, con.theta, con.vega,
                c.pop_blended, c.pop_delta, c.pop_bs, c.pop_historical, c.pop_garch_mc,
                c.ev, c.max_adverse_loss, c.margin_estimate, c.composite_score, c.label,
                c.setup_name or '',
                c.setup_direction or '',
                c.setup_strength if c.setup_strength is not None else '',
                c.reason_for, c.reason_against, ';'.join(c.filter_result.failed_filters),
            ])
```

- [ ] **Step 2: Add JSON fields**

In the JSON candidate dict construction (currently `ui/exporter.py:54-66`), add the three new keys before `'failed_filters'`:

```python
            {
                'contract': _serialize(c.contract),
                'strategy': c.strategy.name,
                'pop_blended': c.pop_blended,
                'pop_delta': c.pop_delta, 'pop_bs': c.pop_bs,
                'pop_historical': c.pop_historical, 'pop_garch_mc': c.pop_garch_mc,
                'stress': _serialize(c.stress),
                'ev': c.ev, 'max_adverse_loss': c.max_adverse_loss,
                'margin_estimate': c.margin_estimate,
                'composite_score': c.composite_score, 'label': c.label,
                'setup_name': c.setup_name,
                'setup_direction': c.setup_direction,
                'setup_strength': c.setup_strength,
                'reason_for': c.reason_for, 'reason_against': c.reason_against,
                'failed_filters': list(c.filter_result.failed_filters),
            }
```

- [ ] **Step 3: Add a smoke test for CSV serialization**

Edit `tests/ui/test_exporter.py` (or create if missing). Add:

```python
def test_exporter_includes_setup_columns_in_csv(tmp_path):
    from datetime import datetime, date
    from engine.scorer import ScoredCandidate
    from engine.stress import StressResult
    from engine.risk_filters import FilterResult
    from domain.contract import Contract
    from domain.strategy import LongCall
    from pipeline.run_scan import ScanResult, ScanConfig
    from ui.exporter import write_scan

    contract = Contract(
        ticker='AAPL', expiration=date(2026, 7, 18), strike=200.0, option_type='call',
        bid=1.0, ask=1.2, mid=1.1, volume=500, open_interest=1000,
        implied_volatility=0.22, delta=0.50, gamma=0.05, theta=-0.04, vega=0.10,
        dte=21, spot_price=200.0,
    )
    candidate = ScoredCandidate(
        contract=contract, strategy=LongCall(),
        pop_blended=0.55, pop_delta=0.5, pop_bs=0.55, pop_historical=0.55, pop_garch_mc=0.55,
        stress=StressResult(stress_1sd=-0.5, stress_2sd=-1.0, percentile_5=-1.5),
        ev=0.2, max_adverse_loss=1.0, margin_estimate=110.0,
        filter_result=FilterResult(passed=True, failed_filters=[]),
        composite_score=68.0, label='TRADE',
        reason_for='test', reason_against='',
        setup_name='compression_breakout', setup_direction='bullish', setup_strength=0.82,
    )
    result = ScanResult(timestamp=datetime.now(), config=ScanConfig(),
                        candidates=[candidate], skipped={})
    csv_path, _ = write_scan(result, tmp_path)

    text = csv_path.read_text()
    assert 'setup_name' in text
    assert 'compression_breakout' in text
```

(Adjust constructor args to match actual signatures if any field differs.)

- [ ] **Step 4: Run**

```bash
pytest tests/ui/test_exporter.py -v
```

Expected: PASS.

- [ ] **Step 5: Add Setup column + filter widget to the dashboard**

Edit `ui/dashboard.py`. Where the candidate DataFrame is constructed for display, ensure `setup_name`, `setup_direction`, `setup_strength` appear. Add a sidebar widget:

```python
    setup_options = sorted({c.get('setup_name', '') for c in candidates_data if c.get('setup_name')})
    selected_setup = st.sidebar.selectbox('Filter by setup', ['(any)'] + setup_options)
    if selected_setup != '(any)':
        candidates_data = [c for c in candidates_data if c.get('setup_name') == selected_setup]
```

(Exact integration depends on existing dashboard code shape — match the established pattern.)

- [ ] **Step 6: Manual dashboard smoke test**

```bash
streamlit run ui/dashboard.py
```

Open `http://localhost:8501` and verify the Setup column shows up and the sidebar filter works. Ctrl-C when done.

- [ ] **Step 7: Commit**

```bash
git add ui/exporter.py ui/dashboard.py tests/ui/test_exporter.py
git commit -m "feat(ui): expose setup_name/direction/strength in CSV, JSON, dashboard"
```

---

## Task 13: Regression test for run_scan with technical filter

**Files:**
- Modify: `tests/pipeline/test_run_scan.py`

- [ ] **Step 1: Add a fixture that produces a bullish-setup OHLCV history**

Edit `tests/pipeline/test_run_scan.py`. Append:

```python
def _bullish_ohlcv_history() -> pd.DataFrame:
    import numpy as np
    np.random.seed(7)
    base = np.linspace(80, 100, 220)
    flat = np.full(30, 100.0) + np.random.normal(0, 0.05, 30)
    close = np.concatenate([base, flat])
    s = pd.Series(close)
    return pd.DataFrame({
        'open':   s.shift(1).fillna(s.iloc[0]),
        'high':   s * 1.003,
        'low':    s * 0.997,
        'close':  s,
        'volume': pd.Series(1_000_000, index=s.index),
    })


def _flat_ohlcv_history() -> pd.DataFrame:
    import numpy as np
    close = pd.Series(np.full(250, 100.0) + np.random.RandomState(0).normal(0, 0.3, 250))
    return pd.DataFrame({
        'open':   close.shift(1).fillna(close.iloc[0]),
        'high':   close * 1.003,
        'low':    close * 0.997,
        'close':  close,
        'volume': pd.Series(1_000_000, index=close.index),
    })
```

- [ ] **Step 2: Extend `_FakeSource` to serve OHLCV histories**

Edit `_FakeSource` in `tests/pipeline/test_run_scan.py`:

```python
class _FakeSource(DataSource):
    def __init__(self, history, spot, chain, ohlcv=None):
        self._history = history
        self._spot = spot
        self._chain = chain
        self._ohlcv = ohlcv
    def fetch_spot(self, ticker): return self._spot
    def fetch_price_history(self, ticker, lookback_days): return self._history
    def fetch_price_history_ohlcv(self, ticker, lookback_days):
        if self._ohlcv is None:
            raise RuntimeError("no ohlcv configured")
        return self._ohlcv
    def fetch_option_chain(self, ticker): return self._chain
```

- [ ] **Step 3: Write the technical-filter regression test**

Append:

```python
def test_run_scan_with_technical_filter_only_fires_aligned_strategies():
    bull_df = _bullish_ohlcv_history()
    chain = [
        _make_raw(option_type='put', strike=98, mid=1.0, iv=0.22),
        _make_raw(option_type='call', strike=102, mid=1.0, iv=0.22),
    ]
    src = _FakeSource(
        history=bull_df['close'], spot=100.0, chain=chain, ohlcv=bull_df,
    )

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=True),
            sources_override=[src],
        )

    # Bullish setup → LongCall (and NakedPut) eligible; LongPut / NakedCall excluded.
    strategy_names = {c.strategy.name for c in result.candidates}
    assert 'long_put' not in strategy_names
    assert 'naked_call' not in strategy_names
    if strategy_names:
        assert strategy_names.issubset({'long_call', 'naked_put'})
    for c in result.candidates:
        assert c.setup_name == 'compression_breakout'
        assert c.setup_direction == 'bullish'


def test_run_scan_with_no_signal_ticker_produces_no_candidates_when_filter_on():
    flat_df = _flat_ohlcv_history()
    chain = [_make_raw(option_type='put', strike=98, mid=1.0)]
    src = _FakeSource(history=flat_df['close'], spot=100.0, chain=chain, ohlcv=flat_df)

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(
            ScanConfig(today=date(2026, 5, 30), use_technical_filter=True),
            sources_override=[src],
        )

    assert result.candidates == []
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest tests/pipeline/test_run_scan.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run the entire suite to confirm nothing regressed**

```bash
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 6: Final commit**

```bash
git add tests/pipeline/test_run_scan.py
git commit -m "test(pipeline): regression coverage for technical-filter strategy gating"
```

---

## Spec coverage verification

| Spec section / requirement | Implemented by |
|---|---|
| `domain/technical_signals.py` module + TechnicalSetup + helpers | Task 2 |
| Detector 1 — Compression breakout (bull + bear) | Task 3 |
| Detector 2 — Pullback in trend (bull + bear) | Task 4 |
| Detector 3 — Stage 2 breakout / Stage 4 breakdown | Task 5 |
| Detector 4 — Failed breakdown / breakout reversal | Task 6 |
| `detect_setups` aggregator + edge cases | Task 7 |
| `pipeline/technical_filter.py` | Task 8 |
| `ScoredCandidate` setup fields | Task 9 |
| `ScanConfig.use_technical_filter` | Task 9 |
| `run_scan.py` filter insertion + strategy gate | Task 10 |
| `--no-technical-filter` CLI flag | Task 11 |
| `composite_score` weights unchanged | Verified by Tasks 9, 10 not modifying `composite_score` |
| Output additions (CSV, JSON, dashboard) | Task 12 |
| Regression tests | Tasks 9, 10, 12, 13 |
| OHLCV fetch source-interface extension (gap discovered after spec) | Task 1 |

The OHLCV interface extension (Task 1) was not in the spec because the spec assumed history was already OHLCV — it isn't (it's close-only `pd.Series`). Implementing the spec requires this prerequisite, hence the new task.

## Notes for the engineer

- **Run the full test suite (`pytest -q`) after every task.** Existing tests (~95 of them) must keep passing.
- **Frequent commits.** One commit per task minimum. The plan suggests commit messages but tweak as you go.
- **If a detector fixture fails to produce the expected setup**, adjust the synthetic OHLCV — don't loosen the detector thresholds. The thresholds are locked per the spec.
- **The `ScoredCandidate` field additions are positional defaults**, so they don't break existing constructor call sites. If you find a positional construction that breaks, switch it to keyword args rather than reordering fields.
- **The Streamlit dashboard test is light** — Streamlit code is hard to unit-test cleanly. Manual smoke test by running `streamlit run ui/dashboard.py` after Task 12.
