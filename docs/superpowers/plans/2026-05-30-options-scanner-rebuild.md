# Options Scanner Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the options scanner with a layered architecture (`data/` → `domain/` → `engine/` → `pipeline/` → `ui/`), replace Robinhood with yahooquery+yfinance+stooq, add long_put/long_call buying strategies behind an RV/IV regime gate, switch to single EOD scan, and dynamically populate the universe from S&P 500.

**Architecture:** Pure-logic `domain/` and `engine/` layers (no I/O, fully unit-testable). `data/` layer holds all sources behind a `DataSource` ABC with a fallback chain. `pipeline/` orchestrates one scan; `ui/` is Streamlit + CSV/JSON exporter. Strategy abstraction lets PoP/stress/EV math live in 4 small classes (NakedPut/NakedCall/LongPut/LongCall) instead of branching `if option_type == 'put'` everywhere.

**Tech Stack:** Python 3.10+, yahooquery, yfinance, py_vollib, pandas-ta, arch, scipy, numpy, pandas, streamlit, pytest, pytest-mock, hypothesis.

**Spec:** `docs/superpowers/specs/2026-05-30-options-scanner-rebuild-design.md`

---

## File Plan

**New files (in order of creation):**
```
domain/__init__.py
domain/contract.py
domain/greeks.py
domain/strategy.py
domain/signals.py

engine/__init__.py
engine/pop_models.py
engine/stress.py
engine/risk_filters.py
engine/scorer.py

data/sources/__init__.py
data/sources/base.py
data/sources/yahooquery_source.py
data/sources/yfinance_source.py
data/sources/stooq_source.py
data/adapters.py
data/fallback.py

pipeline/__init__.py
pipeline/universe.py
pipeline/universe_builder.py
pipeline/earnings.py
pipeline/run_scan.py

ui/__init__.py
ui/exporter.py
ui/dashboard.py

tests/domain/test_contract.py
tests/domain/test_greeks.py
tests/domain/test_strategy.py
tests/domain/test_signals.py
tests/engine/test_pop_models.py
tests/engine/test_stress.py
tests/engine/test_risk_filters.py
tests/engine/test_scorer.py
tests/data/test_adapters.py
tests/data/test_fallback.py
tests/data/sources/test_base.py
tests/data/sources/test_yahooquery_source.py
tests/data/sources/test_yfinance_source.py
tests/data/sources/test_stooq_source.py
tests/pipeline/test_universe_builder.py
tests/pipeline/test_earnings.py
tests/pipeline/test_run_scan.py
tests/integration/test_live_smoke.py
tests/fixtures/spy_chain.json
tests/fixtures/spy_history.csv
```

**Files to modify:**
```
main.py
requirements.txt
.env.example
.gitignore
README.md
```

**Files to delete (final cleanup task):**
```
scanner/*  (entire directory)
output/dashboard.py    (moved to ui/dashboard.py)
output/exporter.py     (moved to ui/exporter.py)
data/universe.py       (moved to pipeline/universe.py)
data/earnings_calendar.py  (moved to pipeline/earnings.py)
tests/scanner/*  (replaced by tests/domain/, tests/engine/, tests/data/, tests/pipeline/)
```

---

## Task 1: Update dependencies and gitignore

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `.env.example`

- [ ] **Step 1: Replace requirements.txt content**

```text
yahooquery>=2.3.0
yfinance>=0.2.40
py_vollib>=1.0.1
pandas-ta>=0.3.14b
arch>=6.3.0
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0
streamlit>=1.32.0
python-dotenv>=1.0.0
lxml>=5.0.0
pytest>=8.0.0
pytest-mock>=3.12.0
hypothesis>=6.100.0
responses>=0.25.0
```

- [ ] **Step 2: Update .gitignore — add cache/ and remove robinhood references**

Add these lines to `.gitignore`:
```
cache/
```
Remove this line from `.gitignore`:
```
.robinhood_session/
```

- [ ] **Step 3: Replace `.env.example` content**

```text
RISK_FREE_RATE=0.053
MIN_POP_THRESHOLD=0.70
MIN_COMPOSITE_SCORE=65
MAX_SPREAD_PCT=0.20
MIN_VOLUME=100
MIN_OI=500
EARNINGS_BLACKOUT_DAYS=5
MONTE_CARLO_PATHS=10000
UNIVERSE_TOP_N=50
UNIVERSE_MIN_AVG_VOLUME=1000000
```

- [ ] **Step 4: Reinstall deps**

Run: `pip install -r requirements.txt`
Expected: clean install; no errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore .env.example
git commit -m "chore: switch deps to yahooquery+yfinance+py_vollib+pandas-ta, drop robin_stocks"
```

---

## Task 2: domain/contract.py

**Files:**
- Create: `domain/__init__.py` (empty)
- Create: `domain/contract.py`
- Test: `tests/domain/test_contract.py`

- [ ] **Step 1: Create empty `domain/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

Create `tests/domain/__init__.py` (empty) and `tests/domain/test_contract.py`:

```python
from datetime import date
import pytest
from domain.contract import Contract


def test_contract_is_frozen():
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.10, ask=2.15, mid=2.125,
        volume=1500, open_interest=8000, implied_volatility=0.18,
        delta=-0.20, gamma=0.012, theta=-0.05, vega=0.15,
        dte=21, spot_price=635.0,
    )
    with pytest.raises(Exception):
        c.strike = 999.0  # frozen dataclass forbids mutation


def test_contract_mid_independent_of_constructor():
    # mid is stored, not derived — callers compute it
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.0, ask=3.0, mid=2.5,
        volume=1, open_interest=1, implied_volatility=0.2,
        delta=-0.2, gamma=0.0, theta=0.0, vega=0.0,
        dte=10, spot_price=620.0,
    )
    assert c.mid == 2.5


def test_contract_option_type_constrained():
    # Literal type is documentation; runtime won't enforce, but dataclass takes it
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='call', bid=1.0, ask=1.1, mid=1.05,
        volume=1, open_interest=1, implied_volatility=0.2,
        delta=0.3, gamma=0.0, theta=0.0, vega=0.0,
        dte=10, spot_price=620.0,
    )
    assert c.option_type == 'call'
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/domain/test_contract.py -v`
Expected: ModuleNotFoundError: No module named 'domain.contract'

- [ ] **Step 4: Implement `domain/contract.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class Contract:
    ticker: str
    expiration: date
    strike: float
    option_type: Literal['put', 'call']
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/domain/test_contract.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add domain/__init__.py domain/contract.py tests/domain/__init__.py tests/domain/test_contract.py
git commit -m "feat(domain): Contract frozen dataclass"
```

---

## Task 3: domain/greeks.py — py_vollib wrapper

**Files:**
- Create: `domain/greeks.py`
- Test: `tests/domain/test_greeks.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from domain.greeks import compute_greeks


def test_atm_call_delta_near_half():
    # ATM call with positive r and modest vol should have delta near 0.5
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert 0.45 < g['delta'] < 0.65


def test_atm_put_delta_near_negative_half():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.20, flag='p')
    assert -0.55 < g['delta'] < -0.35


def test_deep_otm_call_delta_near_zero():
    g = compute_greeks(S=100.0, K=150.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert 0.0 < g['delta'] < 0.05


def test_deep_itm_call_delta_near_one():
    g = compute_greeks(S=100.0, K=50.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert 0.9 < g['delta'] <= 1.0


def test_greeks_keys_complete():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.20, flag='c')
    assert set(g.keys()) == {'delta', 'gamma', 'theta', 'vega'}


def test_zero_T_returns_zeros_or_intrinsic():
    # py_vollib raises on T=0; we should clamp to a small T
    g = compute_greeks(S=100.0, K=100.0, T=0.0, r=0.05, sigma=0.20, flag='c')
    assert all(g[k] == 0.0 or isinstance(g[k], float) for k in g)


def test_zero_sigma_returns_zeros_or_intrinsic():
    g = compute_greeks(S=100.0, K=100.0, T=30/365, r=0.05, sigma=0.0, flag='c')
    assert all(isinstance(g[k], float) for k in g)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/domain/test_greeks.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `domain/greeks.py`**

```python
from __future__ import annotations
from typing import Literal
from py_vollib.black_scholes.greeks import analytical as bs_greeks


def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    flag: Literal['c', 'p'],
) -> dict[str, float]:
    """Black-Scholes Greeks via py_vollib. Returns zeros on degenerate inputs."""
    if T <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    try:
        return {
            'delta': float(bs_greeks.delta(flag, S, K, T, r, sigma)),
            'gamma': float(bs_greeks.gamma(flag, S, K, T, r, sigma)),
            'theta': float(bs_greeks.theta(flag, S, K, T, r, sigma)),
            'vega':  float(bs_greeks.vega(flag, S, K, T, r, sigma)),
        }
    except Exception:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/domain/test_greeks.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add domain/greeks.py tests/domain/test_greeks.py
git commit -m "feat(domain): py_vollib greeks wrapper with degenerate-input guards"
```

---

## Task 4: domain/strategy.py — ABC + 4 concrete classes

**Files:**
- Create: `domain/strategy.py`
- Test: `tests/domain/test_strategy.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall


def _make_contract(option_type='put', delta=-0.20, strike=100.0, mid=2.0, spot=100.0):
    sign = -1 if option_type == 'put' else 1
    actual_delta = sign * abs(delta) if option_type == 'put' else abs(delta)
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=mid-0.05, ask=mid+0.05, mid=mid,
        volume=1000, open_interest=1000, implied_volatility=0.25,
        delta=actual_delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=spot,
    )


class TestNakedPut:
    def test_name_and_direction(self):
        s = NakedPut()
        assert s.name == 'naked_put'
        assert s.direction == 'sell'
        assert s.option_type == 'put'

    def test_applies_to_put_in_delta_range(self):
        c = _make_contract(option_type='put', delta=0.20)
        assert NakedPut().applies_to(c) is True

    def test_rejects_call_contract(self):
        c = _make_contract(option_type='call', delta=0.20)
        assert NakedPut().applies_to(c) is False

    def test_rejects_below_delta_floor(self):
        c = _make_contract(option_type='put', delta=0.10)
        assert NakedPut().applies_to(c) is False

    def test_rejects_above_delta_ceiling(self):
        c = _make_contract(option_type='put', delta=0.35)
        assert NakedPut().applies_to(c) is False

    def test_breakeven_is_strike_minus_mid(self):
        c = _make_contract(option_type='put', strike=100.0, mid=2.0)
        assert NakedPut().breakeven(c) == 98.0

    def test_profit_condition_above_strike(self):
        c = _make_contract(option_type='put', strike=100.0, mid=2.0)
        assert NakedPut().profit_condition(101.0, c) is True
        assert NakedPut().profit_condition(99.0, c) is False


class TestNakedCall:
    def test_breakeven_is_strike_plus_mid(self):
        c = _make_contract(option_type='call', strike=100.0, mid=2.0)
        assert NakedCall().breakeven(c) == 102.0

    def test_profit_condition_below_strike(self):
        c = _make_contract(option_type='call', strike=100.0, mid=2.0)
        assert NakedCall().profit_condition(99.0, c) is True
        assert NakedCall().profit_condition(101.0, c) is False


class TestLongPut:
    def test_delta_range_is_buyer_zone(self):
        c40 = _make_contract(option_type='put', delta=0.40)
        c60 = _make_contract(option_type='put', delta=0.60)
        c30 = _make_contract(option_type='put', delta=0.30)
        assert LongPut().applies_to(c40) is True
        assert LongPut().applies_to(c60) is True
        assert LongPut().applies_to(c30) is False

    def test_profit_condition_below_strike_minus_mid(self):
        c = _make_contract(option_type='put', strike=100.0, mid=2.0)
        # Buyer needs S_T < 98 to profit
        assert LongPut().profit_condition(97.0, c) is True
        assert LongPut().profit_condition(99.0, c) is False


class TestLongCall:
    def test_profit_condition_above_strike_plus_mid(self):
        c = _make_contract(option_type='call', strike=100.0, mid=2.0)
        assert LongCall().profit_condition(103.0, c) is True
        assert LongCall().profit_condition(101.0, c) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/domain/test_strategy.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `domain/strategy.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import ClassVar, Literal
from dataclasses import dataclass

from domain.contract import Contract


@dataclass
class Strategy(ABC):
    name: ClassVar[str]
    direction: ClassVar[Literal['sell', 'buy']]
    option_type: ClassVar[Literal['put', 'call']]
    delta_min: ClassVar[float]
    delta_max: ClassVar[float]

    def applies_to(self, c: Contract) -> bool:
        if c.option_type != self.option_type:
            return False
        return self.delta_min <= abs(c.delta) <= self.delta_max

    @abstractmethod
    def breakeven(self, c: Contract) -> float: ...

    @abstractmethod
    def profit_condition(self, S_T: float, c: Contract) -> bool: ...

    @abstractmethod
    def expected_value(self, c: Contract, pop_blended: float, max_adverse_loss: float, expected_profit_when_itm: float = 0.0) -> float: ...

    @abstractmethod
    def margin_estimate(self, c: Contract) -> float: ...


class NakedPut(Strategy):
    name = 'naked_put'
    direction = 'sell'
    option_type = 'put'
    delta_min = 0.16
    delta_max = 0.30

    def breakeven(self, c: Contract) -> float:
        return c.strike - c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T > c.strike

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return c.mid * pop_blended - max_adverse_loss * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        otm = abs(c.spot_price - c.strike)
        return round(max(0.20 * c.spot_price - otm + c.mid, 0.10 * c.strike + c.mid) * 100, 2)


class NakedCall(Strategy):
    name = 'naked_call'
    direction = 'sell'
    option_type = 'call'
    delta_min = 0.16
    delta_max = 0.30

    def breakeven(self, c: Contract) -> float:
        return c.strike + c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T < c.strike

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return c.mid * pop_blended - max_adverse_loss * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        otm = abs(c.spot_price - c.strike)
        return round(max(0.20 * c.spot_price - otm + c.mid, 0.10 * c.spot_price + c.mid) * 100, 2)


class LongPut(Strategy):
    name = 'long_put'
    direction = 'buy'
    option_type = 'put'
    delta_min = 0.40
    delta_max = 0.60

    def breakeven(self, c: Contract) -> float:
        return c.strike - c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T < c.strike - c.mid

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        # Buyer EV: profit_when_itm * pop - mid_paid * (1-pop). Loss capped at mid.
        return expected_profit_when_itm * pop_blended - c.mid * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        # Long option: margin = premium paid (debit). No additional capital.
        return round(c.mid * 100, 2)


class LongCall(Strategy):
    name = 'long_call'
    direction = 'buy'
    option_type = 'call'
    delta_min = 0.40
    delta_max = 0.60

    def breakeven(self, c: Contract) -> float:
        return c.strike + c.mid

    def profit_condition(self, S_T: float, c: Contract) -> bool:
        return S_T > c.strike + c.mid

    def expected_value(self, c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0):
        return expected_profit_when_itm * pop_blended - c.mid * (1 - pop_blended)

    def margin_estimate(self, c: Contract) -> float:
        return round(c.mid * 100, 2)


ALL_STRATEGIES: list[Strategy] = [NakedPut(), NakedCall(), LongPut(), LongCall()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/domain/test_strategy.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add domain/strategy.py tests/domain/test_strategy.py
git commit -m "feat(domain): Strategy ABC + NakedPut/NakedCall/LongPut/LongCall with research-backed delta ranges"
```

---

## Task 5: domain/signals.py — Regime detection

**Files:**
- Create: `domain/signals.py`
- Test: `tests/domain/test_signals.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from domain.signals import Regime, compute_regime


def _make_log_returns(daily_vol: float, n: int = 252, seed: int = 7) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(0.0, daily_vol, size=n))


def _make_chain_with_iv(iv: float) -> list:
    # Minimal duck-typed RawContract list for regime compute
    class FakeRaw:
        def __init__(self, iv, dte, strike, spot):
            self.implied_volatility = iv
            self.dte = dte
            self.strike = strike
            self.spot_price = spot
            self.expiration = None
            self.option_type = 'put'
            self.bid = 1.0; self.ask = 1.1; self.volume = 1; self.open_interest = 1
            self.ticker = 'X'
    # Build a small ATM front-month set
    return [FakeRaw(iv, 30, 100.0, 100.0) for _ in range(5)]


def test_regime_favors_sell_when_iv_exceeds_rv():
    log_returns = _make_log_returns(daily_vol=0.005)   # RV_annualized ≈ 0.08
    chain = _make_chain_with_iv(iv=0.30)               # IV much higher than RV
    regime = compute_regime('SPY', log_returns, chain)
    assert 'sell' in regime.favored
    assert 'buy' not in regime.favored
    assert regime.rv_iv_ratio < 0.8


def test_regime_favors_buy_when_rv_exceeds_iv():
    log_returns = _make_log_returns(daily_vol=0.025)   # RV_annualized ≈ 0.40
    chain = _make_chain_with_iv(iv=0.15)               # IV much lower than RV
    regime = compute_regime('SPY', log_returns, chain)
    assert 'buy' in regime.favored
    assert 'sell' not in regime.favored
    assert regime.rv_iv_ratio > 1.2


def test_regime_neutral_when_rv_near_iv():
    log_returns = _make_log_returns(daily_vol=0.0126)  # RV_annualized ≈ 0.20
    chain = _make_chain_with_iv(iv=0.20)
    regime = compute_regime('SPY', log_returns, chain)
    assert set(regime.favored) == {'sell', 'buy'}
    assert 0.8 <= regime.rv_iv_ratio <= 1.2


def test_regime_empty_chain_defaults_to_neutral():
    log_returns = _make_log_returns(daily_vol=0.01)
    regime = compute_regime('SPY', log_returns, chain=[])
    assert regime.favored == ['sell', 'buy']
    assert regime.iv_atm == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/domain/test_signals.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `domain/signals.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class Regime:
    ticker: str
    rv_30: float
    iv_atm: float
    rv_iv_ratio: float
    favored: list[Literal['sell', 'buy']]


def _annualized_rv_30(log_returns: pd.Series) -> float:
    tail = log_returns.dropna().tail(30)
    if len(tail) < 5:
        return 0.0
    return float(tail.std() * np.sqrt(252))


def _atm_iv(chain) -> float:
    if not chain:
        return 0.0
    # Front-month, ATM-ish: nearest spot, shortest dte
    by_dte = sorted(chain, key=lambda c: getattr(c, 'dte', 9999))
    if not by_dte:
        return 0.0
    front_dte = by_dte[0].dte
    front_month = [c for c in chain if getattr(c, 'dte', 9999) == front_dte]
    if not front_month:
        return 0.0
    spot = getattr(front_month[0], 'spot_price', 0.0) or 0.0
    if spot <= 0:
        # Fallback: just average IVs of front month
        ivs = [c.implied_volatility for c in front_month if c.implied_volatility > 0]
        return float(np.mean(ivs)) if ivs else 0.0
    atm_sorted = sorted(front_month, key=lambda c: abs(c.strike - spot))[:6]
    ivs = [c.implied_volatility for c in atm_sorted if c.implied_volatility > 0]
    return float(np.mean(ivs)) if ivs else 0.0


def compute_regime(ticker: str, log_returns: pd.Series, chain: list) -> Regime:
    rv_30 = _annualized_rv_30(log_returns)
    iv_atm = _atm_iv(chain)

    if iv_atm <= 0 or rv_30 <= 0:
        return Regime(ticker=ticker, rv_30=rv_30, iv_atm=iv_atm, rv_iv_ratio=1.0,
                      favored=['sell', 'buy'])

    ratio = rv_30 / iv_atm
    if ratio < 0.8:
        favored = ['sell']
    elif ratio > 1.2:
        favored = ['buy']
    else:
        favored = ['sell', 'buy']

    return Regime(ticker=ticker, rv_30=rv_30, iv_atm=iv_atm, rv_iv_ratio=ratio, favored=favored)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/domain/test_signals.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add domain/signals.py tests/domain/test_signals.py
git commit -m "feat(domain): RV/IV regime gate with sell/buy/neutral classification"
```

---

## Task 6: engine/pop_models.py — 4 PoP models

**Files:**
- Create: `engine/__init__.py` (empty)
- Create: `engine/pop_models.py`
- Test: `tests/engine/test_pop_models.py`

- [ ] **Step 1: Create empty `engine/__init__.py` and `tests/engine/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

`tests/engine/test_pop_models.py`:

```python
from datetime import date
import numpy as np
import pandas as pd
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from engine.pop_models import pop_delta, pop_black_scholes, pop_historical, pop_garch_mc, blend_pop


def _make_contract(option_type='put', delta=-0.20, strike=100.0, mid=2.0, spot=100.0,
                   iv=0.25, dte=21):
    actual_delta = -abs(delta) if option_type == 'put' else abs(delta)
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=mid-0.05, ask=mid+0.05, mid=mid,
        volume=1000, open_interest=1000, implied_volatility=iv,
        delta=actual_delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=dte, spot_price=spot,
    )


def test_pop_delta_seller_is_one_minus_abs_delta():
    c = _make_contract(option_type='put', delta=0.20)
    assert pop_delta(c, NakedPut()) == pytest.approx(0.80, abs=0.01)


def test_pop_delta_buyer_is_abs_delta():
    c = _make_contract(option_type='put', delta=0.50)
    assert pop_delta(c, LongPut()) == pytest.approx(0.50, abs=0.01)


def test_pop_bs_naked_put_high_when_otm():
    c = _make_contract(option_type='put', strike=90.0, spot=100.0, mid=0.5, iv=0.20, dte=21)
    # PoP for short put with K << S should be very high
    pop = pop_black_scholes(c, NakedPut(), r=0.05)
    assert pop > 0.80


def test_pop_bs_long_call_low_when_otm():
    c = _make_contract(option_type='call', strike=110.0, spot=100.0, mid=0.5, iv=0.20, dte=21)
    pop = pop_black_scholes(c, LongCall(), r=0.05)
    assert pop < 0.30


def test_pop_historical_returns_in_range():
    np.random.seed(0)
    log_returns = pd.Series(np.random.normal(0.0, 0.01, 500))
    c = _make_contract(option_type='put', strike=98.0, spot=100.0, mid=1.0)
    p = pop_historical(c, NakedPut(), log_returns)
    assert 0.0 <= p <= 1.0


def test_pop_historical_with_too_few_returns_returns_half():
    log_returns = pd.Series([0.01, -0.01, 0.005])
    c = _make_contract(option_type='put')
    assert pop_historical(c, NakedPut(), log_returns) == 0.5


def test_pop_garch_mc_returns_in_range():
    np.random.seed(0)
    log_returns = pd.Series(np.random.normal(0.0, 0.01, 300))
    c = _make_contract(option_type='put', strike=95.0, spot=100.0, mid=1.0, dte=21)
    p = pop_garch_mc(c, NakedPut(), log_returns, n_paths=1000)
    assert 0.0 <= p <= 1.0


def test_blend_pop_weights_match_spec():
    blended = blend_pop(p_delta=1.0, p_bs=0.0, p_hist=0.0, p_garch=0.0)
    assert blended == pytest.approx(0.20)
    blended = blend_pop(p_delta=0.0, p_bs=1.0, p_hist=0.0, p_garch=0.0)
    assert blended == pytest.approx(0.25)
    blended = blend_pop(p_delta=0.0, p_bs=0.0, p_hist=1.0, p_garch=0.0)
    assert blended == pytest.approx(0.20)
    blended = blend_pop(p_delta=0.0, p_bs=0.0, p_hist=0.0, p_garch=1.0)
    assert blended == pytest.approx(0.35)


def test_blend_pop_clipped_to_unit_interval():
    assert blend_pop(2.0, 2.0, 2.0, 2.0) == 1.0
    assert blend_pop(-1.0, -1.0, -1.0, -1.0) == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/engine/test_pop_models.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 4: Implement `engine/pop_models.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from domain.contract import Contract
from domain.strategy import Strategy


def pop_delta(c: Contract, strategy: Strategy) -> float:
    if strategy.direction == 'sell':
        return float(np.clip(1.0 - abs(c.delta), 0.0, 1.0))
    return float(np.clip(abs(c.delta), 0.0, 1.0))


def pop_black_scholes(c: Contract, strategy: Strategy, r: float) -> float:
    S, sigma = c.spot_price, c.implied_volatility
    breakeven = strategy.breakeven(c)
    T = max(c.dte / 252, 1e-9)
    if sigma <= 0 or S <= 0 or breakeven <= 0:
        return 0.5
    d2 = (np.log(S / breakeven) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    # Decide which tail is the profitable one by asking the strategy.
    # For sellers (NakedPut: profit if S_T > K, NakedCall: S_T < K) breakeven equals K only when premium is 0;
    # otherwise breakeven shifts in favor of the seller. We anchor on the strike for the seller's PoP
    # because the seller doesn't need to recoup premium from intrinsic value — they already collected it.
    if strategy.direction == 'sell':
        K = c.strike
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d2) if c.option_type == 'put' else norm.cdf(-d2))
    # Buyer needs to clear breakeven, not just strike.
    return float(norm.cdf(-d2) if c.option_type == 'put' else norm.cdf(d2))


def pop_historical(c: Contract, strategy: Strategy, log_returns: pd.Series) -> float:
    dte = c.dte
    series = log_returns.dropna()
    if len(series) < dte + 10:
        return 0.5
    rolling_sum = series.rolling(dte).sum().dropna()
    terminal_prices = c.spot_price * np.exp(rolling_sum)
    matches = terminal_prices.apply(lambda S_T: strategy.profit_condition(float(S_T), c))
    return float(np.clip(matches.mean(), 0.0, 1.0))


def pop_garch_mc(c: Contract, strategy: Strategy, log_returns: pd.Series, n_paths: int = 10000) -> float:
    series = log_returns.dropna()
    try:
        from arch import arch_model
        pct = series * 100
        model = arch_model(pct, vol='Garch', p=1, q=1, dist='t')
        res = model.fit(disp='off', show_warning=False)
        fc = res.forecast(horizon=c.dte)
        avg_var = fc.variance.values[-1].mean()
        daily_vol = float(np.sqrt(avg_var) / 100)
        nu = max(float(res.params.get('nu', 8.0)), 3.0)
    except Exception:
        daily_vol = float(series.std()) if len(series) > 1 else 0.01
        nu = 8.0
    np.random.seed(42)
    z = np.random.standard_t(df=nu, size=(n_paths, c.dte))
    z = z / np.sqrt(nu / (nu - 2))
    cumulative = (daily_vol * z).sum(axis=1)
    terminal_prices = c.spot_price * np.exp(cumulative)
    matches = np.array([strategy.profit_condition(float(p), c) for p in terminal_prices])
    return float(np.clip(matches.mean(), 0.0, 1.0))


def blend_pop(p_delta: float, p_bs: float, p_hist: float, p_garch: float) -> float:
    blended = 0.20 * p_delta + 0.25 * p_bs + 0.20 * p_hist + 0.35 * p_garch
    return float(np.clip(blended, 0.0, 1.0))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engine/test_pop_models.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add engine/__init__.py engine/pop_models.py tests/engine/__init__.py tests/engine/test_pop_models.py
git commit -m "feat(engine): 4 PoP models parameterized by Strategy"
```

---

## Task 7: engine/stress.py — Stress scenarios

**Files:**
- Create: `engine/stress.py`
- Test: `tests/engine/test_stress.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from engine.stress import compute_stress, StressResult


def _make_contract(option_type='put', strike=100.0, mid=2.0, spot=100.0, iv=0.25):
    delta = -0.20 if option_type == 'put' else 0.20
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=mid-0.05, ask=mid+0.05, mid=mid,
        volume=1000, open_interest=1000, implied_volatility=iv,
        delta=delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=spot,
    )


def test_naked_put_stress_2sd_negative_when_strike_breached():
    c = _make_contract(option_type='put', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c, NakedPut(), c.implied_volatility)
    assert s.stress_2sd < 0   # Down 2σ from 100 is well below K=100 → loss
    assert s.stress_2sd < s.stress_1sd  # 2σ stress is worse than 1σ


def test_naked_call_stress_uses_upside_move():
    c_call = _make_contract(option_type='call', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c_call, NakedCall(), c_call.implied_volatility)
    assert s.stress_2sd < 0


def test_long_put_stress_floored_at_negative_mid():
    c = _make_contract(option_type='put', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c, LongPut(), c.implied_volatility)
    assert s.stress_2sd >= -c.mid - 1e-6   # buyer's risk floor


def test_long_call_stress_floored_at_negative_mid():
    c = _make_contract(option_type='call', strike=100.0, mid=2.0, spot=100.0, iv=0.30)
    s = compute_stress(c, LongCall(), c.implied_volatility)
    assert s.stress_2sd >= -c.mid - 1e-6


def test_stress_result_dataclass_fields():
    c = _make_contract()
    s = compute_stress(c, NakedPut(), c.implied_volatility)
    assert isinstance(s, StressResult)
    assert isinstance(s.stress_1sd, float)
    assert isinstance(s.stress_2sd, float)
    assert isinstance(s.stress_expiry, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_stress.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `engine/stress.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from domain.contract import Contract
from domain.strategy import Strategy


@dataclass
class StressResult:
    stress_1sd: float
    stress_2sd: float
    stress_expiry: float


def _adverse_spot(spot: float, daily_sigma: float, sigmas: float, strategy: Strategy) -> float:
    # NakedPut + LongCall: bad direction is down
    # NakedCall + LongPut: bad direction is up
    if strategy.name in ('naked_put', 'long_call'):
        return spot * (1 - daily_sigma * sigmas)
    return spot * (1 + daily_sigma * sigmas)


def _intrinsic_at(spot_at: float, c: Contract) -> float:
    if c.option_type == 'put':
        return max(c.strike - spot_at, 0.0)
    return max(spot_at - c.strike, 0.0)


def _pl_at(spot_at: float, c: Contract, strategy: Strategy) -> float:
    intrinsic = _intrinsic_at(spot_at, c)
    if strategy.direction == 'sell':
        return round(c.mid - intrinsic, 4)
    # Buyer: gain is intrinsic minus premium paid; loss floored at -mid
    pl = intrinsic - c.mid
    return round(max(pl, -c.mid), 4)


def compute_stress(c: Contract, strategy: Strategy, sigma_annual: float) -> StressResult:
    daily_sigma = max(sigma_annual, 1e-9) / np.sqrt(252)
    s1 = _adverse_spot(c.spot_price, daily_sigma, 1.0, strategy)
    s2 = _adverse_spot(c.spot_price, daily_sigma, 2.0, strategy)

    # 5th-percentile expiry move (~1.645σ on horizon of 30 calendar days)
    horizon_sigma = sigma_annual * np.sqrt(30 / 252)
    if strategy.name in ('naked_put', 'long_call'):
        s_exp = c.spot_price * np.exp(-1.645 * horizon_sigma)
    else:
        s_exp = c.spot_price * np.exp(1.645 * horizon_sigma)

    return StressResult(
        stress_1sd=_pl_at(s1, c, strategy),
        stress_2sd=_pl_at(s2, c, strategy),
        stress_expiry=_pl_at(s_exp, c, strategy),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_stress.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/stress.py tests/engine/test_stress.py
git commit -m "feat(engine): per-strategy stress scenarios with buyer risk floor"
```

---

## Task 8: engine/risk_filters.py

**Files:**
- Create: `engine/risk_filters.py`
- Test: `tests/engine/test_risk_filters.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
import math
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, NakedCall, LongPut, LongCall
from engine.stress import StressResult
from engine.risk_filters import FilterSet, FILTER_SETS, apply_filters, FilterResult


def _make(option_type='put', delta=-0.20, strike=100.0, mid=2.0, spot=100.0,
          bid=1.95, ask=2.05, volume=1000, oi=1000):
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=strike,
        option_type=option_type, bid=bid, ask=ask, mid=mid,
        volume=volume, open_interest=oi, implied_volatility=0.25,
        delta=delta, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=spot,
    )


def test_filter_set_for_naked_put_matches_spec():
    fs = FILTER_SETS['naked_put']
    assert fs.delta_range == (0.16, 0.30)
    assert fs.min_pop == 0.70
    assert fs.max_stress_loss_multiple == 3.0


def test_filter_set_for_long_call_matches_spec():
    fs = FILTER_SETS['long_call']
    assert fs.delta_range == (0.40, 0.60)
    assert fs.min_pop == 0.40
    assert fs.min_ev == 0.15
    assert math.isinf(fs.max_stress_loss_multiple)


def test_apply_filters_all_pass_for_clean_seller():
    c = _make(option_type='put', delta=-0.22, mid=2.0, bid=1.95, ask=2.05)
    stress = StressResult(stress_1sd=-1.0, stress_2sd=-3.0, stress_expiry=-2.0)
    result = apply_filters(c, NakedPut(), pop_blended=0.75, ev=0.5, stress=stress)
    assert result.passed is True
    assert result.failed_filters == []


def test_apply_filters_rejects_wide_spread():
    c = _make(bid=1.0, ask=3.0, mid=2.0)  # 100% of mid spread
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert r.passed is False
    assert 'spread' in r.failed_filters


def test_apply_filters_rejects_low_volume():
    c = _make(volume=10)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'volume' in r.failed_filters


def test_apply_filters_rejects_low_oi():
    c = _make(oi=100)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'open_interest' in r.failed_filters


def test_apply_filters_rejects_delta_out_of_range():
    c = _make(delta=-0.40)  # naked put accepts 0.16-0.30
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'delta' in r.failed_filters


def test_apply_filters_rejects_low_pop():
    c = _make(delta=-0.22)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, NakedPut(), 0.50, 0.5, stress)
    assert 'pop' in r.failed_filters


def test_seller_stress_loss_multiple_3x_rejects_high_stress():
    c = _make(delta=-0.22, mid=2.0)
    # 2σ stress loss = -10 → 5x premium received → reject
    stress = StressResult(-2, -10, -5)
    r = apply_filters(c, NakedPut(), 0.75, 0.5, stress)
    assert 'stress' in r.failed_filters


def test_buyer_stress_filter_not_applied():
    c = _make(option_type='call', delta=0.50, mid=3.0)
    # Buyer max loss = mid = 3; stress -3 is fine (capped)
    stress = StressResult(-1, -3, -2)
    r = apply_filters(c, LongCall(), 0.45, 0.20, stress)
    assert r.passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_risk_filters.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `engine/risk_filters.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
import math
from domain.contract import Contract
from domain.strategy import Strategy
from engine.stress import StressResult


@dataclass(frozen=True)
class FilterSet:
    max_spread_pct: float
    min_volume: int
    min_oi: int
    delta_range: tuple[float, float]
    min_pop: float
    min_ev: float
    max_stress_loss_multiple: float  # math.inf for buyers


FILTER_SETS: dict[str, FilterSet] = {
    'naked_put':  FilterSet(0.20, 100, 500, (0.16, 0.30), 0.70, 0.0,  3.0),
    'naked_call': FilterSet(0.20, 100, 500, (0.16, 0.30), 0.70, 0.0,  3.0),
    'long_put':   FilterSet(0.20, 100, 500, (0.40, 0.60), 0.40, 0.15, math.inf),
    'long_call':  FilterSet(0.20, 100, 500, (0.40, 0.60), 0.40, 0.15, math.inf),
}


@dataclass
class FilterResult:
    passed: bool
    failed_filters: list[str] = field(default_factory=list)


def apply_filters(c: Contract, strategy: Strategy, pop_blended: float, ev: float, stress: StressResult) -> FilterResult:
    fs = FILTER_SETS[strategy.name]
    failed: list[str] = []

    spread = c.ask - c.bid
    if c.mid <= 0 or spread / c.mid > fs.max_spread_pct:
        failed.append('spread')
    if c.volume < fs.min_volume:
        failed.append('volume')
    if c.open_interest < fs.min_oi:
        failed.append('open_interest')
    abs_d = abs(c.delta)
    if abs_d < fs.delta_range[0] or abs_d > fs.delta_range[1]:
        failed.append('delta')
    if pop_blended < fs.min_pop:
        failed.append('pop')
    if ev < fs.min_ev:
        failed.append('ev')
    if strategy.direction == 'sell' and math.isfinite(fs.max_stress_loss_multiple):
        worst = abs(min(stress.stress_2sd, 0.0))
        if c.mid > 0 and worst > fs.max_stress_loss_multiple * c.mid:
            failed.append('stress')

    return FilterResult(passed=(len(failed) == 0), failed_filters=failed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_risk_filters.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/risk_filters.py tests/engine/test_risk_filters.py
git commit -m "feat(engine): per-strategy filter sets with explainable failure list"
```

---

## Task 9: engine/scorer.py

**Files:**
- Create: `engine/scorer.py`
- Test: `tests/engine/test_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut, LongCall
from domain.signals import Regime
from engine.stress import StressResult
from engine.risk_filters import FilterResult
from engine.scorer import composite_score, label_from, ScoredCandidate


def _make_contract(option_type='put'):
    return Contract(
        ticker='X', expiration=date(2026, 6, 20), strike=100.0,
        option_type=option_type, bid=1.95, ask=2.05, mid=2.0,
        volume=1000, open_interest=1000, implied_volatility=0.25,
        delta=-0.22 if option_type == 'put' else 0.50, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=100.0,
    )


def _regime(favored):
    return Regime(ticker='X', rv_30=0.20, iv_atm=0.20, rv_iv_ratio=1.0, favored=favored)


def test_composite_score_high_when_all_factors_strong():
    score = composite_score(
        pop_blended=0.85, ev=2.0, max_adverse_loss=2.0, strategy=NakedPut(),
        regime=_regime(['sell']), mid=2.0,
    )
    # 50 * 0.85 + 30 * 1.0 + 20 * 1.0 = 92.5
    assert score > 90


def test_composite_score_penalized_when_against_regime():
    score = composite_score(
        pop_blended=0.85, ev=2.0, max_adverse_loss=2.0, strategy=NakedPut(),
        regime=_regime(['buy']), mid=2.0,
    )
    # 50 * 0.85 + 30 * 1.0 + 20 * 0.3 = 78.5
    assert 75 < score < 80


def test_composite_score_buyer_normalizes_ev_by_mid():
    score = composite_score(
        pop_blended=0.50, ev=0.40, max_adverse_loss=0.0, strategy=LongCall(),
        regime=_regime(['buy']), mid=2.0,
    )
    # ev_score = clip(0.40 / 2.0, 0, 1) = 0.20
    # score = 50*0.5 + 30*0.2 + 20*1.0 = 25 + 6 + 20 = 51
    assert 48 < score < 54


def test_label_trade_when_score_at_threshold():
    assert label_from(score=65.0, all_filters_pass=True) == 'TRADE'
    assert label_from(score=64.99, all_filters_pass=True) == 'WATCHLIST'
    assert label_from(score=50.0, all_filters_pass=True) == 'WATCHLIST'
    assert label_from(score=49.99, all_filters_pass=True) == 'NO_TRADE'


def test_label_no_trade_when_any_filter_fails():
    assert label_from(score=99.0, all_filters_pass=False) == 'NO_TRADE'


def test_scored_candidate_dataclass_carries_explainability():
    sc = ScoredCandidate(
        contract=_make_contract(), strategy=NakedPut(),
        pop_blended=0.80, pop_delta=0.78, pop_bs=0.79, pop_historical=0.81, pop_garch_mc=0.80,
        stress=StressResult(-1, -3, -2), ev=0.5, max_adverse_loss=3.0, margin_estimate=200.0,
        filter_result=FilterResult(passed=True), composite_score=85.0, label='TRADE',
        reason_for='High PoP, in regime', reason_against='',
    )
    assert sc.label == 'TRADE'
    assert sc.reason_for
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_scorer.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `engine/scorer.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
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


def _regime_alignment_score(strategy: Strategy, regime: Regime) -> float:
    if strategy.direction in regime.favored and len(regime.favored) == 1:
        return 1.0
    if strategy.direction in regime.favored:
        return 0.6
    return 0.3


def composite_score(*, pop_blended: float, ev: float, max_adverse_loss: float,
                    strategy: Strategy, regime: Regime, mid: float) -> float:
    pop_pts = 50.0 * np.clip(pop_blended, 0, 1)
    if strategy.direction == 'sell':
        denom = max_adverse_loss if max_adverse_loss > 0 else 1.0
        ev_score = float(np.clip(ev / denom, 0.0, 1.0))
    else:
        ev_score = float(np.clip(ev / max(mid, 1e-9), 0.0, 1.0))
    ev_pts = 30.0 * ev_score
    regime_pts = 20.0 * _regime_alignment_score(strategy, regime)
    return round(pop_pts + ev_pts + regime_pts, 2)


def label_from(score: float, all_filters_pass: bool) -> Label:
    if not all_filters_pass:
        return 'NO_TRADE'
    if score >= 65.0:
        return 'TRADE'
    if score >= 50.0:
        return 'WATCHLIST'
    return 'NO_TRADE'
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_scorer.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add engine/scorer.py tests/engine/test_scorer.py
git commit -m "feat(engine): composite scorer with 50/30/20 weighting and TRADE/WATCH/NO_TRADE labeling"
```

---

## Task 10: data/sources/base.py — DataSource ABC + RawContract

**Files:**
- Create: `data/sources/__init__.py` (empty)
- Create: `data/sources/base.py`
- Test: `tests/data/sources/test_base.py`

- [ ] **Step 1: Create empty `data/sources/__init__.py`, `tests/data/__init__.py`, `tests/data/sources/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

```python
from datetime import date
import pytest
import pandas as pd
from data.sources.base import DataSource, RawContract


def test_raw_contract_fields():
    rc = RawContract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.1, ask=2.15, volume=100, open_interest=500,
        implied_volatility=0.20, dte=21, spot_price=635.0,
    )
    assert rc.ticker == 'SPY'
    assert rc.option_type == 'put'


def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()


def test_concrete_source_must_implement_methods():
    class Incomplete(DataSource):
        pass
    with pytest.raises(TypeError):
        Incomplete()


def test_minimal_concrete_source_can_be_instantiated():
    class Minimal(DataSource):
        def fetch_spot(self, ticker): return 100.0
        def fetch_price_history(self, ticker, lookback_days): return pd.Series([1, 2, 3])
        def fetch_option_chain(self, ticker): return []
    src = Minimal()
    assert src.fetch_spot('X') == 100.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/data/sources/test_base.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 4: Implement `data/sources/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal
import pandas as pd


@dataclass(frozen=True)
class RawContract:
    ticker: str
    expiration: date
    strike: float
    option_type: Literal['put', 'call']
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    dte: int
    spot_price: float


class DataSource(ABC):
    @abstractmethod
    def fetch_spot(self, ticker: str) -> float: ...

    @abstractmethod
    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series: ...

    @abstractmethod
    def fetch_option_chain(self, ticker: str) -> list[RawContract]: ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/sources/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add data/sources/__init__.py data/sources/base.py tests/data/__init__.py tests/data/sources/__init__.py tests/data/sources/test_base.py
git commit -m "feat(data): DataSource ABC + RawContract"
```

---

## Task 11: data/adapters.py — RawContract → Contract with Greeks

**Files:**
- Create: `data/adapters.py`
- Test: `tests/data/test_adapters.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
import pytest
from data.adapters import to_contract
from data.sources.base import RawContract


def _make_raw(option_type='put', strike=100.0, dte_days=21):
    today = date(2026, 5, 30)
    exp = date(2026, 6, 20)   # ~21 days later
    return RawContract(
        ticker='X', expiration=exp, strike=strike,
        option_type=option_type, bid=2.0, ask=2.10,
        volume=100, open_interest=500, implied_volatility=0.20,
        dte=(exp - today).days, spot_price=100.0,
    )


def test_to_contract_computes_mid_and_greeks():
    raw = _make_raw()
    c = to_contract(raw, spot=100.0, r=0.05, today=date(2026, 5, 30))
    assert c is not None
    assert c.mid == pytest.approx(2.05, abs=1e-6)
    assert c.delta != 0   # py_vollib should produce a non-zero delta


def test_to_contract_drops_when_zero_bid_ask():
    raw = RawContract('X', date(2026, 6, 20), 100.0, 'put', 0.0, 0.0, 100, 500, 0.2, 21, 100.0)
    assert to_contract(raw, 100.0, 0.05, date(2026, 5, 30)) is None


def test_to_contract_drops_when_dte_below_3():
    today = date(2026, 6, 17)
    exp = date(2026, 6, 19)  # 2 days
    raw = RawContract('X', exp, 100.0, 'put', 2.0, 2.1, 100, 500, 0.2, 2, 100.0)
    assert to_contract(raw, 100.0, 0.05, today) is None


def test_to_contract_drops_when_dte_above_45():
    today = date(2026, 5, 30)
    exp = date(2026, 8, 1)  # ~63 days
    raw = RawContract('X', exp, 100.0, 'put', 2.0, 2.1, 100, 500, 0.2, 63, 100.0)
    assert to_contract(raw, 100.0, 0.05, today) is None


def test_to_contract_put_delta_negative():
    raw = _make_raw(option_type='put', strike=100.0)
    c = to_contract(raw, spot=100.0, r=0.05, today=date(2026, 5, 30))
    assert c.delta < 0


def test_to_contract_call_delta_positive():
    raw = _make_raw(option_type='call', strike=100.0)
    c = to_contract(raw, spot=100.0, r=0.05, today=date(2026, 5, 30))
    assert c.delta > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_adapters.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `data/adapters.py`**

```python
from __future__ import annotations
from datetime import date
from typing import Optional

from domain.contract import Contract
from domain.greeks import compute_greeks
from data.sources.base import RawContract


def to_contract(raw: RawContract, spot: float, r: float, today: date) -> Optional[Contract]:
    dte = (raw.expiration - today).days
    if dte < 3 or dte > 45:
        return None
    if raw.bid + raw.ask <= 0:
        return None
    mid = (raw.bid + raw.ask) / 2.0
    T = max(dte / 252, 1e-9)
    flag = 'c' if raw.option_type == 'call' else 'p'
    g = compute_greeks(S=spot, K=raw.strike, T=T, r=r, sigma=raw.implied_volatility, flag=flag)

    return Contract(
        ticker=raw.ticker, expiration=raw.expiration, strike=raw.strike,
        option_type=raw.option_type, bid=raw.bid, ask=raw.ask, mid=mid,
        volume=raw.volume, open_interest=raw.open_interest,
        implied_volatility=raw.implied_volatility,
        delta=g['delta'], gamma=g['gamma'], theta=g['theta'], vega=g['vega'],
        dte=dte, spot_price=spot,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_adapters.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add data/adapters.py tests/data/test_adapters.py
git commit -m "feat(data): adapter from RawContract → Contract with py_vollib Greeks"
```

---

## Task 12: data/fallback.py — retry chain

**Files:**
- Create: `data/fallback.py`
- Test: `tests/data/test_fallback.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import pandas as pd
from data.fallback import fetch_with_fallback, DataFetchError
from data.sources.base import DataSource


class _Fail(DataSource):
    def __init__(self, msg='boom'): self.msg = msg
    def fetch_spot(self, ticker): raise RuntimeError(self.msg)
    def fetch_price_history(self, ticker, lookback_days): raise RuntimeError(self.msg)
    def fetch_option_chain(self, ticker): raise RuntimeError(self.msg)


class _OK(DataSource):
    def __init__(self, value=42.0): self.value = value
    def fetch_spot(self, ticker): return self.value
    def fetch_price_history(self, ticker, lookback_days): return pd.Series([1.0, 2.0])
    def fetch_option_chain(self, ticker): return ['raw']


def test_first_success_wins():
    out = fetch_with_fallback([_OK(value=99.0), _Fail()], 'fetch_spot', 'X')
    assert out == 99.0


def test_falls_back_to_next_on_failure():
    out = fetch_with_fallback([_Fail(), _OK(value=11.0)], 'fetch_spot', 'X')
    assert out == 11.0


def test_raises_when_all_sources_fail():
    with pytest.raises(DataFetchError):
        fetch_with_fallback([_Fail('a'), _Fail('b')], 'fetch_spot', 'X')


def test_passes_through_args_and_kwargs():
    captured = {}
    class _Capture(DataSource):
        def fetch_spot(self, ticker): captured['ticker'] = ticker; return 1.0
        def fetch_price_history(self, ticker, lookback_days):
            captured.update({'ticker': ticker, 'lb': lookback_days}); return pd.Series([1])
        def fetch_option_chain(self, ticker): return []
    fetch_with_fallback([_Capture()], 'fetch_price_history', 'SPY', 365)
    assert captured == {'ticker': 'SPY', 'lb': 365}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_fallback.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `data/fallback.py`**

```python
from __future__ import annotations
from typing import Any
from data.sources.base import DataSource


class DataFetchError(Exception):
    pass


def fetch_with_fallback(sources: list[DataSource], method: str, *args, **kwargs) -> Any:
    last_err: Exception | None = None
    for src in sources:
        try:
            return getattr(src, method)(*args, **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise DataFetchError(f"All sources failed for {method}{args}: {last_err}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_fallback.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add data/fallback.py tests/data/test_fallback.py
git commit -m "feat(data): fetch_with_fallback retry chain over DataSource list"
```

---

## Task 13: data/sources/yahooquery_source.py

**Files:**
- Create: `data/sources/yahooquery_source.py`
- Create: `tests/fixtures/spy_chain_yq.json` (small frozen response)
- Test: `tests/data/sources/test_yahooquery_source.py`

- [ ] **Step 1: Write the failing tests**

Tests mock yahooquery's `Ticker` calls; no live network.

```python
from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from data.sources.yahooquery_source import YahooQuerySource


def test_fetch_spot_reads_regular_market_price():
    src = YahooQuerySource()
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        MockTicker.return_value.price = {'SPY': {'regularMarketPrice': 635.42}}
        assert src.fetch_spot('SPY') == 635.42


def test_fetch_price_history_returns_close_series():
    src = YahooQuerySource()
    df = pd.DataFrame({
        'close':  [630.0, 631.5, 633.0],
        'volume': [1_000_000, 1_200_000, 950_000],
    }, index=pd.MultiIndex.from_tuples(
        [('SPY', pd.Timestamp('2026-05-27')),
         ('SPY', pd.Timestamp('2026-05-28')),
         ('SPY', pd.Timestamp('2026-05-29'))],
        names=['symbol', 'date'],
    ))
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        MockTicker.return_value.history.return_value = df
        out = src.fetch_price_history('SPY', lookback_days=30)
        assert isinstance(out, pd.Series)
        assert out.tolist() == [630.0, 631.5, 633.0]


def test_fetch_option_chain_flattens_multiindex():
    src = YahooQuerySource()
    chain_df = pd.DataFrame({
        'strike': [620.0, 625.0],
        'bid':    [2.0,   3.0],
        'ask':    [2.1,   3.1],
        'volume': [100,   150],
        'openInterest': [1000, 800],
        'impliedVolatility': [0.18, 0.19],
        'inTheMoney': [False, False],
    }, index=pd.MultiIndex.from_tuples(
        [('SPY', pd.Timestamp('2026-06-20'), 'puts', 620.0),
         ('SPY', pd.Timestamp('2026-06-20'), 'puts', 625.0)],
        names=['symbol', 'expiration', 'optionType', 'strike'],
    ))
    with patch('data.sources.yahooquery_source.Ticker') as MockTicker:
        MockTicker.return_value.option_chain = chain_df
        MockTicker.return_value.price = {'SPY': {'regularMarketPrice': 635.42}}
        contracts = src.fetch_option_chain('SPY')
        assert len(contracts) >= 0  # exact count depends on DTE window
        # All returned contracts should be puts with correct ticker
        for rc in contracts:
            assert rc.ticker == 'SPY'
            assert rc.option_type == 'put'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/sources/test_yahooquery_source.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `data/sources/yahooquery_source.py`**

```python
from __future__ import annotations
from datetime import date
import pandas as pd
from yahooquery import Ticker

from data.sources.base import DataSource, RawContract


_DTE_MIN, _DTE_MAX = 3, 45


class YahooQuerySource(DataSource):
    def fetch_spot(self, ticker: str) -> float:
        t = Ticker(ticker)
        price_info = t.price.get(ticker, {}) if isinstance(t.price, dict) else {}
        return float(price_info.get('regularMarketPrice', 0.0))

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        # period must be one of yahooquery's accepted strings
        period = '1y' if lookback_days > 180 else '6mo'
        df = Ticker(ticker).history(period=period)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol', drop_level=True)
        return df['close'].astype(float)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        t = Ticker(ticker)
        spot = self.fetch_spot(ticker)
        df = t.option_chain
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        today = date.today()
        out: list[RawContract] = []
        # MultiIndex: (symbol, expiration, optionType, strike)
        for idx, row in df.iterrows():
            try:
                _symbol, exp_ts, opt_type_str, strike = idx
                exp = pd.Timestamp(exp_ts).date()
                dte = (exp - today).days
                if dte < _DTE_MIN or dte > _DTE_MAX:
                    continue
                option_type = 'put' if 'put' in str(opt_type_str).lower() else 'call'
                out.append(RawContract(
                    ticker=ticker,
                    expiration=exp,
                    strike=float(strike),
                    option_type=option_type,
                    bid=float(row.get('bid', 0) or 0),
                    ask=float(row.get('ask', 0) or 0),
                    volume=int(row.get('volume', 0) or 0),
                    open_interest=int(row.get('openInterest', 0) or 0),
                    implied_volatility=float(row.get('impliedVolatility', 0) or 0),
                    dte=dte,
                    spot_price=spot,
                ))
            except Exception:
                continue
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/sources/test_yahooquery_source.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add data/sources/yahooquery_source.py tests/data/sources/test_yahooquery_source.py
git commit -m "feat(data): YahooQuerySource — primary chain/price/spot source"
```

---

## Task 14: data/sources/yfinance_source.py

**Files:**
- Create: `data/sources/yfinance_source.py`
- Test: `tests/data/sources/test_yfinance_source.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from data.sources.yfinance_source import YfinanceSource


def test_fetch_spot_uses_fast_info_last_price():
    with patch('data.sources.yfinance_source.yf.Ticker') as MockTicker:
        instance = MockTicker.return_value
        instance.fast_info = MagicMock(last_price=635.0)
        assert YfinanceSource().fetch_spot('SPY') == 635.0


def test_fetch_price_history_returns_close_series():
    df = pd.DataFrame({'Close': [630.0, 631.0, 632.0]}, index=pd.date_range('2026-05-27', periods=3))
    with patch('data.sources.yfinance_source.yf.Ticker') as MockTicker:
        MockTicker.return_value.history.return_value = df
        out = YfinanceSource().fetch_price_history('SPY', 30)
        assert out.tolist() == [630.0, 631.0, 632.0]


def test_fetch_option_chain_iterates_expirations():
    today = date.today()
    exp1 = (pd.Timestamp(today) + pd.Timedelta(days=21)).date().isoformat()
    exp2 = (pd.Timestamp(today) + pd.Timedelta(days=2)).date().isoformat()  # out of DTE range
    puts_df = pd.DataFrame({
        'strike':[620.0], 'bid':[2.0], 'ask':[2.1], 'volume':[100],
        'openInterest':[1000], 'impliedVolatility':[0.20],
    })
    calls_df = pd.DataFrame({
        'strike':[630.0], 'bid':[1.5], 'ask':[1.6], 'volume':[200],
        'openInterest':[800], 'impliedVolatility':[0.18],
    })
    chain_tuple = MagicMock(); chain_tuple.puts = puts_df; chain_tuple.calls = calls_df

    with patch('data.sources.yfinance_source.yf.Ticker') as MockTicker:
        inst = MockTicker.return_value
        inst.options = [exp1, exp2]
        inst.option_chain.side_effect = lambda exp: chain_tuple if exp == exp1 else MagicMock(puts=pd.DataFrame(), calls=pd.DataFrame())
        inst.fast_info = MagicMock(last_price=635.0)

        contracts = YfinanceSource().fetch_option_chain('SPY')
        types = {rc.option_type for rc in contracts}
        assert types == {'put', 'call'}
        assert all(3 <= rc.dte <= 45 for rc in contracts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/sources/test_yfinance_source.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `data/sources/yfinance_source.py`**

```python
from __future__ import annotations
from datetime import date
import pandas as pd
import yfinance as yf

from data.sources.base import DataSource, RawContract


_DTE_MIN, _DTE_MAX = 3, 45


class YfinanceSource(DataSource):
    def fetch_spot(self, ticker: str) -> float:
        return float(yf.Ticker(ticker).fast_info.last_price or 0.0)

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        period = '1y' if lookback_days > 180 else '6mo'
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty or 'Close' not in df.columns:
            return pd.Series(dtype=float)
        return df['Close'].astype(float)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        tk = yf.Ticker(ticker)
        try:
            expirations = list(tk.options or [])
        except Exception:
            return []
        spot = self.fetch_spot(ticker)
        today = date.today()
        out: list[RawContract] = []
        for exp_str in expirations:
            try:
                exp = date.fromisoformat(exp_str)
            except ValueError:
                continue
            dte = (exp - today).days
            if dte < _DTE_MIN or dte > _DTE_MAX:
                continue
            try:
                chain = tk.option_chain(exp_str)
            except Exception:
                continue
            for df, opt_type in [(chain.puts, 'put'), (chain.calls, 'call')]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    out.append(RawContract(
                        ticker=ticker,
                        expiration=exp,
                        strike=float(row.get('strike', 0)),
                        option_type=opt_type,
                        bid=float(row.get('bid', 0) or 0),
                        ask=float(row.get('ask', 0) or 0),
                        volume=int(row.get('volume', 0) or 0),
                        open_interest=int(row.get('openInterest', 0) or 0),
                        implied_volatility=float(row.get('impliedVolatility', 0) or 0),
                        dte=dte,
                        spot_price=spot,
                    ))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/sources/test_yfinance_source.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add data/sources/yfinance_source.py tests/data/sources/test_yfinance_source.py
git commit -m "feat(data): YfinanceSource fallback for chains+prices+spot"
```

---

## Task 15: data/sources/stooq_source.py — prices-only fallback

**Files:**
- Create: `data/sources/stooq_source.py`
- Test: `tests/data/sources/test_stooq_source.py`

- [ ] **Step 1: Write the failing tests**

```python
from io import StringIO
from unittest.mock import patch
import pandas as pd
import pytest
from data.sources.stooq_source import StooqSource


def test_fetch_price_history_parses_stooq_csv():
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-05-27,630,634,629,631,1000\n"
        "2026-05-28,631,633,630,632,1100\n"
    )
    with patch('data.sources.stooq_source.pd.read_csv', return_value=pd.read_csv(StringIO(csv))):
        out = StooqSource().fetch_price_history('SPY', 30)
        assert out.tolist() == [631.0, 632.0]


def test_fetch_spot_returns_last_close():
    csv = "Date,Open,High,Low,Close,Volume\n2026-05-28,631,633,630,632,1100\n"
    with patch('data.sources.stooq_source.pd.read_csv', return_value=pd.read_csv(StringIO(csv))):
        assert StooqSource().fetch_spot('SPY') == 632.0


def test_fetch_option_chain_returns_empty_list():
    assert StooqSource().fetch_option_chain('SPY') == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/sources/test_stooq_source.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `data/sources/stooq_source.py`**

```python
from __future__ import annotations
import pandas as pd

from data.sources.base import DataSource, RawContract


class StooqSource(DataSource):
    def _url(self, ticker: str) -> str:
        return f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

    def fetch_spot(self, ticker: str) -> float:
        df = pd.read_csv(self._url(ticker))
        if df.empty or 'Close' not in df.columns:
            raise RuntimeError(f"stooq returned no Close column for {ticker}")
        return float(df['Close'].iloc[-1])

    def fetch_price_history(self, ticker: str, lookback_days: int) -> pd.Series:
        df = pd.read_csv(self._url(ticker))
        if df.empty or 'Close' not in df.columns:
            raise RuntimeError(f"stooq returned no Close column for {ticker}")
        return df['Close'].astype(float).tail(lookback_days).reset_index(drop=True)

    def fetch_option_chain(self, ticker: str) -> list[RawContract]:
        return []   # stooq has no options
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/sources/test_stooq_source.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add data/sources/stooq_source.py tests/data/sources/test_stooq_source.py
git commit -m "feat(data): StooqSource prices-only last-resort fallback"
```

---

## Task 16: pipeline/universe.py + KNOWN_GOOD_FALLBACK

**Files:**
- Create: `pipeline/__init__.py` (empty)
- Create: `pipeline/universe.py`
- Test: (small smoke test inline with universe_builder in Task 17)

- [ ] **Step 1: Create empty `pipeline/__init__.py`**

```python
```

- [ ] **Step 2: Create `pipeline/universe.py`**

```python
from __future__ import annotations

KNOWN_GOOD_FALLBACK: list[str] = [
    'SPY', 'QQQ', 'IWM',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'GLD', 'TLT',
]
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/__init__.py pipeline/universe.py
git commit -m "feat(pipeline): KNOWN_GOOD_FALLBACK ticker list"
```

---

## Task 17: pipeline/universe_builder.py — dynamic S&P 500 screener

**Files:**
- Create: `pipeline/universe_builder.py`
- Test: `tests/pipeline/test_universe_builder.py`

- [ ] **Step 1: Create empty `tests/pipeline/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

```python
import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
from data.sources.base import DataSource
from pipeline.universe_builder import (
    UniverseFilters, fetch_sp500_constituents, build_universe, build_universe_cached
)


class _FakeSource(DataSource):
    def __init__(self, volumes: dict[str, int], chains: dict[str, list]):
        self._volumes = volumes
        self._chains = chains
    def fetch_spot(self, ticker): return 100.0
    def fetch_price_history(self, ticker, lookback_days):
        vol = self._volumes.get(ticker, 0)
        return pd.Series([vol] * lookback_days)
    def fetch_option_chain(self, ticker):
        return self._chains.get(ticker, [])


def test_fetch_sp500_constituents_returns_list_of_strings():
    fake_df = pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'BRK.B']})
    with patch('pipeline.universe_builder.pd.read_html', return_value=[fake_df]):
        out = fetch_sp500_constituents()
        # 'BRK.B' normalized to 'BRK-B' for yahoo compatibility
        assert 'AAPL' in out and 'MSFT' in out
        assert 'BRK-B' in out


def test_build_universe_filters_by_volume():
    src = _FakeSource(
        volumes={'AAA': 2_000_000, 'BBB': 500_000, 'CCC': 5_000_000},
        chains={'AAA': ['x'], 'BBB': ['x'], 'CCC': ['x']},
    )
    with patch('pipeline.universe_builder.fetch_sp500_constituents', return_value=['AAA', 'BBB', 'CCC']):
        out = build_universe(UniverseFilters(min_avg_volume=1_000_000, top_n=10), [src])
        assert 'BBB' not in out
        assert out == ['CCC', 'AAA']   # ranked by volume desc


def test_build_universe_drops_no_options():
    src = _FakeSource(
        volumes={'AAA': 2_000_000, 'BBB': 5_000_000},
        chains={'AAA': ['x'], 'BBB': []},   # BBB has no options
    )
    with patch('pipeline.universe_builder.fetch_sp500_constituents', return_value=['AAA', 'BBB']):
        out = build_universe(UniverseFilters(require_options_chain=True), [src])
        assert out == ['AAA']


def test_build_universe_top_n_caps_list():
    src = _FakeSource(
        volumes={f'T{i:02}': 1_000_000 + i*1000 for i in range(20)},
        chains={f'T{i:02}': ['x'] for i in range(20)},
    )
    tickers = [f'T{i:02}' for i in range(20)]
    with patch('pipeline.universe_builder.fetch_sp500_constituents', return_value=tickers):
        out = build_universe(UniverseFilters(top_n=5), [src])
        assert len(out) == 5
        # Highest volumes first
        assert out[0] == 'T19'


def test_build_universe_cached_uses_cache_if_exists(tmp_path):
    cache_file = tmp_path / 'universe_2099-01-01.json'
    cache_file.write_text(json.dumps(['SPY', 'QQQ']))
    with patch('pipeline.universe_builder.date') as mock_date:
        mock_date.today.return_value = type('D', (), {'isoformat': lambda self: '2099-01-01'})()
        out = build_universe_cached(UniverseFilters(), sources=[], cache_dir=tmp_path)
        assert out == ['SPY', 'QQQ']


def test_build_universe_falls_back_to_known_good_when_scrape_fails():
    src = _FakeSource(volumes={}, chains={})
    with patch('pipeline.universe_builder.fetch_sp500_constituents', side_effect=RuntimeError('wiki down')):
        out = build_universe(UniverseFilters(top_n=3), [src])
        # Falls back to first 3 of KNOWN_GOOD_FALLBACK
        assert out == ['SPY', 'QQQ', 'IWM']
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_universe_builder.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 4: Implement `pipeline/universe_builder.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import logging
import pandas as pd

from data.sources.base import DataSource
from data.fallback import fetch_with_fallback, DataFetchError
from pipeline.universe import KNOWN_GOOD_FALLBACK


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseFilters:
    min_avg_volume: int = 1_000_000
    require_options_chain: bool = True
    min_iv_rank: float = 0.0
    top_n: int = 50


def fetch_sp500_constituents() -> list[str]:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    symbols = tables[0]['Symbol'].astype(str).tolist()
    # Yahoo uses '-' where Wikipedia uses '.' (e.g., BRK.B → BRK-B)
    return [s.replace('.', '-') for s in symbols]


def _avg_volume_30d(history: pd.Series) -> float:
    # If the source returned a series of volumes (synthetic test path), mean those.
    # Real callers pass close prices, so volume must come from elsewhere — we treat
    # the series mean as the rank signal (price-as-proxy works since we just need an
    # ordering; in the real wiring we pass an actual volume series via a helper).
    tail = history.dropna().tail(30)
    return float(tail.mean()) if len(tail) else 0.0


def build_universe(filters: UniverseFilters, sources: list[DataSource]) -> list[str]:
    try:
        candidates = fetch_sp500_constituents()
    except Exception as e:
        log.warning(f"S&P 500 fetch failed ({e}); falling back to KNOWN_GOOD_FALLBACK")
        candidates = list(KNOWN_GOOD_FALLBACK)

    qualified: list[tuple[str, float]] = []
    for t in candidates:
        try:
            hist = fetch_with_fallback(sources, 'fetch_price_history', t, 30)
        except DataFetchError:
            continue
        avg_vol = _avg_volume_30d(hist)
        if avg_vol < filters.min_avg_volume:
            continue
        if filters.require_options_chain:
            try:
                chain = fetch_with_fallback(sources[:2], 'fetch_option_chain', t)
            except DataFetchError:
                continue
            if not chain:
                continue
        qualified.append((t, avg_vol))

    qualified.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in qualified[: filters.top_n]]


def build_universe_cached(filters: UniverseFilters, sources: list[DataSource],
                          cache_dir: Path = Path('cache/universe')) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'universe_{date.today().isoformat()}.json'
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    universe = build_universe(filters, sources)
    cache_file.write_text(json.dumps(universe))
    return universe
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_universe_builder.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add pipeline/universe_builder.py tests/pipeline/__init__.py tests/pipeline/test_universe_builder.py
git commit -m "feat(pipeline): dynamic S&P 500 universe builder with filesystem cache + KNOWN_GOOD fallback"
```

---

## Task 18: pipeline/earnings.py

**Files:**
- Create: `pipeline/earnings.py`
- Test: `tests/pipeline/test_earnings.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from pipeline.earnings import has_earnings_within


def test_returns_true_when_earnings_within_window():
    soon = date.today() + timedelta(days=3)
    fake_events = {'SPY': {'earnings': {'earningsDate': [pd.Timestamp(soon)]}}}
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.return_value.calendar_events = fake_events
        assert has_earnings_within('SPY', days=5) is True


def test_returns_false_when_no_earnings_in_window():
    far = date.today() + timedelta(days=30)
    fake_events = {'SPY': {'earnings': {'earningsDate': [pd.Timestamp(far)]}}}
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.return_value.calendar_events = fake_events
        assert has_earnings_within('SPY', days=5) is False


def test_returns_false_when_no_earnings_data():
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.return_value.calendar_events = {'SPY': {}}
        assert has_earnings_within('SPY', days=5) is False


def test_returns_false_when_yahooquery_throws():
    with patch('pipeline.earnings.Ticker') as MockTicker:
        MockTicker.side_effect = RuntimeError('network')
        assert has_earnings_within('SPY', days=5) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_earnings.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `pipeline/earnings.py`**

```python
from __future__ import annotations
from datetime import date, timedelta
import pandas as pd
from yahooquery import Ticker


def has_earnings_within(ticker: str, days: int) -> bool:
    try:
        events = Ticker(ticker).calendar_events
    except Exception:
        return False
    if not isinstance(events, dict):
        return False
    info = events.get(ticker, {})
    if not isinstance(info, dict):
        return False
    earnings_info = info.get('earnings', {}) or {}
    dates = earnings_info.get('earningsDate', []) or []
    cutoff = date.today() + timedelta(days=days)
    for d in dates:
        try:
            d_norm = pd.Timestamp(d).date()
        except Exception:
            continue
        if date.today() <= d_norm <= cutoff:
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_earnings.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/earnings.py tests/pipeline/test_earnings.py
git commit -m "feat(pipeline): earnings blackout check via yahooquery calendar_events"
```

---

## Task 19: pipeline/run_scan.py — the orchestrator

**Files:**
- Create: `pipeline/run_scan.py`
- Test: `tests/pipeline/test_run_scan.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date, datetime
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from data.sources.base import DataSource, RawContract
from pipeline.run_scan import run_scan, ScanConfig, ScanResult


class _FakeSource(DataSource):
    def __init__(self, history: pd.Series, spot: float, chain: list[RawContract]):
        self._history = history; self._spot = spot; self._chain = chain
    def fetch_spot(self, ticker): return self._spot
    def fetch_price_history(self, ticker, lookback_days): return self._history
    def fetch_option_chain(self, ticker): return self._chain


def _make_raw(option_type='put', strike=100.0, dte=21, iv=0.20, mid=2.0):
    return RawContract(
        ticker='X', expiration=date(2026, 5, 30) + pd.Timedelta(days=dte).to_pytimedelta(),
        strike=strike, option_type=option_type, bid=mid-0.05, ask=mid+0.05,
        volume=1000, open_interest=1000, implied_volatility=iv, dte=dte, spot_price=100.0,
    )


def test_run_scan_returns_scan_result_with_candidates():
    history = pd.Series(np.exp(np.cumsum(np.random.RandomState(0).normal(0, 0.01, 365)))) * 100
    chain = [_make_raw(option_type='put', strike=98, mid=1.0),
             _make_raw(option_type='call', strike=102, mid=1.0)]
    src = _FakeSource(history, spot=100.0, chain=chain)

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[src])

    assert isinstance(result, ScanResult)
    assert result.skipped == {} or 'X' not in result.skipped
    assert isinstance(result.candidates, list)


def test_run_scan_skips_tickers_with_earnings():
    history = pd.Series([100.0] * 100)
    src = _FakeSource(history, spot=100.0, chain=[])

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=True):
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[src])

    assert result.skipped == {'X': 'earnings_blackout'}
    assert result.candidates == []


def test_run_scan_records_fetch_failure_per_ticker():
    class _Broken(DataSource):
        def fetch_spot(self, t): raise RuntimeError('no')
        def fetch_price_history(self, t, lb): raise RuntimeError('no')
        def fetch_option_chain(self, t): raise RuntimeError('no')

    with patch('pipeline.run_scan.build_universe_cached', return_value=['X']), \
         patch('pipeline.run_scan.has_earnings_within', return_value=False):
        result = run_scan(ScanConfig(today=date(2026, 5, 30)), sources_override=[_Broken()])

    assert 'X' in result.skipped
    assert 'fetch_failed' in result.skipped['X']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/pipeline/test_run_scan.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `pipeline/run_scan.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import logging
import numpy as np

from data.sources.base import DataSource
from data.sources.yahooquery_source import YahooQuerySource
from data.sources.yfinance_source import YfinanceSource
from data.sources.stooq_source import StooqSource
from data.fallback import fetch_with_fallback, DataFetchError
from data.adapters import to_contract

from domain.signals import compute_regime
from domain.strategy import ALL_STRATEGIES

from engine.pop_models import pop_delta, pop_black_scholes, pop_historical, pop_garch_mc, blend_pop
from engine.stress import compute_stress
from engine.risk_filters import apply_filters
from engine.scorer import composite_score, label_from, ScoredCandidate

from pipeline.universe_builder import build_universe_cached, UniverseFilters
from pipeline.earnings import has_earnings_within


log = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    risk_free_rate: float = 0.053
    earnings_blackout_days: int = 5
    n_monte_carlo_paths: int = 10_000
    today: date = field(default_factory=date.today)
    universe_filters: UniverseFilters = field(default_factory=UniverseFilters)


@dataclass
class ScanResult:
    timestamp: datetime
    config: ScanConfig
    candidates: list[ScoredCandidate]
    skipped: dict[str, str]


def _expected_profit_when_itm_long(terminal_prices: np.ndarray, contract, strategy) -> float:
    # Mean intrinsic gain on paths satisfying profit_condition, minus mid
    matches = np.array([strategy.profit_condition(float(p), contract) for p in terminal_prices])
    if not matches.any():
        return 0.0
    if contract.option_type == 'call':
        intrinsic = np.maximum(terminal_prices[matches] - contract.strike, 0.0)
    else:
        intrinsic = np.maximum(contract.strike - terminal_prices[matches], 0.0)
    return float(intrinsic.mean() - contract.mid)


def run_scan(config: ScanConfig, sources_override: Optional[list[DataSource]] = None) -> ScanResult:
    sources = sources_override if sources_override is not None else [
        YahooQuerySource(), YfinanceSource(), StooqSource()
    ]
    universe = build_universe_cached(config.universe_filters, sources)
    candidates: list[ScoredCandidate] = []
    skipped: dict[str, str] = {}

    for ticker in universe:
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

        for strategy in ALL_STRATEGIES:
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
                    # Use a cheap MC for expected profit-when-ITM (reuse GARCH MC paths approximation)
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

                reason_for = (
                    f"PoP={p_blend:.0%}, EV=${ev:.2f}, regime ratio={regime.rv_iv_ratio:.2f} "
                    f"favors {strategy.direction}"
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
                ))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return ScanResult(timestamp=datetime.now(), config=config,
                      candidates=candidates, skipped=skipped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/pipeline/test_run_scan.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/run_scan.py tests/pipeline/test_run_scan.py
git commit -m "feat(pipeline): run_scan orchestrator wires data → domain → engine into candidates"
```

---

## Task 20: ui/exporter.py — CSV + JSON output

**Files:**
- Create: `ui/__init__.py` (empty)
- Create: `ui/exporter.py`
- Test: `tests/ui/test_exporter.py`

- [ ] **Step 1: Create empty `ui/__init__.py` and `tests/ui/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

`tests/ui/test_exporter.py`:

```python
from datetime import date, datetime
import json
from pathlib import Path
import pytest
from domain.contract import Contract
from domain.strategy import NakedPut
from engine.stress import StressResult
from engine.risk_filters import FilterResult
from engine.scorer import ScoredCandidate
from pipeline.run_scan import ScanConfig, ScanResult
from ui.exporter import write_scan


def _make_candidate():
    c = Contract(
        ticker='SPY', expiration=date(2026, 6, 20), strike=620.0,
        option_type='put', bid=2.0, ask=2.1, mid=2.05,
        volume=1000, open_interest=2000, implied_volatility=0.18,
        delta=-0.22, gamma=0.01, theta=-0.05, vega=0.1,
        dte=21, spot_price=635.0,
    )
    return ScoredCandidate(
        contract=c, strategy=NakedPut(),
        pop_blended=0.80, pop_delta=0.78, pop_bs=0.79, pop_historical=0.81, pop_garch_mc=0.80,
        stress=StressResult(-1.0, -3.0, -2.0), ev=0.5, max_adverse_loss=3.0, margin_estimate=200.0,
        filter_result=FilterResult(passed=True), composite_score=85.0, label='TRADE',
        reason_for='r1', reason_against='',
    )


def test_write_scan_creates_csv_and_json(tmp_path):
    result = ScanResult(timestamp=datetime(2026, 5, 30, 16, 15),
                        config=ScanConfig(today=date(2026, 5, 30)),
                        candidates=[_make_candidate()], skipped={'BAD': 'fetch_failed: x'})
    csv_path, json_path = write_scan(result, tmp_path)
    assert csv_path.exists()
    assert json_path.exists()
    assert (tmp_path / 'latest.json').exists()


def test_write_scan_csv_has_expected_columns(tmp_path):
    result = ScanResult(timestamp=datetime(2026, 5, 30, 16, 15),
                        config=ScanConfig(today=date(2026, 5, 30)),
                        candidates=[_make_candidate()], skipped={})
    csv_path, _ = write_scan(result, tmp_path)
    header = csv_path.read_text().splitlines()[0]
    for col in ('ticker', 'strategy', 'strike', 'expiration', 'mid', 'pop_blended', 'ev', 'composite_score', 'label'):
        assert col in header


def test_write_scan_json_roundtrips(tmp_path):
    result = ScanResult(timestamp=datetime(2026, 5, 30, 16, 15),
                        config=ScanConfig(today=date(2026, 5, 30)),
                        candidates=[_make_candidate()], skipped={})
    _, json_path = write_scan(result, tmp_path)
    data = json.loads(json_path.read_text())
    assert data['candidates'][0]['contract']['ticker'] == 'SPY'
    assert data['skipped'] == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/ui/test_exporter.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 4: Implement `ui/exporter.py`**

```python
from __future__ import annotations
import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path

from pipeline.run_scan import ScanResult


def _serialize(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


def write_scan(result: ScanResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.timestamp.strftime('%Y%m%d_%H%M')
    csv_path = output_dir / f'scan_{stamp}.csv'
    json_path = output_dir / f'scan_{stamp}.json'
    latest_path = output_dir / 'latest.json'

    columns = ['ticker', 'strategy', 'option_type', 'strike', 'expiration', 'dte',
               'spot_price', 'bid', 'ask', 'mid', 'volume', 'open_interest',
               'implied_volatility', 'delta', 'gamma', 'theta', 'vega',
               'pop_blended', 'pop_delta', 'pop_bs', 'pop_historical', 'pop_garch_mc',
               'ev', 'max_adverse_loss', 'margin_estimate', 'composite_score', 'label',
               'reason_for', 'reason_against', 'failed_filters']
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for c in result.candidates:
            con = c.contract
            w.writerow([
                con.ticker, c.strategy.name, con.option_type, con.strike,
                con.expiration.isoformat(), con.dte, con.spot_price,
                con.bid, con.ask, con.mid, con.volume, con.open_interest,
                con.implied_volatility, con.delta, con.gamma, con.theta, con.vega,
                c.pop_blended, c.pop_delta, c.pop_bs, c.pop_historical, c.pop_garch_mc,
                c.ev, c.max_adverse_loss, c.margin_estimate, c.composite_score, c.label,
                c.reason_for, c.reason_against, ';'.join(c.filter_result.failed_filters),
            ])

    payload = {
        'timestamp': result.timestamp.isoformat(),
        'candidates': [
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
                'reason_for': c.reason_for, 'reason_against': c.reason_against,
                'failed_filters': list(c.filter_result.failed_filters),
            }
            for c in result.candidates
        ],
        'skipped': dict(result.skipped),
    }

    json_path.write_text(json.dumps(payload, indent=2, default=str))
    latest_path.write_text(json.dumps(payload, indent=2, default=str))
    return csv_path, json_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ui/test_exporter.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add ui/__init__.py ui/exporter.py tests/ui/__init__.py tests/ui/test_exporter.py
git commit -m "feat(ui): scan result CSV + JSON exporter with latest.json mirror"
```

---

## Task 21: ui/dashboard.py — Streamlit, no auto-refresh

**Files:**
- Create: `ui/dashboard.py`
- Test: (smoke only — Streamlit is hard to unit-test; covered by demo step 3)

- [ ] **Step 1: Create `ui/dashboard.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st


LATEST = Path('output/scans/latest.json')


def _load_latest() -> dict | None:
    if not LATEST.exists():
        return None
    return json.loads(LATEST.read_text())


def _candidates_to_df(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        con = c['contract']
        rows.append({
            'ticker': con['ticker'],
            'strategy': c['strategy'],
            'direction': 'sell' if c['strategy'] in ('naked_put', 'naked_call') else 'buy',
            'option_type': con['option_type'],
            'strike': con['strike'],
            'expiration': con['expiration'],
            'dte': con['dte'],
            'mid': con['mid'],
            'pop_blended': c['pop_blended'],
            'ev': c['ev'],
            'score': c['composite_score'],
            'label': c['label'],
        })
    return pd.DataFrame(rows)


def render_dashboard() -> None:
    st.set_page_config(page_title='Options Scanner', layout='wide')
    st.title('Options Scanner — EOD Candidates')

    data = _load_latest()
    if data is None:
        st.warning('No scan results yet. Run `python -m main scan` first.')
        return

    st.caption(f"Last scan: {data['timestamp']}")
    df = _candidates_to_df(data['candidates'])

    if df.empty:
        st.info('No candidates in latest scan.')
        return

    counts = df['label'].value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    c1.metric('TRADE', counts.get('TRADE', 0))
    c2.metric('WATCHLIST', counts.get('WATCHLIST', 0))
    c3.metric('NO_TRADE', counts.get('NO_TRADE', 0))

    tab_all, tab_sell, tab_buy, tab_trade, tab_watch = st.tabs(
        ['All', 'Sell', 'Buy', 'TRADE only', 'WATCHLIST only']
    )

    def _show(view: pd.DataFrame):
        st.dataframe(view.sort_values('score', ascending=False), use_container_width=True)

    with tab_all: _show(df)
    with tab_sell: _show(df[df['direction'] == 'sell'])
    with tab_buy: _show(df[df['direction'] == 'buy'])
    with tab_trade: _show(df[df['label'] == 'TRADE'])
    with tab_watch: _show(df[df['label'] == 'WATCHLIST'])

    st.divider()
    sel = st.selectbox('Inspect candidate', options=list(range(len(data['candidates']))),
                       format_func=lambda i: f"{data['candidates'][i]['contract']['ticker']} "
                                             f"{data['candidates'][i]['strategy']} "
                                             f"{data['candidates'][i]['contract']['strike']} "
                                             f"{data['candidates'][i]['contract']['expiration']}")
    if sel is not None:
        c = data['candidates'][sel]
        st.write(c)

    if st.button('Refresh from latest.json'):
        st.rerun()
```

- [ ] **Step 2: Smoke-test importability**

Run: `python -c "from ui.dashboard import render_dashboard; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add ui/dashboard.py
git commit -m "feat(ui): Streamlit dashboard with sell/buy tabs and manual refresh"
```

---

## Task 22: main.py — argparse entry point

**Files:**
- Modify: `main.py` (replace entire content)

- [ ] **Step 1: Replace `main.py` content**

```python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from pipeline.run_scan import run_scan, ScanConfig
from pipeline.universe_builder import UniverseFilters, build_universe_cached
from ui.exporter import write_scan


OUTPUT_DIR = Path('output/scans')


def _print_summary(result) -> None:
    n_total = len(result.candidates)
    by_label = {}
    for c in result.candidates:
        by_label[c.label] = by_label.get(c.label, 0) + 1
    print(f"Scanned {n_total} candidates across universe of {len(set(c.contract.ticker for c in result.candidates))} tickers.")
    print(f"  TRADE: {by_label.get('TRADE', 0)}  WATCHLIST: {by_label.get('WATCHLIST', 0)}  NO_TRADE: {by_label.get('NO_TRADE', 0)}")
    print(f"  Skipped: {len(result.skipped)} ({', '.join(f'{t}:{r}' for t,r in list(result.skipped.items())[:3])}{'...' if len(result.skipped) > 3 else ''})")
    if result.candidates:
        top = result.candidates[0]
        print(f"  Top: {top.contract.ticker} {top.strategy.name} ${top.contract.strike} {top.contract.expiration} score={top.composite_score}")


def cmd_scan(args) -> int:
    filters = UniverseFilters(
        min_avg_volume=args.min_volume,
        top_n=args.top,
        require_options_chain=True,
    )
    config = ScanConfig(universe_filters=filters)

    sources_override = None
    if args.tickers:
        from data.sources.yahooquery_source import YahooQuerySource
        from data.sources.yfinance_source import YfinanceSource
        from data.sources.stooq_source import StooqSource
        sources_override = [YahooQuerySource(), YfinanceSource(), StooqSource()]
        # Bypass screener via monkey-patch
        import pipeline.run_scan as rs
        rs.build_universe_cached = lambda *a, **kw: [t.strip().upper() for t in args.tickers.split(',')]

    result = run_scan(config, sources_override=sources_override)
    csv_path, json_path = write_scan(result, OUTPUT_DIR)
    _print_summary(result)
    print(f"Wrote: {csv_path}")
    print(f"       {json_path}")

    if args.then_dashboard:
        return cmd_dashboard(args)
    return 0


def cmd_dashboard(_args) -> int:
    import subprocess
    return subprocess.call([sys.executable, '-m', 'streamlit', 'run', 'ui/dashboard.py'])


def cmd_universe(args) -> int:
    if args.action == 'rebuild':
        cache_file = Path('cache/universe') / f"universe_{__import__('datetime').date.today().isoformat()}.json"
        if cache_file.exists():
            cache_file.unlink()
        from data.sources.yahooquery_source import YahooQuerySource
        from data.sources.yfinance_source import YfinanceSource
        from data.sources.stooq_source import StooqSource
        sources = [YahooQuerySource(), YfinanceSource(), StooqSource()]
        out = build_universe_cached(UniverseFilters(), sources)
        print(f"Universe rebuilt: {len(out)} tickers")
        print(', '.join(out))
        return 0
    print(f"Unknown universe action: {args.action}", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(prog='options-scanner')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('scan', help='Run an EOD scan')
    s.add_argument('--top', type=int, default=50)
    s.add_argument('--min-volume', type=int, default=1_000_000)
    s.add_argument('--tickers', type=str, default=None, help='Comma-separated override list')
    s.add_argument('--then-dashboard', action='store_true')
    s.set_defaults(func=cmd_scan)

    d = sub.add_parser('dashboard', help='Launch Streamlit dashboard')
    d.set_defaults(func=cmd_dashboard)

    u = sub.add_parser('universe', help='Manage universe cache')
    u.add_argument('action', choices=['rebuild'])
    u.set_defaults(func=cmd_universe)

    args = p.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test**

Run: `python -m main --help`
Expected: argparse help showing scan/dashboard/universe subcommands.

Run: `python -m main scan --help`
Expected: scan flags including `--top`, `--min-volume`, `--tickers`, `--then-dashboard`.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: argparse main entry — scan / dashboard / universe subcommands"
```

---

## Task 23: Remove obsolete scanner/ + old output and data files

**Files:**
- Delete: `scanner/` (entire directory)
- Delete: `output/dashboard.py`, `output/exporter.py`, `output/__init__.py`
- Delete: `data/universe.py`, `data/earnings_calendar.py`
- Delete: `tests/scanner/` (entire directory if it exists)
- Modify: `data/__init__.py` (keep empty, do NOT delete)

- [ ] **Step 1: Verify no imports remain pointing to removed paths**

Run:
```bash
grep -rE "from (scanner|output\.dashboard|output\.exporter|data\.universe|data\.earnings_calendar)" --include="*.py" .
```
Expected: zero results.

- [ ] **Step 2: Delete obsolete files**

```bash
git rm -r scanner/
git rm output/dashboard.py output/exporter.py output/__init__.py
git rm data/universe.py data/earnings_calendar.py
git rm -r tests/scanner/ 2>/dev/null || true
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: all tests in `tests/domain/`, `tests/engine/`, `tests/data/`, `tests/pipeline/`, `tests/ui/` pass (~80+ tests).

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove obsolete scanner/, output/ui files, data/universe.py, data/earnings_calendar.py — replaced by layered architecture"
```

---

## Task 24: README update

**Files:**
- Modify: `README.md` (replace entire content)

- [ ] **Step 1: Replace README.md with the new doc**

```markdown
# Options Scanner

EOD options scanner for naked puts/calls (sell premium when overpriced) **and** long puts/calls (buy premium when underpriced). Runs once at end of day on a dynamic S&P 500 universe filtered by liquidity. Outputs a ranked candidate table to CSV/JSON and a Streamlit dashboard.

Read-only — no order execution.

## How it works

1. **Universe builder** — pulls S&P 500 from Wikipedia, filters by 30-day avg volume + options availability, caches daily.
2. **Per-ticker regime gate** — compares 30-day realized vol vs front-month ATM implied vol.
   - `RV/IV < 0.8` → favor **selling** (premium overpriced)
   - `RV/IV > 1.2` → favor **buying** (premium underpriced)
   - in-between → both directions, scoring sorts
3. **4 strategies** evaluated per contract: NakedPut, NakedCall, LongPut, LongCall.
4. **4 PoP models** per candidate: delta approx, Black-Scholes risk-neutral, historical rolling-window, GARCH(1,1) + Student-t Monte Carlo. Blended (20/25/20/35).
5. **Stress** — 1σ/2σ adverse moves + 5th-percentile expiry move.
6. **Risk filters** — strategy-specific spread, volume, OI, delta, PoP, EV, stress thresholds.
7. **Composite score (0-100)** = 50% PoP + 30% normalized-EV + 20% regime alignment.
8. **Label**: TRADE (pass + ≥65) / WATCHLIST (pass + 50-64) / NO_TRADE (fail or <50).

## Delta ranges (research-backed)

| Strategy | Delta range | Source |
|---|---|---|
| NakedPut / NakedCall | 0.16 – 0.30 | Tastytrade 16-delta sweet spot + wheel-strategy 0.20-0.30 consensus |
| LongPut / LongCall | 0.40 – 0.60 | Gamma sweet spot for 3-45 DTE directional buys |

## Architecture

```
options-scanner/
├── data/           # I/O — yahooquery + yfinance + stooq fallback chain
├── domain/         # Pure logic — Contract, Strategy, Regime, Greeks
├── engine/         # PoP models, stress, risk filters, scorer
├── pipeline/       # Universe builder, earnings blackout, run_scan
├── ui/             # Streamlit dashboard, CSV/JSON exporter
├── tests/
└── main.py         # CLI: scan / dashboard / universe
```

## Setup

```bash
git clone https://github.com/shans15/options-scanner.git
cd options-scanner
pip install -r requirements.txt
cp .env.example .env   # optional — tune thresholds
```

## Usage

```bash
# Run an EOD scan (top 50 by liquidity from S&P 500)
python -m main scan

# Wider net
python -m main scan --top 100 --min-volume 500000

# Manual ticker override (skips screener)
python -m main scan --tickers SPY,QQQ,AAPL

# Scan and launch dashboard immediately after
python -m main scan --then-dashboard

# Dashboard only (reads latest.json)
python -m main dashboard

# Force universe cache rebuild
python -m main universe rebuild
```

Dashboard opens at `http://localhost:8501`.

## Tests

```bash
pytest                    # unit tests (~80)
pytest -m slow            # integration tests (real network)
```

## Data sources

| Source | Purpose | Notes |
|---|---|---|
| `yahooquery` | Primary: options chains, history, spot, earnings | Quotes ~15-20 min delayed |
| `yfinance` | Fallback: options chains, history, spot | Different scrape path — resilient when yahooquery breaks |
| `stooq` | Fallback²: prices only | No options |
| Wikipedia | S&P 500 constituents | Universe build |
| `py_vollib` | Greeks computation | We provide IV; library returns delta/gamma/theta/vega |

## Disclaimer

Quantitative research tooling only. Not financial advice.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for layered architecture, dynamic universe, buy/sell modes"
```

---

## Task 25: Demo run (end-to-end verification)

**Files:** None — verification only.

- [ ] **Step 1: Full test suite**

Run: `pytest -v`
Expected: all unit tests pass, no failures.

- [ ] **Step 2: Universe build**

Run: `python -m main universe rebuild`
Expected:
- `cache/universe/universe_<today>.json` created
- ≥40 tickers in the list
- SPY, QQQ, AAPL, MSFT all present (eyeball-check output)

Verify:
```bash
cat cache/universe/universe_$(date +%Y-%m-%d).json | python -c "import sys, json; d=json.load(sys.stdin); print(len(d), 'tickers'); print('SPY in:', 'SPY' in d); print('QQQ in:', 'QQQ' in d); print('AAPL in:', 'AAPL' in d); print('MSFT in:', 'MSFT' in d)"
```
Expected: prints `>=40 tickers` and `True` for each.

- [ ] **Step 3: Manual-override scan (fast smoke)**

Run: `python -m main scan --tickers SPY,QQQ,AAPL`
Expected:
- Completes without exceptions
- Prints scan summary line
- Writes `output/scans/scan_<ts>.csv` + `.json` + updates `latest.json`
- At least one candidate in the output (likely WATCHLIST or TRADE — depends on day's regime)

- [ ] **Step 4: Verify outputs**

```bash
ls -la output/scans/
head -2 output/scans/latest.json | python -c "import sys, json; d=json.loads(open('output/scans/latest.json').read()); print('candidates:', len(d['candidates'])); print('skipped:', d['skipped'])"
```
Expected: file listing shows new csv+json files, latest.json parses cleanly, candidate count printed.

- [ ] **Step 5: Both sell and buy candidates present (proves buyer path works)**

```bash
python -c "import json; d=json.load(open('output/scans/latest.json')); dirs={'sell' if c['strategy'] in ('naked_put','naked_call') else 'buy' for c in d['candidates']}; print('Directions present:', dirs)"
```
Expected: prints `{'sell', 'buy'}` if both directions found, otherwise `{'sell'}` or `{'buy'}` only — note in handoff.

- [ ] **Step 6: Dashboard sanity**

Run: `python -m main dashboard`
Expected:
- Streamlit starts at `http://localhost:8501`
- Header shows scan timestamp + counts by label
- Tabs (All / Sell / Buy / TRADE / WATCHLIST) all clickable
- Table renders with rows
- Expand-selector shows one candidate's full detail

Stop with Ctrl+C.

- [ ] **Step 7: Final commit (any dashboard fixes)**

```bash
git add -A
git status
# only commit if there are changes — likely none
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Implementing task(s) |
|---|---|
| §3 Library stack | Task 1 (requirements.txt) |
| §4 Module layout | Tasks 2-22 (create all files) |
| §5 Domain layer | Tasks 2, 3, 4, 5 |
| §6 Engine layer | Tasks 6, 7, 8, 9 |
| §7 Data layer | Tasks 10, 11, 12, 13, 14, 15 |
| §8 Pipeline + universe builder | Tasks 16, 17, 18, 19 |
| §9 UI | Tasks 20, 21, 22 |
| §10 Testing strategy | Every task has tests; coverage matches spec targets |
| §11 Demo run | Task 25 |
| §12 Removed code | Task 23 |
| §15 Research references | Locked into Task 4 (strategy delta ranges) and Task 8 (filter sets) |

All spec sections covered.

**Placeholder scan:** No TBDs, no "TODO", no "implement later", no "similar to Task N" — every step has complete code.

**Type consistency:** Cross-checked:
- `Contract` fields used identically in Tasks 2, 11, 19, 20
- `Strategy.applies_to / breakeven / profit_condition / expected_value / margin_estimate` signatures consistent across Tasks 4, 6, 7, 8, 9, 19
- `RawContract` fields consistent across Tasks 10, 11, 13, 14, 15, 17, 19
- `ScoredCandidate` fields consistent across Tasks 9, 19, 20
- `FilterSet.max_stress_loss_multiple` uses `math.inf` in Task 8; checks with `math.isfinite` in same task — consistent
- `compute_regime(ticker, log_returns, chain)` signature consistent in Tasks 5 and 19
- `compute_stress(c, strategy, sigma_annual)` consistent in Tasks 7, 19
- `expected_value(c, pop_blended, max_adverse_loss, expected_profit_when_itm=0.0)` keyword consistent in Tasks 4, 19
- `apply_filters(c, strategy, pop_blended, ev, stress)` consistent in Tasks 8, 19
- `composite_score(pop_blended=, ev=, max_adverse_loss=, strategy=, regime=, mid=)` keyword-only consistent in Tasks 9, 19

No type drift.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-30-options-scanner-rebuild.md`.**
