# Options Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live, time-aware scanner for short naked puts and naked calls using Robinhood chain data, multi-model probability estimates, hard risk filters, and a Streamlit dashboard with CSV/JSON export.

**Architecture:** robin_stocks pulls live option chains; yfinance provides historical price data for IV rank and PoP calculations; APScheduler triggers scans at 09:45/11:00/13:00/15:00 ET; results are written to `output/scans/latest.json` and displayed in a Streamlit dashboard that auto-refreshes.

**Tech Stack:** Python 3.10+, robin_stocks, yfinance, arch (GARCH), scipy, numpy, pandas, streamlit, apscheduler, python-dotenv, pandas-market-calendars, plotly

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | Entry point: starts background scheduler, renders dashboard |
| `scanner/clock.py` | `is_market_open()`, `get_next_scan_times()` |
| `scanner/screener.py` | `screen_universe()` → top 30-50 scored candidates |
| `scanner/chain_fetcher.py` | `fetch_contracts(ticker)` → list of contract dicts |
| `scanner/quant_engine.py` | All 5 PoP models + stress tests + blended PoP |
| `scanner/risk_filters.py` | `apply_hard_filters(contract)` → (passed, reason) |
| `scanner/scorer.py` | `compute_composite_score()`, `assign_decision()`, `build_scan_result()` |
| `scanner/run_scan.py` | `run_full_scan()` — orchestrates full pipeline |
| `scanner/scheduler.py` | Singleton APScheduler, safe for Streamlit reruns |
| `data/universe.py` | `get_universe()` → list of tickers, cached 24h |
| `data/earnings_calendar.py` | `has_earnings_soon(ticker, days)` → bool |
| `output/exporter.py` | `export_scan(results)` → CSV + JSON |
| `output/dashboard.py` | `render_dashboard()` — full Streamlit UI |
| `tests/` | One test file per module |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `scanner/__init__.py`, `data/__init__.py`, `output/__init__.py`, `tests/__init__.py`
- Create: `output/scans/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
robin_stocks>=2.1.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
arch>=6.3.0
pandas-market-calendars>=4.3.0
streamlit>=1.33.0
python-dotenv>=1.0.0
apscheduler>=3.10.0
plotly>=5.20.0
requests>=2.31.0
pytest>=8.0.0
pytest-mock>=3.14.0
pytz>=2024.1
```

- [ ] **Step 2: Create .env.example**

```
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
RISK_FREE_RATE=0.053
MIN_POP_THRESHOLD=0.70
MIN_COMPOSITE_SCORE=65
MAX_SPREAD_PCT=0.20
MIN_VOLUME=100
MIN_OI=500
EARNINGS_BLACKOUT_DAYS=5
MONTE_CARLO_PATHS=10000
```

- [ ] **Step 3: Create .gitignore**

```
.env
__pycache__/
*.pyc
.pytest_cache/
output/scans/*.csv
output/scans/*.json
data/universe_cache.json
*.pkl
.robinhood_session/
```

- [ ] **Step 4: Create all __init__.py files and output/scans/.gitkeep**

```bash
touch scanner/__init__.py data/__init__.py output/__init__.py tests/__init__.py
mkdir -p output/scans && touch output/scans/.gitkeep
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: no errors. Verify with `python -c "import robin_stocks, yfinance, arch, streamlit, apscheduler"`.

- [ ] **Step 6: Copy .env.example to .env and fill in credentials**

```bash
cp .env.example .env
# Edit .env with your Robinhood email and password
```

- [ ] **Step 7: Commit**

```bash
git init
git add requirements.txt .env.example .gitignore scanner/__init__.py data/__init__.py output/__init__.py tests/__init__.py output/scans/.gitkeep
git commit -m "feat: project setup and dependencies"
```

---

## Task 2: Ticker Universe (`data/universe.py`)

**Files:**
- Create: `data/universe.py`
- Create: `tests/test_universe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_universe.py
from data.universe import get_universe, LIQUID_ETFS

def test_get_universe_returns_list_of_strings():
    universe = get_universe()
    assert isinstance(universe, list)
    assert len(universe) > 10
    assert all(isinstance(t, str) for t in universe)

def test_liquid_etfs_included():
    universe = get_universe()
    for etf in ['SPY', 'QQQ', 'IWM']:
        assert etf in universe

def test_no_duplicates():
    universe = get_universe()
    assert len(universe) == len(set(universe))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_universe.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data.universe'`

- [ ] **Step 3: Implement data/universe.py**

```python
# data/universe.py
import json
import time
from pathlib import Path

LIQUID_ETFS = [
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'XLF', 'XLE', 'XLK',
    'XBI', 'ARKK', 'EEM', 'HYG', 'SLV', 'DIA', 'UVXY'
]

MEME_EXCLUSION_LIST = ['GME', 'AMC', 'BBBY', 'KOSS', 'EXPR']

_CACHE_FILE = Path(__file__).parent / 'universe_cache.json'
_CACHE_MAX_AGE_SECONDS = 24 * 3600


def _fetch_sp500_tickers() -> list[str]:
    import pandas as pd
    tables = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        attrs={'id': 'constituents'}
    )
    df = tables[0]
    return df['Symbol'].str.replace('.', '-', regex=False).tolist()


def get_universe(use_cache: bool = True) -> list[str]:
    """Return deduplicated list of S&P 500 + liquid ETF tickers, cached 24h."""
    if use_cache and _CACHE_FILE.exists():
        age = time.time() - _CACHE_FILE.stat().st_mtime
        if age < _CACHE_MAX_AGE_SECONDS:
            return json.loads(_CACHE_FILE.read_text())

    try:
        sp500 = _fetch_sp500_tickers()
    except Exception:
        sp500 = []  # Fall back to ETFs only if Wikipedia fetch fails

    tickers = sorted(set(sp500 + LIQUID_ETFS) - set(MEME_EXCLUSION_LIST))
    _CACHE_FILE.write_text(json.dumps(tickers))
    return tickers
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_universe.py -v
```

Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add data/universe.py tests/test_universe.py
git commit -m "feat: ticker universe with S&P 500 + ETFs, 24h cache"
```

---

## Task 3: Market Clock (`scanner/clock.py`)

**Files:**
- Create: `scanner/clock.py`
- Create: `tests/test_clock.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clock.py
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz
from scanner.clock import is_market_open, get_next_scan_times, SCAN_TIMES_ET

ET = pytz.timezone('America/New_York')

def test_scan_times_defined():
    assert len(SCAN_TIMES_ET) == 4
    assert (9, 45) in SCAN_TIMES_ET
    assert (11, 0) in SCAN_TIMES_ET
    assert (13, 0) in SCAN_TIMES_ET
    assert (15, 0) in SCAN_TIMES_ET

def test_market_closed_on_weekend():
    # Saturday
    saturday = ET.localize(datetime(2026, 5, 2, 10, 0, 0))
    with patch('scanner.clock._now_et', return_value=saturday):
        assert is_market_open() is False

def test_market_open_on_weekday_during_hours():
    # Monday 10am ET
    monday_10am = ET.localize(datetime(2026, 5, 4, 10, 0, 0))
    with patch('scanner.clock._now_et', return_value=monday_10am):
        assert is_market_open() is True

def test_market_closed_before_open():
    # Monday 8am ET
    monday_8am = ET.localize(datetime(2026, 5, 4, 8, 0, 0))
    with patch('scanner.clock._now_et', return_value=monday_8am):
        assert is_market_open() is False

def test_get_next_scan_times_returns_list():
    times = get_next_scan_times()
    assert isinstance(times, list)
    assert all(isinstance(t, str) for t in times)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_clock.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/clock.py**

```python
# scanner/clock.py
from datetime import datetime
import pytz
import pandas_market_calendars as mcal

ET = pytz.timezone('America/New_York')
SCAN_TIMES_ET = [(9, 45), (11, 0), (13, 0), (15, 0)]
_nyse = mcal.get_calendar('NYSE')


def _now_et() -> datetime:
    return datetime.now(ET)


def is_market_open() -> bool:
    """Return True if NYSE is currently open."""
    now = _now_et()
    schedule = _nyse.schedule(
        start_date=now.strftime('%Y-%m-%d'),
        end_date=now.strftime('%Y-%m-%d')
    )
    if schedule.empty:
        return False
    market_open = schedule.iloc[0]['market_open'].to_pydatetime()
    market_close = schedule.iloc[0]['market_close'].to_pydatetime()
    return market_open <= now <= market_close


def get_next_scan_times() -> list[str]:
    """Return list of today's remaining scan times as 'HH:MM ET' strings."""
    now = _now_et()
    remaining = []
    for hour, minute in SCAN_TIMES_ET:
        scan_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if scan_dt > now:
            remaining.append(f"{hour:02d}:{minute:02d} ET")
    return remaining


def get_market_status() -> str:
    """Return human-readable market status string."""
    if is_market_open():
        next_scans = get_next_scan_times()
        next_str = next_scans[0] if next_scans else 'None today'
        return f"OPEN | Next scan: {next_str}"
    return "CLOSED"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_clock.py -v
```

Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add scanner/clock.py tests/test_clock.py
git commit -m "feat: market clock with NYSE calendar and scan schedule"
```

---

## Task 4: Earnings Calendar (`data/earnings_calendar.py`)

**Files:**
- Create: `data/earnings_calendar.py`
- Create: `tests/test_earnings_calendar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_earnings_calendar.py
from unittest.mock import patch
from datetime import date, timedelta
from data.earnings_calendar import has_earnings_soon

def test_has_earnings_soon_returns_bool():
    with patch('data.earnings_calendar._get_earnings_date', return_value=None):
        result = has_earnings_soon('AAPL', days=5)
    assert isinstance(result, bool)

def test_earnings_within_window_returns_true():
    tomorrow = date.today() + timedelta(days=2)
    with patch('data.earnings_calendar._get_earnings_date', return_value=tomorrow):
        assert has_earnings_soon('AAPL', days=5) is True

def test_earnings_outside_window_returns_false():
    far_future = date.today() + timedelta(days=30)
    with patch('data.earnings_calendar._get_earnings_date', return_value=far_future):
        assert has_earnings_soon('AAPL', days=5) is False

def test_no_earnings_date_returns_false():
    with patch('data.earnings_calendar._get_earnings_date', return_value=None):
        assert has_earnings_soon('AAPL', days=5) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_earnings_calendar.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement data/earnings_calendar.py**

```python
# data/earnings_calendar.py
from datetime import date, timedelta
import yfinance as yf


def _get_earnings_date(ticker: str) -> date | None:
    """Return the next earnings date for a ticker, or None if unknown."""
    try:
        info = yf.Ticker(ticker).calendar
        if info is None or info.empty:
            return None
        # calendar index 0 = 'Earnings Date'
        earnings_dt = info.columns[0]
        if hasattr(earnings_dt, 'date'):
            return earnings_dt.date()
        return None
    except Exception:
        return None


def has_earnings_soon(ticker: str, days: int = 5) -> bool:
    """Return True if ticker has earnings within `days` calendar days."""
    earnings_date = _get_earnings_date(ticker)
    if earnings_date is None:
        return False
    today = date.today()
    return today <= earnings_date <= today + timedelta(days=days)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_earnings_calendar.py -v
```

Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add data/earnings_calendar.py tests/test_earnings_calendar.py
git commit -m "feat: earnings calendar filter via yfinance"
```

---

## Task 5: Stock Screener (`scanner/screener.py`)

**Files:**
- Create: `scanner/screener.py`
- Create: `tests/test_screener.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_screener.py
from unittest.mock import patch
import pandas as pd
import numpy as np
from scanner.screener import compute_iv_rank, compute_rsi, score_ticker, TickerScore

def _make_price_series(n=252, trend='up'):
    prices = np.linspace(100, 130 if trend == 'up' else 70, n)
    return pd.Series(prices)

def test_compute_iv_rank_in_range():
    prices = _make_price_series()
    rank = compute_iv_rank(0.28, prices)
    assert 0.0 <= rank <= 100.0

def test_compute_rsi_in_range():
    prices = _make_price_series()
    rsi = compute_rsi(prices)
    assert 0.0 <= rsi <= 100.0

def test_score_ticker_returns_ticker_score():
    prices = _make_price_series(trend='up')
    result = score_ticker(
        ticker='AAPL',
        current_price=150.0,
        avg_volume=5_000_000,
        current_iv=0.28,
        price_history=prices,
        vix_level=18.0,
        spy_trend='up',
    )
    assert isinstance(result, TickerScore)
    assert 0.0 <= result.score <= 100.0
    assert result.ticker == 'AAPL'

def test_low_volume_ticker_scores_low():
    prices = _make_price_series()
    result = score_ticker(
        ticker='TINY',
        current_price=5.0,
        avg_volume=100_000,
        current_iv=0.20,
        price_history=prices,
        vix_level=18.0,
        spy_trend='up',
    )
    assert result.score < 40.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_screener.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/screener.py**

```python
# scanner/screener.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from data.universe import get_universe
from data.earnings_calendar import has_earnings_soon
import os

MIN_PRICE = 10.0
MIN_AVG_VOLUME = 1_000_000
EARNINGS_BLACKOUT_DAYS = int(os.getenv('EARNINGS_BLACKOUT_DAYS', '5'))
TOP_N = 50


@dataclass
class TickerScore:
    ticker: str
    score: float           # 0-100
    current_price: float
    avg_volume: float
    iv_rank: float
    trend: str             # 'up', 'down', 'neutral'
    rsi: float
    current_iv: float


def compute_iv_rank(current_iv: float, price_history: pd.Series) -> float:
    """IV rank proxy: position of current_iv within 52-week RV range."""
    log_returns = np.log(price_history / price_history.shift(1)).dropna()
    rv_30d = log_returns.rolling(30).std() * np.sqrt(252)
    low_rv = rv_30d.min()
    high_rv = rv_30d.max()
    if high_rv <= low_rv:
        return 50.0
    rank = (current_iv - low_rv) / (high_rv - low_rv) * 100
    return float(np.clip(rank, 0.0, 100.0))


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI(14) from a price series."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _compute_trend(prices: pd.Series) -> str:
    if len(prices) < 50:
        return 'neutral'
    sma20 = prices.iloc[-20:].mean()
    sma50 = prices.iloc[-50:].mean()
    if sma20 > sma50 * 1.01:
        return 'up'
    if sma20 < sma50 * 0.99:
        return 'down'
    return 'neutral'


def score_ticker(
    ticker: str,
    current_price: float,
    avg_volume: float,
    current_iv: float,
    price_history: pd.Series,
    vix_level: float,
    spy_trend: str,
) -> TickerScore:
    """Compute a 0-100 score for a ticker as an options selling candidate."""
    iv_rank = compute_iv_rank(current_iv, price_history)
    rsi = compute_rsi(price_history)
    trend = _compute_trend(price_history)

    # IV Rank (30%)
    iv_score = iv_rank * 0.30

    # Trend alignment (25%): puts prefer uptrend, calls prefer downtrend
    trend_score = {'up': 75.0, 'neutral': 50.0, 'down': 25.0}.get(trend, 50.0) * 0.25

    # Momentum (20%): RSI 40-60 = neutral/good for premium selling
    if 40 <= rsi <= 65:
        momentum_score = 80.0
    elif rsi < 30 or rsi > 80:
        momentum_score = 20.0
    else:
        momentum_score = 50.0
    momentum_score *= 0.20

    # Liquidity (15%): based on avg volume
    if avg_volume >= 5_000_000:
        liq = 100.0
    elif avg_volume >= 1_000_000:
        liq = 70.0
    elif avg_volume >= 500_000:
        liq = 40.0
    else:
        liq = 10.0
    liq_score = liq * 0.15

    # Market regime (10%): prefer elevated but not extreme VIX
    if 15 <= vix_level <= 30:
        regime = 80.0
    elif vix_level > 35:
        regime = 30.0
    else:
        regime = 50.0
    regime_score = regime * 0.10

    total = iv_score + trend_score + momentum_score + liq_score + regime_score
    return TickerScore(
        ticker=ticker,
        score=round(total, 2),
        current_price=current_price,
        avg_volume=avg_volume,
        iv_rank=iv_rank,
        trend=trend,
        rsi=rsi,
        current_iv=current_iv,
    )


def screen_universe(top_n: int = TOP_N) -> list[TickerScore]:
    """
    Download market data for the full universe, apply hard filters,
    score each ticker, and return the top N candidates.
    """
    tickers = get_universe()

    # Fetch VIX and SPY for market regime context
    try:
        vix_data = yf.download('^VIX', period='5d', progress=False)['Close']
        vix_level = float(vix_data.iloc[-1])
        spy_data = yf.download('SPY', period='60d', progress=False)['Close']
        spy_trend = _compute_trend(spy_data)
    except Exception:
        vix_level = 20.0
        spy_trend = 'neutral'

    scored = []
    for ticker in tickers:
        try:
            hist = yf.download(ticker, period='1y', progress=False)['Close'].squeeze()
            if hist is None or len(hist) < 60:
                continue

            info = yf.Ticker(ticker).fast_info
            current_price = float(getattr(info, 'last_price', 0) or hist.iloc[-1])
            avg_volume = float(getattr(info, 'three_month_average_volume', 0) or 0)

            if current_price < MIN_PRICE:
                continue
            if avg_volume < MIN_AVG_VOLUME:
                continue
            if has_earnings_soon(ticker, days=EARNINGS_BLACKOUT_DAYS):
                continue

            # Use 30-day ATM IV approximation from recent realized vol
            log_returns = np.log(hist / hist.shift(1)).dropna()
            current_iv = float(log_returns.iloc[-30:].std() * np.sqrt(252))

            ts = score_ticker(
                ticker=ticker,
                current_price=current_price,
                avg_volume=avg_volume,
                current_iv=current_iv,
                price_history=hist,
                vix_level=vix_level,
                spy_trend=spy_trend,
            )
            scored.append(ts)
        except Exception:
            continue

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_screener.py -v
```

Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add scanner/screener.py tests/test_screener.py
git commit -m "feat: stock screener with IV rank, RSI, trend, liquidity scoring"
```

---

## Task 6: Option Chain Fetcher (`scanner/chain_fetcher.py`)

**Files:**
- Create: `scanner/chain_fetcher.py`
- Create: `tests/test_chain_fetcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chain_fetcher.py
from unittest.mock import patch, MagicMock
from scanner.chain_fetcher import (
    parse_contract, filter_by_delta, Contract, fetch_contracts
)

RAW_PUT = {
    'strike_price': '185.0',
    'bid_price': '1.10',
    'ask_price': '1.30',
    'volume': '450',
    'open_interest': '1200',
    'implied_volatility': '0.2800',
    'delta': '-0.1800',
    'gamma': '0.0500',
    'theta': '-0.0400',
    'vega': '0.1200',
    'expiration_date': '2026-05-15',
}

def test_parse_contract_put():
    c = parse_contract(RAW_PUT, ticker='AAPL', spot_price=195.0, option_type='put')
    assert c is not None
    assert c['ticker'] == 'AAPL'
    assert c['strategy'] == 'naked_put'
    assert c['strike'] == 185.0
    assert c['bid'] == 1.10
    assert c['delta'] == -0.18
    assert c['dte'] > 0

def test_parse_contract_call():
    raw = {**RAW_PUT, 'delta': '0.1800'}
    c = parse_contract(raw, ticker='AAPL', spot_price=195.0, option_type='call')
    assert c['strategy'] == 'naked_call'

def test_filter_by_delta_puts():
    contracts = [
        {'delta': -0.05, 'strategy': 'naked_put'},
        {'delta': -0.20, 'strategy': 'naked_put'},  # keep
        {'delta': -0.35, 'strategy': 'naked_put'},  # keep
        {'delta': -0.50, 'strategy': 'naked_put'},
    ]
    filtered = filter_by_delta(contracts)
    assert len(filtered) == 2
    assert all(-0.35 <= c['delta'] <= -0.05 for c in filtered)

def test_parse_contract_missing_fields_returns_none():
    c = parse_contract({}, ticker='AAPL', spot_price=195.0, option_type='put')
    assert c is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_chain_fetcher.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/chain_fetcher.py**

```python
# scanner/chain_fetcher.py
import time
from datetime import date, datetime
from typing import TypedDict
import robin_stocks.robinhood as r
from dotenv import load_dotenv
import os

load_dotenv()

DELTA_MIN = 0.05
DELTA_MAX = 0.35
DTE_MIN = 3
DTE_MAX = 45
MAX_EXPIRATIONS = 4
RATE_LIMIT_SECONDS = 0.5

_logged_in = False


class Contract(TypedDict):
    ticker: str
    strategy: str          # 'naked_put' | 'naked_call'
    expiration: str        # 'YYYY-MM-DD'
    strike: float
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


def _ensure_login() -> None:
    global _logged_in
    if not _logged_in:
        r.login(
            username=os.getenv('ROBINHOOD_USERNAME'),
            password=os.getenv('ROBINHOOD_PASSWORD'),
            store_session=True,
        )
        _logged_in = True


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default


def parse_contract(
    raw: dict,
    ticker: str,
    spot_price: float,
    option_type: str,
) -> Contract | None:
    """Parse a raw robin_stocks option dict into a Contract. Returns None if required fields missing."""
    try:
        strike = _safe_float(raw.get('strike_price'))
        bid = _safe_float(raw.get('bid_price'))
        ask = _safe_float(raw.get('ask_price'))
        delta = _safe_float(raw.get('delta'))
        exp_str = raw.get('expiration_date', '')

        if not strike or not exp_str or not delta:
            return None

        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        dte = (exp_date - date.today()).days

        if dte < DTE_MIN or dte > DTE_MAX:
            return None

        mid = round((bid + ask) / 2, 4) if bid + ask > 0 else 0.0
        strategy = 'naked_put' if option_type == 'put' else 'naked_call'

        return Contract(
            ticker=ticker,
            strategy=strategy,
            expiration=exp_str,
            strike=strike,
            bid=bid,
            ask=ask,
            mid=mid,
            volume=_safe_int(raw.get('volume')),
            open_interest=_safe_int(raw.get('open_interest')),
            implied_volatility=_safe_float(raw.get('implied_volatility')),
            delta=delta,
            gamma=_safe_float(raw.get('gamma')),
            theta=_safe_float(raw.get('theta')),
            vega=_safe_float(raw.get('vega')),
            dte=dte,
            spot_price=spot_price,
        )
    except Exception:
        return None


def filter_by_delta(contracts: list[dict]) -> list[dict]:
    """Keep only contracts within the target delta range."""
    result = []
    for c in contracts:
        d = abs(c.get('delta', 0))
        if DELTA_MIN <= d <= DELTA_MAX:
            result.append(c)
    return result


def fetch_contracts(ticker: str, spot_price: float) -> list[Contract]:
    """
    Fetch option contracts for a ticker from Robinhood.
    Returns filtered list of Contract dicts within delta and DTE range.
    """
    _ensure_login()
    contracts = []

    try:
        chains = r.options.get_chains(ticker)
        if not chains:
            return []

        expirations = chains.get('expiration_dates', [])
        # Filter to DTE range and take first MAX_EXPIRATIONS
        valid_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_date - date.today()).days
            if DTE_MIN <= dte <= DTE_MAX:
                valid_expirations.append(exp)
        valid_expirations = valid_expirations[:MAX_EXPIRATIONS]

        for exp in valid_expirations:
            for option_type in ('put', 'call'):
                time.sleep(RATE_LIMIT_SECONDS)
                raw_options = r.options.find_options_by_expiration(
                    ticker, exp, optionType=option_type
                )
                if not raw_options:
                    continue
                for raw in raw_options:
                    contract = parse_contract(raw, ticker, spot_price, option_type)
                    if contract:
                        contracts.append(contract)

    except Exception:
        return []

    return filter_by_delta(contracts)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_chain_fetcher.py -v
```

Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add scanner/chain_fetcher.py tests/test_chain_fetcher.py
git commit -m "feat: option chain fetcher with robin_stocks, delta and DTE filtering"
```

---

## Task 7: Quantitative Engine (`scanner/quant_engine.py`)

**Files:**
- Create: `scanner/quant_engine.py`
- Create: `tests/test_quant_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quant_engine.py
import numpy as np
import pandas as pd
import pytest
from scanner.quant_engine import (
    pop_delta, pop_black_scholes, pop_historical,
    pop_garch_mc, compute_stress_scenarios, blend_pop,
    StressResult, PopResult,
)

# Realistic OTM put: SPY at 530, strike 510, 14 DTE, IV 18%
S, K, IV, DTE, R = 530.0, 510.0, 0.18, 14, 0.053

def _make_returns(n=252, vol=0.01):
    np.random.seed(42)
    return pd.Series(np.random.normal(0, vol, n))

def test_pop_delta_put():
    result = pop_delta(delta=-0.15, option_type='put')
    assert 0.7 < result < 0.95

def test_pop_delta_call():
    result = pop_delta(delta=0.15, option_type='call')
    assert 0.7 < result < 0.95

def test_pop_black_scholes_put():
    result = pop_black_scholes(S=S, K=K, r=R, sigma=IV, T=DTE/252, option_type='put')
    assert 0.5 < result < 1.0

def test_pop_black_scholes_call():
    result = pop_black_scholes(S=530, K=550, r=R, sigma=IV, T=DTE/252, option_type='call')
    assert 0.5 < result < 1.0

def test_pop_historical_returns_float():
    returns = _make_returns()
    result = pop_historical(S=S, K=K, dte=DTE, log_returns=returns)
    assert 0.0 <= result <= 1.0

def test_pop_garch_mc_returns_float():
    returns = _make_returns()
    result = pop_garch_mc(S=S, K=K, dte=DTE, log_returns=returns, n_paths=500)
    assert 0.0 <= result <= 1.0

def test_blend_pop_weights_sum_to_one():
    result = blend_pop(
        pop_d=0.85, pop_bs=0.83, pop_hist=0.80, pop_garch=0.82
    )
    # blended should be a weighted avg in range of inputs
    assert 0.80 <= result <= 0.86

def test_stress_scenarios_put():
    result = compute_stress_scenarios(
        S=S, K=K, premium=1.50, sigma=IV, option_type='put'
    )
    assert isinstance(result, StressResult)
    assert result.stress_1sd <= 0    # OTM put stress scenarios should show loss
    assert result.stress_2sd <= result.stress_1sd
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_quant_engine.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/quant_engine.py**

```python
# scanner/quant_engine.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class StressResult:
    stress_1sd: float    # P&L if underlying moves 1 SD against position
    stress_2sd: float    # P&L if underlying moves 2 SD against position
    stress_expiry: float # P&L at expiry 5th percentile path


@dataclass
class PopResult:
    pop_delta: float
    pop_bs: float
    pop_historical: float
    pop_garch_mc: float
    pop_blended: float
    stress: StressResult
    expected_value: float
    breakeven: float
    margin_estimate: float
    iv_rank: float


def pop_delta(delta: float, option_type: str) -> float:
    """PoP from delta approximation: 1 - |delta|."""
    return float(np.clip(1.0 - abs(delta), 0.0, 1.0))


def pop_black_scholes(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str
) -> float:
    """Risk-neutral probability of expiring OTM (i.e., profitable for seller)."""
    if T <= 0 or sigma <= 0:
        return 0.5
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'put':
        return float(norm.cdf(d2))    # P(S_T > K)
    else:
        return float(norm.cdf(-d2))   # P(S_T < K)


def pop_historical(
    S: float, K: float, dte: int, log_returns: pd.Series
) -> float:
    """
    Historical PoP: fraction of rolling DTE-day log-return windows
    where the stock would have stayed on the profitable side of the strike.
    """
    if len(log_returns) < dte + 10:
        return 0.5

    rolling = log_returns.rolling(dte).sum().dropna()
    required_return = np.log(K / S)

    if K < S:  # Put: profitable if cumulative return > required_return (stay above K)
        pop = (rolling > required_return).mean()
    else:       # Call: profitable if cumulative return < required_return (stay below K)
        pop = (rolling < required_return).mean()

    return float(np.clip(pop, 0.0, 1.0))


def pop_garch_mc(
    S: float,
    K: float,
    dte: int,
    log_returns: pd.Series,
    n_paths: int = 10000,
) -> float:
    """
    GARCH(1,1) + Monte Carlo PoP with Student-t innovations.
    Falls back to historical vol if GARCH fails.
    """
    try:
        from arch import arch_model
        pct_returns = log_returns * 100
        model = arch_model(pct_returns, vol='Garch', p=1, q=1, dist='t')
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=dte)
        avg_var = forecast.variance.values[-1].mean()
        daily_vol = np.sqrt(avg_var) / 100
        nu = float(result.params.get('nu', 8.0))
        nu = max(nu, 3.0)
    except Exception:
        daily_vol = float(log_returns.std())
        nu = 8.0

    np.random.seed(42)
    z = np.random.standard_t(df=nu, size=(n_paths, dte))
    z = z / np.sqrt(nu / (nu - 2))  # standardize to unit variance
    cumulative = (daily_vol * z).sum(axis=1)
    final_prices = S * np.exp(cumulative)

    if K < S:
        pop = (final_prices > K).mean()
    else:
        pop = (final_prices < K).mean()

    return float(np.clip(pop, 0.0, 1.0))


def blend_pop(
    pop_d: float, pop_bs: float, pop_hist: float, pop_garch: float
) -> float:
    """Weighted blend of the four PoP estimates."""
    blended = 0.20 * pop_d + 0.25 * pop_bs + 0.20 * pop_hist + 0.35 * pop_garch
    return float(np.clip(blended, 0.0, 1.0))


def compute_stress_scenarios(
    S: float, K: float, premium: float, sigma: float, option_type: str
) -> StressResult:
    """
    Compute unrealized P&L under 3 stress scenarios.
    Premium received = max profit. Losses are negative.
    """
    daily_vol = sigma / np.sqrt(252)

    def intrinsic_loss(spot_at_scenario: float) -> float:
        """Unrealized loss at scenario price (ignoring time value)."""
        if option_type == 'put':
            intrinsic = max(K - spot_at_scenario, 0)
        else:
            intrinsic = max(spot_at_scenario - K, 0)
        return round(premium - intrinsic, 4)

    # 1 SD move against position
    if option_type == 'put':
        s1 = S * (1 - daily_vol)      # down move hurts puts
        s2 = S * (1 - 2 * daily_vol)
    else:
        s1 = S * (1 + daily_vol)      # up move hurts calls
        s2 = S * (1 + 2 * daily_vol)

    # Expiry worst case: 5th percentile move over full DTE
    # Use 1-year approximation (30 days)
    worst_case_return = -1.645 * sigma * np.sqrt(30 / 252)
    if option_type == 'put':
        s_expiry = S * np.exp(worst_case_return)
    else:
        s_expiry = S * np.exp(-worst_case_return)

    return StressResult(
        stress_1sd=intrinsic_loss(s1),
        stress_2sd=intrinsic_loss(s2),
        stress_expiry=intrinsic_loss(s_expiry),
    )


def estimate_margin(S: float, K: float, premium: float, strategy: str) -> float:
    """Simplified Reg-T naked option margin estimate (per contract = x100)."""
    otm_amount = abs(S - K)
    if strategy == 'naked_put':
        margin = max(0.20 * S - otm_amount + premium, 0.10 * K + premium)
    else:
        margin = max(0.20 * S - otm_amount + premium, 0.10 * S + premium)
    return round(margin * 100, 2)


def compute_iv_rank_from_history(current_iv: float, log_returns: pd.Series) -> float:
    rv_30d = log_returns.rolling(30).std() * np.sqrt(252)
    low_rv = rv_30d.min()
    high_rv = rv_30d.max()
    if high_rv <= low_rv:
        return 50.0
    return float(np.clip((current_iv - low_rv) / (high_rv - low_rv) * 100, 0, 100))


def run_quant_engine(
    contract: dict,
    log_returns: pd.Series,
    risk_free_rate: float = 0.053,
    n_paths: int = 10000,
) -> PopResult:
    """Run all quantitative models for a single contract. Returns PopResult."""
    S = contract['spot_price']
    K = contract['strike']
    IV = contract['implied_volatility'] or float(log_returns.std() * np.sqrt(252))
    dte = contract['dte']
    delta = contract['delta']
    premium = contract['mid']
    strategy = contract['strategy']
    option_type = 'put' if strategy == 'naked_put' else 'call'
    T = dte / 252

    p_delta = pop_delta(delta, option_type)
    p_bs = pop_black_scholes(S, K, risk_free_rate, IV, T, option_type)
    p_hist = pop_historical(S, K, dte, log_returns)
    p_garch = pop_garch_mc(S, K, dte, log_returns, n_paths)
    p_blended = blend_pop(p_delta, p_bs, p_hist, p_garch)

    stress = compute_stress_scenarios(S, K, premium, IV, option_type)

    max_stress_loss = abs(min(stress.stress_2sd, 0))
    ev = (premium * p_blended) - (max_stress_loss * (1 - p_blended))

    if option_type == 'put':
        breakeven = K - premium
    else:
        breakeven = K + premium

    margin = estimate_margin(S, K, premium, strategy)
    iv_rank = compute_iv_rank_from_history(IV, log_returns)

    return PopResult(
        pop_delta=p_delta,
        pop_bs=p_bs,
        pop_historical=p_hist,
        pop_garch_mc=p_garch,
        pop_blended=p_blended,
        stress=stress,
        expected_value=round(ev, 4),
        breakeven=round(breakeven, 2),
        margin_estimate=margin,
        iv_rank=iv_rank,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_quant_engine.py -v
```

Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add scanner/quant_engine.py tests/test_quant_engine.py
git commit -m "feat: quant engine — delta, BS, historical, GARCH-MC PoP + stress tests"
```

---

## Task 8: Hard Risk Filters (`scanner/risk_filters.py`)

**Files:**
- Create: `scanner/risk_filters.py`
- Create: `tests/test_risk_filters.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_filters.py
from scanner.risk_filters import apply_hard_filters, FilterResult

def _base_contract(**overrides):
    c = {
        'ticker': 'AAPL',
        'strategy': 'naked_put',
        'bid': 1.10,
        'ask': 1.30,
        'mid': 1.20,
        'volume': 500,
        'open_interest': 1000,
        'delta': -0.18,
        'implied_volatility': 0.28,
        'dte': 14,
        'spot_price': 195.0,
        'strike': 185.0,
    }
    c.update(overrides)
    return c

def _base_pop(**overrides):
    p = {
        'pop_blended': 0.82,
        'expected_value': 0.90,
        'stress': type('S', (), {'stress_2sd': -1.20})(),
    }
    p.update(overrides)
    return p

def test_good_contract_passes():
    result = apply_hard_filters(_base_contract(), _base_pop())
    assert result.passed is True
    assert result.reason == ''

def test_wide_spread_fails():
    c = _base_contract(bid=0.10, ask=1.50, mid=0.80)
    result = apply_hard_filters(c, _base_pop())
    assert result.passed is False
    assert 'spread' in result.reason.lower()

def test_low_volume_fails():
    result = apply_hard_filters(_base_contract(volume=50), _base_pop())
    assert result.passed is False
    assert 'volume' in result.reason.lower()

def test_low_oi_fails():
    result = apply_hard_filters(_base_contract(open_interest=100), _base_pop())
    assert result.passed is False
    assert 'open interest' in result.reason.lower()

def test_low_pop_fails():
    result = apply_hard_filters(_base_contract(), _base_pop(pop_blended=0.60))
    assert result.passed is False
    assert 'pop' in result.reason.lower()

def test_negative_ev_fails():
    result = apply_hard_filters(_base_contract(), _base_pop(expected_value=-0.50))
    assert result.passed is False
    assert 'expected value' in result.reason.lower()

def test_excessive_stress_loss_fails():
    # 2SD loss = -5.0, premium = 1.20, threshold = 3x premium = 3.60
    pop = _base_pop(stress=type('S', (), {'stress_2sd': -5.0})())
    result = apply_hard_filters(_base_contract(), pop)
    assert result.passed is False
    assert 'stress' in result.reason.lower()

def test_delta_out_of_range_fails():
    result = apply_hard_filters(_base_contract(delta=-0.50), _base_pop())
    assert result.passed is False
    assert 'delta' in result.reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_risk_filters.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/risk_filters.py**

```python
# scanner/risk_filters.py
from dataclasses import dataclass
import os

MAX_SPREAD_PCT = float(os.getenv('MAX_SPREAD_PCT', '0.20'))
MIN_VOLUME = int(os.getenv('MIN_VOLUME', '100'))
MIN_OI = int(os.getenv('MIN_OI', '500'))
MIN_POP = float(os.getenv('MIN_POP_THRESHOLD', '0.70'))
MAX_STRESS_MULTIPLIER = 3.0
DELTA_MIN = 0.05
DELTA_MAX = 0.35


@dataclass
class FilterResult:
    passed: bool
    reason: str  # empty string if passed


def apply_hard_filters(contract: dict, pop_result: dict | object) -> FilterResult:
    """
    Apply all hard filters. Any failure returns passed=False with reason.
    pop_result can be a PopResult dataclass or a dict with same keys.
    """
    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    bid = contract.get('bid', 0)
    ask = contract.get('ask', 0)
    mid = contract.get('mid', 0)
    volume = contract.get('volume', 0)
    oi = contract.get('open_interest', 0)
    delta = contract.get('delta', 0)
    premium = mid
    strategy = contract.get('strategy', 'naked_put')

    pop_blended = _get(pop_result, 'pop_blended', 0)
    ev = _get(pop_result, 'expected_value', 0)
    stress_obj = _get(pop_result, 'stress', None)
    stress_2sd = getattr(stress_obj, 'stress_2sd', 0) if stress_obj else 0

    # 1. Bid/ask spread
    if mid > 0:
        spread_pct = (ask - bid) / mid
        if spread_pct > MAX_SPREAD_PCT:
            return FilterResult(False, f"Bid/ask spread {spread_pct:.1%} exceeds {MAX_SPREAD_PCT:.0%} of mid")

    # 2. Volume
    if volume < MIN_VOLUME:
        return FilterResult(False, f"Volume {volume} below minimum {MIN_VOLUME}")

    # 3. Open interest
    if oi < MIN_OI:
        return FilterResult(False, f"Open interest {oi} below minimum {MIN_OI}")

    # 4. Delta range
    abs_delta = abs(delta)
    if not (DELTA_MIN <= abs_delta <= DELTA_MAX):
        return FilterResult(False, f"Delta {delta:.2f} outside target range [{DELTA_MIN},{DELTA_MAX}]")

    # 5. Blended PoP
    if pop_blended < MIN_POP:
        return FilterResult(False, f"Blended PoP {pop_blended:.1%} below minimum {MIN_POP:.0%}")

    # 6. Expected value
    if ev <= 0:
        return FilterResult(False, f"Expected value {ev:.4f} is not positive")

    # 7. Stress loss
    stress_loss = abs(min(stress_2sd, 0))
    if stress_loss > MAX_STRESS_MULTIPLIER * premium:
        return FilterResult(
            False,
            f"2SD stress loss ${stress_loss:.2f} exceeds {MAX_STRESS_MULTIPLIER}x premium ${premium:.2f}"
        )

    # 8. Naked call extra restrictions (delta check already done above)
    if strategy == 'naked_call':
        # No squeeze/meme names — checked at chain fetch level via exclusion list
        pass

    return FilterResult(True, '')
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_risk_filters.py -v
```

Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add scanner/risk_filters.py tests/test_risk_filters.py
git commit -m "feat: hard risk filters — spread, volume, OI, delta, PoP, EV, stress"
```

---

## Task 9: Composite Scorer (`scanner/scorer.py`)

**Files:**
- Create: `scanner/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorer.py
from scanner.scorer import compute_composite_score, assign_decision, build_scan_result

def _pop():
    class P:
        pop_delta = 0.85
        pop_bs = 0.83
        pop_historical = 0.80
        pop_garch_mc = 0.82
        pop_blended = 0.823
        expected_value = 1.05
        breakeven = 183.80
        margin_estimate = 1850.0
        iv_rank = 72.0
        class stress:
            stress_1sd = -0.30
            stress_2sd = -1.10
            stress_expiry = -2.00
    return P()

def _contract():
    return {
        'ticker': 'AAPL', 'strategy': 'naked_put', 'expiration': '2026-05-15',
        'strike': 185.0, 'bid': 1.10, 'ask': 1.30, 'mid': 1.20,
        'volume': 800, 'open_interest': 2000, 'implied_volatility': 0.28,
        'delta': -0.18, 'gamma': 0.05, 'theta': -0.04, 'vega': 0.12,
        'dte': 11, 'spot_price': 195.0,
    }

def test_composite_score_in_range():
    score = compute_composite_score(_contract(), _pop())
    assert 0 <= score <= 100

def test_trade_decision_above_threshold():
    assert assign_decision(score=70.0, hard_filter_passed=True) == 'TRADE'

def test_watchlist_decision():
    assert assign_decision(score=55.0, hard_filter_passed=True) == 'WATCHLIST'

def test_no_trade_when_filter_fails():
    assert assign_decision(score=80.0, hard_filter_passed=False) == 'NO TRADE'

def test_no_trade_when_score_too_low():
    assert assign_decision(score=40.0, hard_filter_passed=True) == 'NO TRADE'

def test_build_scan_result_has_all_keys():
    result = build_scan_result(_contract(), _pop(), hard_filter_passed=True, filter_reason='')
    required_keys = [
        'ticker', 'strategy', 'PoP_blended', 'composite_score',
        'decision', 'breakeven', 'margin_estimate', 'reason_for',
        'reason_against', 'stop_trigger',
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_scorer.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/scorer.py**

```python
# scanner/scorer.py
import os
import numpy as np

MIN_COMPOSITE_SCORE = int(os.getenv('MIN_COMPOSITE_SCORE', '65'))
WATCHLIST_FLOOR = 50


def compute_composite_score(contract: dict, pop_result) -> float:
    """
    Compute 0-100 composite trade score from contract + PopResult.
    Weights: PoP 30%, EV 20%, Premium/Margin 15%, Liquidity 15%, Spread 10%, Trend 10%
    """
    def _get(obj, key, default=0.0):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    pop_blended = _get(pop_result, 'pop_blended', 0)
    ev = _get(pop_result, 'expected_value', 0)
    margin = _get(pop_result, 'margin_estimate', 1)
    iv_rank = _get(pop_result, 'iv_rank', 50)

    premium = contract.get('mid', 0)
    volume = contract.get('volume', 0)
    oi = contract.get('open_interest', 0)
    bid = contract.get('bid', 0)
    ask = contract.get('ask', 0)
    mid = contract.get('mid', 1)

    # 1. PoP component (30%): scale 0.70-1.0 → 0-100
    pop_score = np.clip((pop_blended - 0.70) / 0.30 * 100, 0, 100) * 0.30

    # 2. EV component (20%): normalize EV vs premium
    ev_ratio = ev / premium if premium > 0 else 0
    ev_score = np.clip(ev_ratio * 100, 0, 100) * 0.20

    # 3. Premium/Margin ratio (15%): higher = better capital efficiency
    pm_ratio = (premium * 100) / margin if margin > 0 else 0
    pm_score = np.clip(pm_ratio * 200, 0, 100) * 0.15

    # 4. Liquidity (15%): based on volume and OI
    if volume >= 1000 and oi >= 5000:
        liq = 100.0
    elif volume >= 300 and oi >= 1000:
        liq = 70.0
    elif volume >= 100 and oi >= 500:
        liq = 40.0
    else:
        liq = 10.0
    liq_score = liq * 0.15

    # 5. Bid/ask tightness (10%): tighter = better
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
    spread_score = np.clip((1 - spread_pct / 0.20) * 100, 0, 100) * 0.10

    # 6. IV Rank alignment (10%): high IV rank = good for sellers
    iv_score = np.clip(iv_rank, 0, 100) * 0.10

    total = pop_score + ev_score + pm_score + liq_score + spread_score + iv_score
    return round(float(np.clip(total, 0, 100)), 2)


def assign_decision(score: float, hard_filter_passed: bool) -> str:
    """Return 'TRADE', 'WATCHLIST', or 'NO TRADE'."""
    if not hard_filter_passed:
        return 'NO TRADE'
    if score >= MIN_COMPOSITE_SCORE:
        return 'TRADE'
    if score >= WATCHLIST_FLOOR:
        return 'WATCHLIST'
    return 'NO TRADE'


def _build_reason_for(contract: dict, pop_result) -> str:
    reasons = []
    iv_rank = getattr(pop_result, 'iv_rank', None) or 0
    if iv_rank > 60:
        reasons.append(f"High IV rank ({iv_rank:.0f})")
    pop = getattr(pop_result, 'pop_blended', 0)
    if pop > 0.80:
        reasons.append(f"Strong PoP ({pop:.0%})")
    if contract.get('volume', 0) > 500:
        reasons.append("Liquid chain")
    dte = contract.get('dte', 0)
    if 7 <= dte <= 21:
        reasons.append(f"Optimal DTE ({dte})")
    return ', '.join(reasons) if reasons else 'Meets all quantitative criteria'


def _build_reason_against(contract: dict, pop_result) -> str:
    reasons = []
    strategy = contract.get('strategy', '')
    if strategy == 'naked_call':
        reasons.append('Theoretically unlimited upside risk')
    dte = contract.get('dte', 0)
    if dte < 7:
        reasons.append('Very short DTE — gamma risk elevated')
    iv = contract.get('implied_volatility', 0)
    if iv > 0.60:
        reasons.append('Very high IV — indicates elevated uncertainty')
    stress = getattr(pop_result, 'stress', None)
    if stress and getattr(stress, 'stress_2sd', 0) < -3.0:
        reasons.append('2SD stress scenario shows significant loss')
    return ', '.join(reasons) if reasons else 'Standard tail risk applies'


def _build_stop_trigger(contract: dict, pop_result) -> str:
    premium = contract.get('mid', 0)
    strategy = contract.get('strategy', '')
    strike = contract.get('strike', 0)
    spot = contract.get('spot_price', 0)
    stop_loss = round(premium * 2, 2)
    if strategy == 'naked_put':
        price_trigger = round(strike * 1.005, 2)
        return f"Close if loss > ${stop_loss} (2x premium) OR stock closes below ${price_trigger}"
    else:
        price_trigger = round(strike * 0.995, 2)
        return f"Close if loss > ${stop_loss} (2x premium) OR stock closes above ${price_trigger}"


def build_scan_result(
    contract: dict,
    pop_result,
    hard_filter_passed: bool,
    filter_reason: str,
) -> dict:
    """Assemble the full ScanResult dict from contract + PopResult."""
    score = compute_composite_score(contract, pop_result)
    decision = assign_decision(score, hard_filter_passed)

    stress = getattr(pop_result, 'stress', None)

    return {
        # Contract fields
        'ticker': contract.get('ticker'),
        'strategy': contract.get('strategy'),
        'expiration': contract.get('expiration'),
        'strike': contract.get('strike'),
        'bid': contract.get('bid'),
        'ask': contract.get('ask'),
        'mid': contract.get('mid'),
        'premium': contract.get('mid'),
        'delta': contract.get('delta'),
        'gamma': contract.get('gamma'),
        'theta': contract.get('theta'),
        'vega': contract.get('vega'),
        'IV': contract.get('implied_volatility'),
        'dte': contract.get('dte'),
        'spot_price': contract.get('spot_price'),
        # Quant fields
        'IV_rank': getattr(pop_result, 'iv_rank', None),
        'PoP_delta': getattr(pop_result, 'pop_delta', None),
        'PoP_BS': getattr(pop_result, 'pop_bs', None),
        'PoP_historical': getattr(pop_result, 'pop_historical', None),
        'PoP_GARCH_MC': getattr(pop_result, 'pop_garch_mc', None),
        'PoP_blended': getattr(pop_result, 'pop_blended', None),
        'expected_value': getattr(pop_result, 'expected_value', None),
        'breakeven': getattr(pop_result, 'breakeven', None),
        'margin_estimate': getattr(pop_result, 'margin_estimate', None),
        'stress_1SD': getattr(stress, 'stress_1sd', None) if stress else None,
        'stress_2SD': getattr(stress, 'stress_2sd', None) if stress else None,
        'stress_expiry': getattr(stress, 'stress_expiry', None) if stress else None,
        # Scoring
        'composite_score': score,
        'hard_filter_passed': hard_filter_passed,
        'hard_filter_reason': filter_reason,
        'decision': decision,
        # Narrative
        'reason_for': _build_reason_for(contract, pop_result),
        'reason_against': _build_reason_against(contract, pop_result),
        'stop_trigger': _build_stop_trigger(contract, pop_result),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_scorer.py -v
```

Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add scanner/scorer.py tests/test_scorer.py
git commit -m "feat: composite scorer, decision assignment, scan result builder"
```

---

## Task 10: File Exporter (`output/exporter.py`)

**Files:**
- Create: `output/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exporter.py
import json
import tempfile
from pathlib import Path
from output.exporter import export_scan

SAMPLE_RESULTS = [
    {
        'ticker': 'SPY', 'strategy': 'naked_put', 'expiration': '2026-05-15',
        'strike': 515.0, 'bid': 1.20, 'ask': 1.40, 'mid': 1.30,
        'premium': 1.30, 'delta': -0.15, 'IV': 0.18, 'IV_rank': 65.0,
        'PoP_blended': 0.84, 'expected_value': 1.05, 'composite_score': 72.0,
        'decision': 'TRADE', 'hard_filter_passed': True,
    }
]

def test_export_creates_csv_and_json(tmp_path):
    export_scan(SAMPLE_RESULTS, output_dir=tmp_path)
    files = list(tmp_path.glob('scan_*.csv'))
    assert len(files) == 1
    json_files = list(tmp_path.glob('scan_*.json'))
    assert len(json_files) == 1

def test_latest_json_written(tmp_path):
    export_scan(SAMPLE_RESULTS, output_dir=tmp_path)
    latest = tmp_path / 'latest.json'
    assert latest.exists()
    data = json.loads(latest.read_text())
    assert isinstance(data, dict)
    assert 'results' in data
    assert 'scan_time' in data
    assert data['results'][0]['ticker'] == 'SPY'

def test_csv_contains_data(tmp_path):
    export_scan(SAMPLE_RESULTS, output_dir=tmp_path)
    csv_file = list(tmp_path.glob('scan_*.csv'))[0]
    content = csv_file.read_text()
    assert 'SPY' in content
    assert 'TRADE' in content

def test_empty_results_still_writes_files(tmp_path):
    export_scan([], output_dir=tmp_path)
    assert (tmp_path / 'latest.json').exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_exporter.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement output/exporter.py**

```python
# output/exporter.py
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'scans'


def export_scan(results: list[dict], output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """
    Write scan results to a timestamped CSV + JSON file and update latest.json.
    output_dir is created if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    base_name = f'scan_{timestamp}'

    # Write timestamped CSV
    df = pd.DataFrame(results) if results else pd.DataFrame()
    csv_path = output_dir / f'{base_name}.csv'
    df.to_csv(csv_path, index=False)

    # Write timestamped JSON
    json_path = output_dir / f'{base_name}.json'
    json_path.write_text(json.dumps(results, default=str, indent=2))

    # Always update latest.json
    latest_path = output_dir / 'latest.json'
    latest_data = {
        'scan_time': datetime.now().isoformat(),
        'results': results,
    }
    latest_path.write_text(json.dumps(latest_data, default=str, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_exporter.py -v
```

Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add output/exporter.py tests/test_exporter.py
git commit -m "feat: CSV and JSON exporter with latest.json for dashboard"
```

---

## Task 11: Scan Orchestrator (`scanner/run_scan.py`)

**Files:**
- Create: `scanner/run_scan.py`
- Create: `tests/test_run_scan.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_scan.py
from unittest.mock import patch, MagicMock
from scanner.run_scan import run_full_scan

def test_run_full_scan_returns_list(tmp_path):
    mock_ts = MagicMock()
    mock_ts.ticker = 'SPY'
    mock_ts.current_price = 530.0
    mock_ts.current_iv = 0.18
    mock_ts.score = 75.0

    mock_contract = {
        'ticker': 'SPY', 'strategy': 'naked_put', 'expiration': '2026-05-15',
        'strike': 515.0, 'bid': 1.20, 'ask': 1.40, 'mid': 1.30,
        'volume': 800, 'open_interest': 2000, 'implied_volatility': 0.18,
        'delta': -0.15, 'gamma': 0.03, 'theta': -0.02, 'vega': 0.10,
        'dte': 11, 'spot_price': 530.0,
    }

    with patch('scanner.run_scan.screen_universe', return_value=[mock_ts]), \
         patch('scanner.run_scan.fetch_contracts', return_value=[mock_contract]), \
         patch('scanner.run_scan.yf.download') as mock_dl, \
         patch('scanner.run_scan.export_scan') as mock_export:

        import pandas as pd, numpy as np
        mock_dl.return_value = pd.DataFrame(
            {'Close': np.linspace(100, 130, 252)},
            index=pd.date_range('2025-05-01', periods=252)
        )
        results = run_full_scan(output_dir=tmp_path)

    assert isinstance(results, list)
    mock_export.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_scan.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scanner/run_scan.py**

```python
# scanner/run_scan.py
import os
from pathlib import Path
import numpy as np
import yfinance as yf

from scanner.screener import screen_universe
from scanner.chain_fetcher import fetch_contracts
from scanner.quant_engine import run_quant_engine
from scanner.risk_filters import apply_hard_filters
from scanner.scorer import build_scan_result
from output.exporter import export_scan

RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.053'))
MONTE_CARLO_PATHS = int(os.getenv('MONTE_CARLO_PATHS', '10000'))
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'scans'


def run_full_scan(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict]:
    """
    Full pipeline: screen → fetch chains → quant engine → filters → score → export.
    Returns list of ScanResult dicts, sorted by composite_score descending.
    """
    print("[Scan] Starting stock screen...")
    candidates = screen_universe()
    print(f"[Scan] {len(candidates)} candidates after screening.")

    all_results = []

    for ts in candidates:
        ticker = ts.ticker
        spot_price = ts.current_price

        # Fetch 1-year of daily returns for quant models
        try:
            hist = yf.download(ticker, period='1y', progress=False)['Close'].squeeze()
            log_returns = np.log(hist / hist.shift(1)).dropna()
        except Exception:
            continue

        # Fetch option contracts from Robinhood
        contracts = fetch_contracts(ticker, spot_price)
        if not contracts:
            continue

        for contract in contracts:
            try:
                pop_result = run_quant_engine(
                    contract=contract,
                    log_returns=log_returns,
                    risk_free_rate=RISK_FREE_RATE,
                    n_paths=MONTE_CARLO_PATHS,
                )
                filter_result = apply_hard_filters(contract, pop_result)
                scan_result = build_scan_result(
                    contract=contract,
                    pop_result=pop_result,
                    hard_filter_passed=filter_result.passed,
                    filter_reason=filter_result.reason,
                )
                all_results.append(scan_result)
            except Exception as e:
                print(f"[Scan] Error processing {ticker}: {e}")
                continue

    # Sort: TRADE first, then WATCHLIST, then NO TRADE; within each by composite_score
    decision_order = {'TRADE': 0, 'WATCHLIST': 1, 'NO TRADE': 2}
    all_results.sort(
        key=lambda x: (decision_order.get(x.get('decision', 'NO TRADE'), 2),
                       -x.get('composite_score', 0))
    )

    print(f"[Scan] Complete. {len(all_results)} contracts evaluated.")
    export_scan(all_results, output_dir=output_dir)
    return all_results
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_run_scan.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scanner/run_scan.py tests/test_run_scan.py
git commit -m "feat: scan orchestrator — full pipeline screen→chain→quant→filter→score→export"
```

---

## Task 12: Streamlit Dashboard (`output/dashboard.py`)

**Files:**
- Create: `output/dashboard.py`

- [ ] **Step 1: Implement output/dashboard.py**

```python
# output/dashboard.py
import json
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from scanner.clock import get_market_status, is_market_open
import yfinance as yf

LATEST_JSON = Path(__file__).parent / 'scans' / 'latest.json'
REFRESH_INTERVAL_SECONDS = 60

DECISION_COLORS = {
    'TRADE': '#00c853',
    'WATCHLIST': '#ffab00',
    'NO TRADE': '#d50000',
}

DISPLAY_COLUMNS = [
    'ticker', 'strategy', 'expiration', 'strike', 'bid', 'ask', 'mid',
    'delta', 'IV', 'IV_rank', 'PoP_blended', 'expected_value',
    'breakeven', 'composite_score', 'decision',
]

DETAIL_COLUMNS = [
    'PoP_delta', 'PoP_BS', 'PoP_historical', 'PoP_GARCH_MC', 'PoP_blended',
    'stress_1SD', 'stress_2SD', 'stress_expiry',
    'margin_estimate', 'reason_for', 'reason_against', 'stop_trigger',
]


def _load_latest() -> tuple[list[dict], str]:
    """Load latest.json. Returns (results, scan_time)."""
    if not LATEST_JSON.exists():
        return [], 'No scan yet'
    try:
        data = json.loads(LATEST_JSON.read_text())
        return data.get('results', []), data.get('scan_time', 'Unknown')
    except Exception:
        return [], 'Error loading results'


def _get_market_header() -> dict:
    """Fetch VIX and SPY for the header bar."""
    try:
        vix = float(yf.download('^VIX', period='2d', progress=False)['Close'].iloc[-1])
        spy = yf.download('SPY', period='5d', progress=False)['Close']
        spy_chg = float((spy.iloc[-1] / spy.iloc[-2] - 1) * 100)
        return {'vix': round(vix, 2), 'spy_chg': round(spy_chg, 2)}
    except Exception:
        return {'vix': None, 'spy_chg': None}


def _color_decision(val: str) -> str:
    color = DECISION_COLORS.get(val, '#ffffff')
    return f'background-color: {color}; color: white; font-weight: bold'


def render_dashboard() -> None:
    st.set_page_config(
        page_title='Options Scanner',
        page_icon='📊',
        layout='wide',
    )

    # Auto-refresh every REFRESH_INTERVAL_SECONDS
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0
    if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL_SECONDS:
        st.session_state.last_refresh = time.time()
        st.rerun()

    results, scan_time = _load_latest()
    market_status = get_market_status()
    header = _get_market_header()

    # ── Header ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Market', market_status)
    col2.metric('Last Scan', scan_time[:16] if len(scan_time) > 16 else scan_time)
    if header['vix']:
        col3.metric('VIX', header['vix'])
        spy_label = f"{'▲' if header['spy_chg'] >= 0 else '▼'} {abs(header['spy_chg']):.2f}%"
        col4.metric('SPY', spy_label)

    st.divider()

    if not results:
        if is_market_open():
            st.info('Scan in progress or no qualifying contracts found. Results update at 09:45, 11:00, 13:00, 15:00 ET.')
        else:
            st.warning('Market is closed. Showing last available scan results.')
        return

    df = pd.DataFrame(results)

    # ── Filter tabs ──────────────────────────────────────────────────────
    tab_all, tab_puts, tab_calls, tab_trade, tab_watch = st.tabs(
        ['All', 'Naked Puts', 'Naked Calls', 'TRADE Only', 'WATCHLIST']
    )

    def _render_table(filtered_df: pd.DataFrame, tab) -> None:
        if filtered_df.empty:
            tab.write('No results for this filter.')
            return

        display_df = filtered_df[
            [c for c in DISPLAY_COLUMNS if c in filtered_df.columns]
        ].copy()

        # Format percentages
        for col in ['PoP_blended', 'IV']:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f'{x:.1%}' if x else '')

        styled = display_df.style.applymap(
            _color_decision, subset=['decision']
        )
        tab.dataframe(styled, use_container_width=True, height=400)

        # Detail expander per row
        tab.markdown('**Click a ticker below for full detail:**')
        for _, row in filtered_df.iterrows():
            with tab.expander(
                f"{row.get('ticker')} | {row.get('strategy')} | "
                f"Strike {row.get('strike')} | {row.get('expiration')} | "
                f"{row.get('decision')}"
            ):
                det_col1, det_col2 = st.columns(2)

                det_col1.markdown('**Probability Models**')
                for k in ['PoP_delta', 'PoP_BS', 'PoP_historical', 'PoP_GARCH_MC', 'PoP_blended']:
                    v = row.get(k)
                    det_col1.write(f"{k}: {v:.1%}" if v is not None else f"{k}: N/A")

                det_col1.markdown('**Stress Scenarios**')
                for k in ['stress_1SD', 'stress_2SD', 'stress_expiry']:
                    v = row.get(k)
                    det_col1.write(f"{k}: ${v:.2f}" if v is not None else f"{k}: N/A")

                det_col2.markdown('**Risk & Margin**')
                det_col2.write(f"Margin estimate: ${row.get('margin_estimate', 'N/A')}")
                det_col2.write(f"Breakeven: ${row.get('breakeven', 'N/A')}")
                det_col2.write(f"Hard filter passed: {row.get('hard_filter_passed')}")
                if not row.get('hard_filter_passed'):
                    det_col2.warning(f"Filter reason: {row.get('hard_filter_reason')}")

                det_col2.markdown('**Trade Narrative**')
                det_col2.success(f"FOR: {row.get('reason_for', '')}")
                det_col2.error(f"AGAINST: {row.get('reason_against', '')}")
                det_col2.info(f"STOP: {row.get('stop_trigger', '')}")

    with tab_all:
        _render_table(df, tab_all)
    with tab_puts:
        _render_table(df[df['strategy'] == 'naked_put'], tab_puts)
    with tab_calls:
        _render_table(df[df['strategy'] == 'naked_call'], tab_calls)
    with tab_trade:
        _render_table(df[df['decision'] == 'TRADE'], tab_trade)
    with tab_watch:
        _render_table(df[df['decision'] == 'WATCHLIST'], tab_watch)

    st.divider()
    st.caption(
        'This is quantitative research tooling only — not financial advice. '
        'Naked options carry theoretically unlimited risk. Always verify independently before acting.'
    )
```

- [ ] **Step 2: Smoke test — verify dashboard renders without errors**

```bash
streamlit run output/dashboard.py
```

Expected: Browser opens, shows "No scan yet" message, no Python errors in terminal.

- [ ] **Step 3: Commit**

```bash
git add output/dashboard.py
git commit -m "feat: Streamlit dashboard with market header, filter tabs, detail expanders"
```

---

## Task 13: Scheduler + Main Entry Point (`scanner/scheduler.py` + `main.py`)

**Files:**
- Create: `scanner/scheduler.py`
- Create: `main.py`

- [ ] **Step 1: Implement scanner/scheduler.py**

```python
# scanner/scheduler.py
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

_scheduler = None
_lock = threading.Lock()


def get_or_create_scheduler():
    """Singleton APScheduler. Safe to call from multiple Streamlit reruns."""
    global _scheduler
    with _lock:
        if _scheduler is None:
            from scanner.run_scan import run_full_scan
            tz = pytz.timezone('America/New_York')
            _scheduler = BackgroundScheduler(timezone=tz)
            for hour, minute in [(9, 45), (11, 0), (13, 0), (15, 0)]:
                _scheduler.add_job(
                    run_full_scan,
                    trigger='cron',
                    hour=hour,
                    minute=minute,
                    day_of_week='mon-fri',
                    id=f'scan_{hour:02d}{minute:02d}',
                    replace_existing=True,
                )
            _scheduler.start()
            print('[Scheduler] Started. Scans at 09:45, 11:00, 13:00, 15:00 ET (Mon-Fri).')
    return _scheduler
```

- [ ] **Step 2: Implement main.py**

```python
# main.py
"""
Entry point for the Options Scanner.
Run with: streamlit run main.py

The scheduler starts in a background thread and triggers scans at:
  09:45, 11:00, 13:00, 15:00 ET (Monday-Friday)

The Streamlit dashboard reads from output/scans/latest.json and auto-refreshes.
"""
import streamlit as st

# Start scheduler once per process (singleton — safe across Streamlit reruns)
if 'scheduler_initialized' not in st.session_state:
    from scanner.scheduler import get_or_create_scheduler
    get_or_create_scheduler()
    st.session_state.scheduler_initialized = True

# Render the dashboard
from output.dashboard import render_dashboard
render_dashboard()
```

- [ ] **Step 3: Smoke test — run the full application**

```bash
streamlit run main.py
```

Expected:
- Browser opens showing the dashboard
- Terminal shows `[Scheduler] Started.`
- Dashboard shows market status, VIX, SPY data
- At the next scheduled scan time, terminal shows `[Scan] Starting stock screen...`

- [ ] **Step 4: Commit**

```bash
git add scanner/scheduler.py main.py
git commit -m "feat: scheduler singleton + main entry point (streamlit run main.py)"
```

---

## Task 14: Full Test Suite + Final Verification

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass. Note any failures and fix before proceeding.

- [ ] **Step 2: Run a manual on-demand scan to verify the pipeline end-to-end**

```python
# In a Python REPL or temporary script:
from dotenv import load_dotenv
load_dotenv()
from scanner.run_scan import run_full_scan
results = run_full_scan()
print(f"Total contracts evaluated: {len(results)}")
trades = [r for r in results if r['decision'] == 'TRADE']
print(f"TRADE decisions: {len(trades)}")
if trades:
    print(trades[0])
```

Expected: Results print without errors. If no TRADE decisions, NO TRADE is also valid output.

- [ ] **Step 3: Verify CSV and JSON export files exist**

```bash
ls -la output/scans/
```

Expected: `latest.json`, `scan_YYYYMMDD_HHMM.csv`, `scan_YYYYMMDD_HHMM.json` all present.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: options scanner complete — full pipeline, tests, dashboard"
```

---

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Robinhood email and password

# 3. Run the scanner (starts scheduler + opens dashboard)
streamlit run main.py

# 4. Run tests
pytest tests/ -v

# 5. Trigger a manual scan (outside scheduled times)
python -c "from dotenv import load_dotenv; load_dotenv(); from scanner.run_scan import run_full_scan; run_full_scan()"
```

---

## Disclaimer

This tool is quantitative research tooling only. It is not financial advice. Naked options positions carry theoretically unlimited risk. The outputs of this scanner are inputs to your own research — never act on them without independent verification, proper position sizing, and appropriate risk management.
