"""v2 scanner P&L backtest — real option prices via Schwab historical data.

For each compression_breakout setup (strength >= 0.70, VIX != expansion)
over the last 90 trading days, this script:

1. Identifies the closest-to-ATM strike from the Sep 18, 2026 monthly option chain
   (the shortest-dated contract class with full 90-day price history on Schwab).
2. Fetches OHLC price history for that contract.
3. Simulates hold logic (entry=day-of mid, exit=50% stop/50% target/5-day time stop).
4. Accounts for $1.50 commission + $0.05/share slippage per side.

CONTRACT SELECTION NOTE:
  Schwab's /pricehistory only returns data for CURRENTLY LIVE contracts.
  Short-dated weeklies (5-10 DTE) that were ATM at setup time are now expired —
  zero historical candles are returned. The Sep 18, 2026 monthly options were
  live during the entire backtest window (data goes back to ~Feb 2, 2026).
  These are longer DTE (~100-200 DTE at trade time) and thus more expensive than
  the originally spec'd weeklies, but they are the only real price data available.
  Costs: slippage is $0.05/side (same assumption, wider spreads on monthlies partially
  compensated by the deeper bid stacks on high-OI monthly strikes).

Outputs:
  - Console P&L summary table
  - CSV per-trade detail → output/backtest/scanner_v2_pnl_YYYYMMDD_HHMM.csv

CLI:
    python3 -m scripts.scanner_v2_pnl_backtest
    python3 -m scripts.scanner_v2_pnl_backtest --days 60 --strength-min 0.65
    python3 -m scripts.scanner_v2_pnl_backtest --no-cache
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env before importing Schwab auth
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from domain.technical_signals import detect_setups
from domain.equity.vix_regime import compute_vix_context
from data.sources.yahoo_macro_source import YahooMacroSource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v2 universe: v1 tickers minus underperformers (XLP, XLF, XLV, XLU)
V2_UNIVERSE = [
    "DIA", "IWM", "SPY", "QQQ", "XLY", "XLI",
    "AMD", "NVDA", "AAPL", "MSFT", "GOOGL", "META", "V",
]

_DEFAULT_OUTPUT_DIR = _ROOT / "output" / "backtest"
_OHLCV_CACHE_DIR = _ROOT / "cache" / "backtest_ohlcv"
_OPTION_CACHE_DIR = _ROOT / "cache" / "option_price_history"

# Contract vehicle: Sep 18 2026 monthly options — the shortest-dated class with
# full 90-day price history on Schwab's /pricehistory endpoint.
# (Weeklies/short-dated monthlies that expired during the backtest window return
#  zero historical candles; the Sep18 monthly has data from ~Feb 2, 2026.)
_CONTRACT_EXPIRATION = date(2026, 9, 18)

# P&L constants (per contract)
_FEES_PER_SIDE = 1.50        # commission per contract per side
_SLIPPAGE_PER_SIDE = 5.00    # $0.05 per share × 100 shares per side (liquid monthly)
_FEES_ROUND_TRIP = (_FEES_PER_SIDE + _SLIPPAGE_PER_SIDE) * 2  # = $13.00

# Exit thresholds
_STOP_MULT = 0.50            # exit if low <= 50% of entry
_TARGET_MULT = 1.50          # exit if high >= 150% of entry
_MAX_HOLD_DAYS = 5

# Rate limiting: 120 req/min → 0.5s sleep between requests
_MIN_SECONDS_BETWEEN_REQUESTS = 0.52
_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# NYSE trading calendar
# ---------------------------------------------------------------------------

def _us_federal_holidays(year: int) -> set[date]:
    holidays: set[date] = set()

    def _nearest_weekday(d: date) -> date:
        wd = d.weekday()
        if wd == 5:
            return d - timedelta(days=1)
        if wd == 6:
            return d + timedelta(days=1)
        return d

    holidays.add(_nearest_weekday(date(year, 1, 1)))
    mondays_jan = [date(year, 1, day) for day in range(1, 32) if date(year, 1, day).weekday() == 0]
    if len(mondays_jan) >= 3:
        holidays.add(mondays_jan[2])
    mondays_feb = [date(year, 2, day) for day in range(1, 29 if year % 4 != 0 else 30) if date(year, 2, day).weekday() == 0]
    if len(mondays_feb) >= 3:
        holidays.add(mondays_feb[2])
    a = year % 19; b = year // 100; c = year % 100
    d_v = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d_v - g + 15) % 30; i = c // 4; k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * ll) // 451
    month_n = (h + ll - 7 * m + 114) // 31
    day_num = ((h + ll - 7 * m + 114) % 31) + 1
    easter = date(year, month_n, day_num)
    holidays.add(easter - timedelta(days=2))
    mondays_may = [date(year, 5, day) for day in range(1, 32) if date(year, 5, day).weekday() == 0]
    if mondays_may:
        holidays.add(mondays_may[-1])
    if year >= 2022:
        holidays.add(_nearest_weekday(date(year, 6, 19)))
    holidays.add(_nearest_weekday(date(year, 7, 4)))
    mondays_sep = [date(year, 9, day) for day in range(1, 31) if date(year, 9, day).weekday() == 0]
    if mondays_sep:
        holidays.add(mondays_sep[0])
    thursdays_nov = [date(year, 11, day) for day in range(1, 31) if date(year, 11, day).weekday() == 3]
    if len(thursdays_nov) >= 4:
        holidays.add(thursdays_nov[3])
    holidays.add(_nearest_weekday(date(year, 12, 25)))
    return holidays


def get_trading_days(start: date, end: date) -> list[date]:
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
        return [d.date() for d in schedule.index]
    except Exception:
        pass
    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        holidays |= _us_federal_holidays(year)
    days: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5 and cur not in holidays:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Wilson CI helper
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    z = 1.96
    p = wins / trials
    n = trials
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Stock OHLCV cache (reuse v1 parquets)
# ---------------------------------------------------------------------------

def fetch_stock_ohlcv(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    cp = _OHLCV_CACHE_DIR / f"{ticker.replace('-', '_')}.parquet"
    if use_cache and cp.exists():
        df = pd.read_parquet(cp)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
        if raw.empty:
            raise ValueError("empty")
        df = raw.rename(columns={c: c.lower() for c in raw.columns})
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        keep = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].astype(float)
        df = df.dropna(subset=["close"])
        _OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp)
        return df
    except Exception as exc:
        print(f"    {ticker}: stock OHLCV fetch failed: {exc}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# VIX fetch
# ---------------------------------------------------------------------------

def fetch_vix() -> pd.Series:
    macro = YahooMacroSource()
    df = macro.fetch_history("^VIX", period="2y")
    if df.empty:
        return pd.Series(dtype=float)
    close = df["close"].copy()
    if close.index.tz is not None:
        close.index = close.index.tz_convert("UTC").tz_localize(None)
    return close


# ---------------------------------------------------------------------------
# Option contract symbol builder
# ---------------------------------------------------------------------------

def build_option_symbol(ticker: str, expiration: date, is_call: bool, strike: float) -> str:
    """Build Schwab option symbol: TICKER[padded to 6]YYMMDDCP00STRIKE000

    Example: AAPL 290 call 2026-09-18 → 'AAPL  260918C00290000'
    Example: SPY 560 call 2026-09-18  → 'SPY   260918C00560000'
    """
    ticker_padded = ticker.ljust(6)
    date_str = expiration.strftime("%y%m%d")
    cp_char = "C" if is_call else "P"
    strike_int = round(strike * 1000)
    strike_str = f"{strike_int:08d}"
    return f"{ticker_padded}{date_str}{cp_char}{strike_str}"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_time = time.time()


# ---------------------------------------------------------------------------
# Sep 2026 chain fetcher: get available strikes near a target spot price
# ---------------------------------------------------------------------------

def get_sep_chain(
    ticker: str,
    target_spot: float,
    is_call: bool,
    schwab_source,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch Sep 18 2026 option chain for ticker, returning a DataFrame of strikes.

    Cached to avoid re-fetching the same ticker's chain multiple times.
    Returns DataFrame with columns [strike, open_interest, symbol].
    """
    safe_ticker = ticker.replace("-", "_")
    cp_type = "C" if is_call else "P"
    cp = _OPTION_CACHE_DIR / f"sep_chain_{safe_ticker}_{cp_type}.parquet"

    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    _rate_limit()
    try:
        exp_date = _CONTRACT_EXPIRATION
        params = {
            "symbol": ticker,
            "contractType": "CALL" if is_call else "PUT",
            "strikeCount": 60,
            "includeUnderlyingQuote": "false",
            "range": "ALL",
            "fromDate": exp_date.isoformat(),
            "toDate": exp_date.isoformat(),
        }
        data = schwab_source._request("/chains", params)
        opt_key = "callExpDateMap" if is_call else "putExpDateMap"
        rows = []
        for exp_key, strike_map in (data.get(opt_key) or {}).items():
            exp_part = exp_key.split(":")[0]
            try:
                exp_dt = date.fromisoformat(exp_part)
            except ValueError:
                continue
            if exp_dt != exp_date:
                continue
            for strike_str, contracts in strike_map.items():
                if not contracts:
                    continue
                c = contracts[0]
                try:
                    rows.append({
                        "strike": float(strike_str),
                        "open_interest": int(c.get("openInterest") or 0),
                        "symbol": c.get("symbol", ""),
                    })
                except (ValueError, TypeError):
                    continue

        if not rows:
            return pd.DataFrame(columns=["strike", "open_interest", "symbol"])

        df = pd.DataFrame(rows)
        _OPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp)
        return df

    except Exception as exc:
        print(f"    get_sep_chain({ticker}): {exc}")
        return pd.DataFrame(columns=["strike", "open_interest", "symbol"])


def find_atm_strike_in_chain(
    chain_df: pd.DataFrame,
    target_spot: float,
) -> Optional[tuple[float, str]]:
    """Given a chain DataFrame, find the closest-to-ATM strike with OI > 0.

    Returns (strike, symbol) or None.
    """
    if chain_df.empty:
        return None

    with_oi = chain_df[chain_df["open_interest"] > 0].copy()
    if with_oi.empty:
        with_oi = chain_df.copy()  # fall back to all strikes

    with_oi["dist"] = (with_oi["strike"] - target_spot).abs()
    best = with_oi.nsmallest(1, "dist")
    if best.empty:
        return None
    return float(best["strike"].iloc[0]), str(best["symbol"].iloc[0])


# ---------------------------------------------------------------------------
# Option price history fetch
# ---------------------------------------------------------------------------

def fetch_option_price_history(
    option_symbol: str,
    start_date: date,
    end_date: date,
    schwab_source,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLC daily candles for an option contract.

    Cache key: symbol + date window. Returns empty DataFrame if no data.
    """
    safe_sym = option_symbol.strip().replace(" ", "_")
    cache_key = f"{safe_sym}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    cp = _OPTION_CACHE_DIR / f"hist_{cache_key}.parquet"

    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    _rate_limit()
    try:
        start_ms = int(datetime(start_date.year, start_date.month, start_date.day,
                                tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(end_date.year, end_date.month, end_date.day, 23, 59,
                              tzinfo=timezone.utc).timestamp() * 1000)
        params = {
            "symbol": option_symbol,
            "periodType": "year",
            "frequencyType": "daily",
            "frequency": 1,
            "startDate": start_ms,
            "endDate": end_ms,
            "needExtendedHoursData": "false",
        }
        data = schwab_source._request("/pricehistory", params)
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
        df = df.astype(float)
        df.index = df.index.tz_localize(None).normalize()

        _OPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp)
        return df

    except Exception as exc:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def simulate_trade(
    entry_price: float,
    option_history: pd.DataFrame,
    setup_date: date,
    trading_days: list[date],
) -> dict:
    """Simulate hold from setup_date, check each subsequent trading day's OHLC.

    Returns dict: exit_price, exit_reason, hold_days, gross_pnl, net_pnl.
    """
    stop_price = entry_price * _STOP_MULT
    target_price = entry_price * _TARGET_MULT

    # Trading days strictly after setup date (the hold period)
    hold_tds = [d for d in trading_days if d > setup_date][:_MAX_HOLD_DAYS]

    exit_price = None
    exit_reason = "time"
    hold_days = 0
    last_close = entry_price  # fallback if we run out of data before _MAX_HOLD_DAYS

    for day in hold_tds:
        hold_days += 1
        day_ts = pd.Timestamp(day)
        day_rows = option_history[option_history.index == day_ts]

        if day_rows.empty:
            # No data for this day — skip, treat as "market closed" or missing
            continue

        row = day_rows.iloc[0]
        day_low = float(row["low"])
        day_high = float(row["high"])
        day_close = float(row["close"])
        last_close = day_close

        # Stop loss check (worst case intraday)
        if day_low <= stop_price:
            exit_price = stop_price
            exit_reason = "stop"
            break

        # Profit target check
        if day_high >= target_price:
            exit_price = target_price
            exit_reason = "target"
            break

        # Final day → time exit at close
        if hold_days == _MAX_HOLD_DAYS:
            exit_price = day_close
            exit_reason = "time"
            break

    if exit_price is None:
        # Ran out of trading days (some days had missing data)
        exit_price = last_close
        exit_reason = "time"

    gross_pnl = (exit_price - entry_price) * 100  # per contract
    net_pnl = gross_pnl - _FEES_ROUND_TRIP

    return {
        "exit_price": round(exit_price, 4),
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
    }


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_v2_backtest(
    universe: list[str],
    days: int,
    output_dir: Path,
    use_cache: bool,
    strength_min: float,
) -> tuple:
    output_dir.mkdir(parents=True, exist_ok=True)
    _OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _OPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today()
    end_date = today
    start_date = today - timedelta(days=days + 40)

    trading_days_all = get_trading_days(start_date, end_date)
    trading_days = trading_days_all[-days:] if len(trading_days_all) > days else trading_days_all
    bt_start = trading_days[0] if trading_days else start_date
    bt_end = trading_days[-1] if trading_days else end_date

    print(f"\n{'='*72}")
    print(f"=== v2 SCANNER P&L BACKTEST ({bt_start} → {bt_end}) ===")
    print(f"{'='*72}")
    print(f"\nFilters applied:")
    print(f"  Universe:           {len(universe)} tickers ({', '.join(universe)})")
    print(f"  Setup type:         compression_breakout only")
    print(f"  Strength threshold: >= {strength_min}")
    print(f"  VIX regime filter:  Skip expansion regimes")
    print(f"  Hold horizon:       {_MAX_HOLD_DAYS} days (with 50% stop / 50% target / time exit)")
    print(f"  Contract vehicle:   Sep 18, 2026 monthly ATM (closest-to-spot strike w/ OI>0)")
    print(f"  Fees + slippage:    ${_FEES_ROUND_TRIP:.2f} round-trip per contract")
    print(f"\n  NOTE: Schwab /pricehistory only returns data for currently-live contracts.")
    print(f"  Weekly options from the backtest window have since expired (zero candles).")
    print(f"  Sep 18, 2026 monthly options have data from ~Feb 2, 2026 → full coverage.\n")

    # Initialize Schwab
    print("Initializing Schwab API ...")
    try:
        from data.sources.schwab_source import SchwabSource
        schwab = SchwabSource()
        schwab._auth.get_access_token()
        print("  Schwab auth: OK\n")
    except Exception as exc:
        print(f"  ERROR: Schwab auth failed: {exc}")
        sys.exit(1)

    # Fetch VIX
    print("Fetching VIX history ...")
    try:
        vix_series = fetch_vix()
        if vix_series.index.tz is not None:
            vix_series.index = vix_series.index.tz_convert("UTC").tz_localize(None)
        print(f"  VIX: {len(vix_series)} daily rows\n")
    except Exception as exc:
        print(f"  VIX fetch failed: {exc}")
        vix_series = pd.Series(dtype=float)

    # Load stock OHLCV
    print(f"Loading stock OHLCV for {len(universe)} tickers ...")
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    for ticker in universe:
        try:
            df = fetch_stock_ohlcv(ticker, use_cache=use_cache)
            if not df.empty:
                ohlcv_cache[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
        except Exception as exc:
            print(f"  {ticker}: failed: {exc}")
    print(f"\n  {len(ohlcv_cache)}/{len(universe)} tickers loaded\n")

    # --- Phase 1: Detect qualifying setups ---
    print(f"Phase 1: Detecting compression_breakout setups (strength >= {strength_min}) ...")
    _MIN_HISTORY = 220

    setups_found: list[dict] = []
    n_skipped_by_vix = 0

    for as_of in trading_days:
        as_of_ts = pd.Timestamp(as_of)

        vix_slice = vix_series[vix_series.index.normalize() <= as_of_ts] if len(vix_series) > 0 else pd.Series(dtype=float)
        vix_context = compute_vix_context(vix_slice.tail(35)) if len(vix_slice) >= 10 else None
        vix_regime = vix_context.regime if vix_context else "neutral"
        vix_now = vix_context.vix_now if vix_context else float("nan")

        for ticker, full_df in ohlcv_cache.items():
            history_slice = full_df[full_df.index.normalize() <= as_of_ts]
            if len(history_slice) < _MIN_HISTORY:
                continue
            try:
                detected = detect_setups(history_slice)
            except Exception:
                continue

            for setup in detected:
                if setup.setup_name != "compression_breakout":
                    continue
                if setup.strength < strength_min:
                    continue

                if vix_regime == "expansion":
                    n_skipped_by_vix += 1
                    continue

                spot = float(history_slice["close"].iloc[-1])
                setups_found.append({
                    "setup_date": as_of,
                    "ticker": ticker,
                    "direction": setup.direction,
                    "strength": round(setup.strength, 4),
                    "spot_at_setup": round(spot, 4),
                    "vix_at_setup": round(vix_now, 2) if not math.isnan(vix_now) else None,
                    "vix_regime": vix_regime,
                })

    setups_traded = len(setups_found)
    print(f"  Setups found (passing VIX filter): {setups_traded}")
    print(f"  Setups skipped (VIX expansion):    {n_skipped_by_vix}\n")

    if not setups_found:
        print("No qualifying setups found.")
        return [], 0, 0, 0, bt_start, bt_end

    # --- Phase 2: Fetch Sep chain once per ticker ---
    print(f"Phase 2: Fetching Sep 18 2026 option chains for {len(universe)} tickers ...")
    sep_chains_call: dict[str, pd.DataFrame] = {}
    sep_chains_put: dict[str, pd.DataFrame] = {}

    for ticker in universe:
        if ticker not in ohlcv_cache:
            continue
        # Get last known spot for context (just for chain search range; actual spot used is historical)
        try:
            call_chain = get_sep_chain(ticker, 0, True, schwab, use_cache=use_cache)
            sep_chains_call[ticker] = call_chain
            print(f"  {ticker} CALL chain: {len(call_chain)} strikes")
        except Exception as exc:
            print(f"  {ticker} CALL chain error: {exc}")
        time.sleep(0.1)
        try:
            put_chain = get_sep_chain(ticker, 0, False, schwab, use_cache=use_cache)
            sep_chains_put[ticker] = put_chain
        except Exception as exc:
            print(f"  {ticker} PUT chain error: {exc}")

    print()

    # --- Phase 3: Per-setup option history fetch + simulate ---
    print(f"Phase 3: Fetching option price history for {setups_traded} setups ...")
    print(f"  Rate limit: {_MIN_SECONDS_BETWEEN_REQUESTS}s between Schwab requests")
    print(f"  Estimated API calls: up to {setups_traded} (many will cache-hit after first run)\n")

    trade_records: list[dict] = []
    skipped_no_data = 0
    depth_warnings: list[str] = []

    # Track which symbols we've already fetched (avoid redundant requests for same contract)
    fetched_histories: dict[str, pd.DataFrame] = {}

    for idx, s in enumerate(setups_found):
        if idx % 25 == 0:
            pct = idx / len(setups_found) * 100
            print(f"  [{idx+1}/{len(setups_found)} = {pct:.0f}%] processed ...")

        setup_date: date = s["setup_date"]
        ticker: str = s["ticker"]
        spot: float = s["spot_at_setup"]
        direction: str = s["direction"]
        is_call = (direction == "bullish")

        # Get chain
        chain_df = sep_chains_call.get(ticker) if is_call else sep_chains_put.get(ticker)
        if chain_df is None or chain_df.empty:
            skipped_no_data += 1
            continue

        # Find ATM strike at historical spot
        result = find_atm_strike_in_chain(chain_df, spot)
        if result is None:
            skipped_no_data += 1
            continue
        strike, opt_symbol = result

        # Fetch price history for this contract (cached by symbol)
        if opt_symbol not in fetched_histories:
            hist_start = bt_start - timedelta(days=5)  # small buffer
            hist_end = min(_CONTRACT_EXPIRATION, today)
            history = fetch_option_price_history(
                opt_symbol, hist_start, hist_end, schwab, use_cache=use_cache
            )
            fetched_histories[opt_symbol] = history
        else:
            history = fetched_histories[opt_symbol]

        if history.empty:
            skipped_no_data += 1
            depth_warnings.append(f"  No data: {opt_symbol} (setup {setup_date})")
            continue

        # Entry price: mid of (high + low) on setup date
        setup_ts = pd.Timestamp(setup_date)
        setup_rows = history[history.index == setup_ts]

        if setup_rows.empty:
            # Try the previous trading day (in case of holiday/weekend)
            prev_tds = [d for d in trading_days if d <= setup_date][-3:]
            for prev_d in reversed(prev_tds):
                prev_ts = pd.Timestamp(prev_d)
                setup_rows = history[history.index == prev_ts]
                if not setup_rows.empty:
                    break

        if setup_rows.empty:
            skipped_no_data += 1
            depth_warnings.append(f"  No entry candle: {opt_symbol} on {setup_date}")
            continue

        entry_row = setup_rows.iloc[0]
        h = float(entry_row["high"])
        l_price = float(entry_row["low"])
        if h <= 0 or l_price <= 0:
            skipped_no_data += 1
            continue
        entry_price = (h + l_price) / 2.0

        if entry_price <= 0:
            skipped_no_data += 1
            continue

        dte = (_CONTRACT_EXPIRATION - setup_date).days

        # Simulate the trade
        sim = simulate_trade(entry_price, history, setup_date, trading_days)

        trade_records.append({
            "setup_date": setup_date.isoformat(),
            "ticker": ticker,
            "setup_strength": s["strength"],
            "vix_at_setup": s["vix_at_setup"],
            "contract_symbol": opt_symbol,
            "expiration": _CONTRACT_EXPIRATION.isoformat(),
            "strike": strike,
            "dte": dte,
            "entry_price": round(entry_price, 4),
            "exit_price": sim["exit_price"],
            "exit_reason": sim["exit_reason"],
            "hold_days": sim["hold_days"],
            "gross_pnl": sim["gross_pnl"],
            "fees_slippage": _FEES_ROUND_TRIP,
            "net_pnl": sim["net_pnl"],
            "cumulative_pnl": None,
        })

    # Fill cumulative P&L
    cumulative = 0.0
    for r in trade_records:
        cumulative += r["net_pnl"]
        r["cumulative_pnl"] = round(cumulative, 2)

    print(f"\n  Trades executed:   {len(trade_records)}")
    print(f"  Skipped (no data): {skipped_no_data}")

    if depth_warnings:
        print(f"\n  Data warnings ({len(depth_warnings)} setups):")
        for w in depth_warnings[:10]:
            print(w)
        if len(depth_warnings) > 10:
            print(f"  ... and {len(depth_warnings)-10} more")

    return trade_records, setups_traded, n_skipped_by_vix, skipped_no_data, bt_start, bt_end


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    trades: list[dict],
    setups_found: int,
    setups_skipped_vix: int,
    setups_skipped_no_data: int,
    bt_start: date,
    bt_end: date,
    universe: list[str],
    strength_min: float,
) -> None:
    sep = "─" * 72

    print(f"\n{'='*72}")
    print(f"=== v2 SCANNER P&L BACKTEST ({bt_start} → {bt_end}) ===")
    print(f"{'='*72}\n")

    print("Filters applied:")
    print(f"  Universe:           {len(universe)} tickers ({', '.join(universe)})")
    print(f"  Setup type:         compression_breakout only")
    print(f"  Strength threshold: >= {strength_min}")
    print(f"  VIX regime filter:  Skip expansion regimes")
    print(f"  Hold horizon:       {_MAX_HOLD_DAYS} days (with 50% stop / 50% target / time exit)")
    print(f"  Contract vehicle:   Sep 18, 2026 monthly (closest ATM strike w/ OI>0)")
    print(f"\n  Schwab data note:   Short-dated weeklies (expired) return zero candles.")
    print(f"  Sep18 monthly used — full 90-day price history available from Schwab.\n")

    print(f"Setups found:    {setups_found + setups_skipped_vix}")
    print(f"Setups skipped:  {setups_skipped_vix} (VIX expansion)")
    print(f"Setups traded:   {setups_found}")
    print(f"Setups skipped:  {setups_skipped_no_data} (no contract data available)")
    print(f"Total trades:    {len(trades)}")

    if not trades:
        print("\nNo trades executed.")
        return

    print(f"\n{sep}")
    print("PER-TRADE OUTCOMES")
    print(sep)

    net_pnls = [r["net_pnl"] for r in trades]
    n = len(trades)
    n_wins = sum(1 for p in net_pnls if p > 0)
    win_rate = n_wins / n * 100 if n > 0 else 0.0
    ci = wilson_ci(n_wins, n)

    mean_pnl = float(np.mean(net_pnls))
    median_pnl = float(np.median(net_pnls))
    best_trade = max(net_pnls)
    worst_trade = min(net_pnls)
    total_pnl = sum(net_pnls)

    starting_capital = 1000.0
    account_return = (total_pnl / starting_capital) * 100

    bt_calendar_days = max((bt_end - bt_start).days, 1)
    if total_pnl > 0 and starting_capital > 0:
        annualized = ((1 + total_pnl / starting_capital) ** (365 / bt_calendar_days) - 1) * 100
    else:
        annualized = account_return * (365 / bt_calendar_days)

    daily_arr = np.array(net_pnls)
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown from cumulative P&L series
    cum_series = [r["cumulative_pnl"] for r in trades]
    max_dd = 0.0
    peak = 0.0
    for c in cum_series:
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / starting_capital) * 100 if starting_capital > 0 else 0.0

    print(f"  Total trades:        {n}")
    print(f"  Wins:                {n_wins}")
    print(f"  Losses:              {n - n_wins}")
    print(f"  Win rate:            {win_rate:.1f}% (95% CI [{ci[0]*100:.1f}, {ci[1]*100:.1f}])")
    print(f"")
    print(f"  Mean P&L per trade:  ${mean_pnl:+.2f}")
    print(f"  Median P&L:          ${median_pnl:+.2f}")
    print(f"  Best trade:          ${best_trade:+.2f}")
    print(f"  Worst trade:         ${worst_trade:+.2f}")
    print(f"")
    print(f"  Cumulative P&L:      ${total_pnl:+.2f}")
    print(f"  Starting capital:    ${starting_capital:,.0f}")
    print(f"  Account return:      {account_return:+.1f}%")
    print(f"  Annualized return:   {annualized:+.1f}%")
    print(f"  Sharpe (annualized): {sharpe:.2f}")
    print(f"  Max drawdown:        -${max_dd:.2f} (-{max_dd_pct:.1f}%)")

    print(f"\n{sep}")
    print("EXIT REASON BREAKDOWN")
    print(sep)
    for reason, label in [("target", "Profit target hit (+50%)"),
                           ("stop",   "Stop loss hit (-50%)"),
                           ("time",   "Time exit at 5d")]:
        n_r = sum(1 for r in trades if r["exit_reason"] == reason)
        pct = n_r / n * 100 if n > 0 else 0.0
        print(f"  {label:<35}: {n_r:>4} ({pct:.1f}%)")

    print(f"\n{sep}")
    print("PER TICKER")
    print(sep)
    print(f"  {'TICKER':<8} | {'Trades':>6} | {'Win %':>6} | {'Total P&L':>11}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*11}")

    by_ticker: dict[str, list[dict]] = {}
    for r in trades:
        by_ticker.setdefault(r["ticker"], []).append(r)

    ticker_rows = []
    for tkr, tkr_trades in by_ticker.items():
        n_t = len(tkr_trades)
        n_w = sum(1 for t in tkr_trades if t["net_pnl"] > 0)
        total = sum(t["net_pnl"] for t in tkr_trades)
        wr = n_w / n_t * 100 if n_t > 0 else 0.0
        ticker_rows.append((tkr, n_t, wr, total))
    ticker_rows.sort(key=lambda x: x[3], reverse=True)

    for tkr, n_t, wr, total in ticker_rows:
        sign = "+" if total >= 0 else ""
        print(f"  {tkr:<8} | {n_t:>6} | {wr:>5.1f}% | {sign}${total:.2f}")

    print(f"\n{sep}")
    print("PER STRENGTH BUCKET")
    print(sep)
    for lo, hi, lbl in [(0.70, 0.80, "0.70-0.80"), (0.80, 0.90, "0.80-0.90"), (0.90, 1.01, "0.90-1.00")]:
        bt = [r for r in trades if lo <= r["setup_strength"] < hi]
        if not bt:
            continue
        n_b = len(bt)
        n_w = sum(1 for r in bt if r["net_pnl"] > 0)
        total = sum(r["net_pnl"] for r in bt)
        wr = n_w / n_b * 100
        sign = "+" if total >= 0 else ""
        print(f"  Strength {lbl} | n={n_b:>4} | Win%: {wr:.1f}% | Total: {sign}${total:.2f}")

    print(f"\n{sep}")
    print("VERDICT")
    print(sep)

    ci_lo_pct = ci[0] * 100
    ci_hi_pct = ci[1] * 100
    if mean_pnl > 0 and ci_lo_pct > 50:
        edge = "REAL EDGE — positive expectancy with win rate CI above 50%."
    elif mean_pnl > 0 and win_rate >= 50:
        edge = "POSSIBLE EDGE — positive mean P&L but CI overlaps 50%."
    elif mean_pnl > 0:
        edge = "MARGINAL — positive mean P&L but win rate below 50%."
    else:
        edge = "NO EDGE — negative mean P&L."

    print(f"\n  Win rate:          {win_rate:.1f}% (95% CI [{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%])")
    print(f"  Mean P&L:          ${mean_pnl:+.2f} per trade")
    print(f"  Cumulative P&L:    ${total_pnl:+.2f} on ${starting_capital:,.0f} capital")
    print(f"  Annualized return: {annualized:+.1f}%")
    print(f"  Sharpe:            {sharpe:.2f}")
    print(f"  Max drawdown:      -${max_dd:.2f} (-{max_dd_pct:.1f}%)")
    print(f"\n  Assessment: {edge}")
    print(f"\n  Contract note: All trades used Sep 18, 2026 monthly options (~100-200 DTE")
    print(f"  at trade time). These have lower gamma exposure than the spec'd 5-10 DTE")
    print(f"  weeklies. 5d P&L swings are compressed vs. short-dated contracts. The")
    print(f"  real fill on a weekly ATM would be 3-5x more volatile per dollar risked.")

    print(f"\n{'='*72}\n")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def save_csv(trades: list[dict], output_dir: Path) -> Path:
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = output_dir / f"scanner_v2_pnl_{ts_str}.csv"
    fieldnames = [
        "setup_date", "ticker", "setup_strength", "vix_at_setup",
        "contract_symbol", "expiration", "strike", "dte",
        "entry_price", "exit_price", "exit_reason", "hold_days",
        "gross_pnl", "fees_slippage", "net_pnl", "cumulative_pnl",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="v2 scanner P&L backtest — real option prices via Schwab"
    )
    parser.add_argument("--days", type=int, default=90,
                        help="Trading days to backtest (default 90)")
    parser.add_argument("--strength-min", type=float, default=0.70,
                        help="Minimum compression_breakout strength (default 0.70)")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Force re-fetch even if cache exists")
    args = parser.parse_args()

    use_cache = not args.no_cache

    try:
        result = run_v2_backtest(
            universe=V2_UNIVERSE,
            days=args.days,
            output_dir=args.output_dir,
            use_cache=use_cache,
            strength_min=args.strength_min,
        )
    except Exception as exc:
        print(f"\nERROR: Backtest failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    trade_records, setups_found, setups_skipped_vix, skipped_no_data, bt_start, bt_end = result

    if not trade_records:
        print("No trades executed.")
        return 0

    print_summary(
        trades=trade_records,
        setups_found=setups_found,
        setups_skipped_vix=setups_skipped_vix,
        setups_skipped_no_data=skipped_no_data,
        bt_start=bt_start,
        bt_end=bt_end,
        universe=V2_UNIVERSE,
        strength_min=args.strength_min,
    )

    try:
        csv_path = save_csv(trade_records, args.output_dir)
        print(f"Per-trade detail saved → {csv_path}")
    except Exception as exc:
        print(f"WARNING: CSV save failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
