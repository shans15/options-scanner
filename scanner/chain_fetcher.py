# scanner/chain_fetcher.py
from __future__ import annotations
import time
from datetime import date, datetime
from typing import TypedDict, Optional
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default


def parse_contract(
    raw: dict,
    ticker: str,
    spot_price: float,
    option_type: str,
) -> Optional[Contract]:
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


def filter_by_delta(contracts: list) -> list:
    """Keep only contracts within the target delta range."""
    result = []
    for c in contracts:
        d = abs(c.get('delta', 0))
        if DELTA_MIN <= d <= DELTA_MAX:
            result.append(c)
    return result


def fetch_contracts(ticker: str, spot_price: float) -> list:
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
