from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


@dataclass(frozen=True)
class KalshiMarket:
    """A Kalshi prediction market on a BTC price condition."""

    ticker: str           # e.g., 'KXBTC-26JUN1316-T62500'
    event_ticker: str     # e.g., 'KXBTC-26JUN1316'
    title: str            # e.g., 'BTC > $62,500 at 4:00 PM ET'
    strike: float         # the price threshold
    side: Literal["above", "below"]  # 'above' = YES wins if BTC > strike
    expiration: datetime  # UTC
    close_time: datetime  # market close (typically same as expiration)
    yes_bid: float        # 0.0 to 1.0
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_24h: int       # dollar volume past 24h
    open_interest: int
    status: Literal["open", "closed", "settled"]
    resolution: Optional[Literal["yes", "no"]] = None  # for settled markets

    @property
    def market_implied_prob(self) -> float:
        """Mid of yes_bid and yes_ask. 0.0 = certain no, 1.0 = certain yes."""
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid


@dataclass(frozen=True)
class PredictionMarketEdge:
    """An identified mispricing. Produced by engine/prediction/ensemble.py."""

    market: KalshiMarket
    fair_prob_A: float             # Strategy A: vol-model only
    fair_prob_B: float             # Strategy B: research-paper ensemble
    fair_prob_AB: float            # Strategy A+B: combined
    canonical_fair_prob: float     # use AB by default; A or B if user runs single-strategy
    edge: float                    # canonical_fair_prob - market_implied_prob
    expected_value_per_dollar: float  # P(win) * payoff - P(lose) * cost
    estimator_agreement: tuple[int, int]  # (n_agree, n_total)
    suggested_position_dollars: float
    adjustments_applied: list[str]  # human-readable list


# ---------------------------------------------------------------------------
# Ticker parsing helpers
# ---------------------------------------------------------------------------

# Ticker format: KXBTC-{date}{hour}-{prefix}{strike}
# prefix 'T' = "above" (YES wins if BTC > strike)
# prefix 'B' = "below" (YES wins if BTC < strike)
_TICKER_RE = re.compile(
    r"^KXBTC-[A-Z0-9]+-([TB])(\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def _parse_strike_and_side(ticker: str) -> tuple[float, Literal["above", "below"]]:
    """Extract strike and side from a Kalshi BTC ticker.

    Ticker examples:
      KXBTC-26JUN1316-T62500  → strike=62500.0, side='above'
      KXBTC-26JUN1316-B62500  → strike=62500.0, side='below'

    Falls back to inferring from title if ticker does not match the pattern.
    """
    m = _TICKER_RE.match(ticker)
    if m:
        prefix = m.group(1).upper()
        strike = float(m.group(2))
        side: Literal["above", "below"] = "above" if prefix == "T" else "below"
        return strike, side

    # Fallback: try to find a numeric token after the last '-'
    parts = ticker.rsplit("-", 1)
    if len(parts) == 2:
        suffix = parts[1]
        # Strip any leading letter
        numeric_part = suffix.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        if numeric_part:
            return float(numeric_part), "above"

    raise ValueError(f"Cannot parse strike/side from ticker: {ticker!r}")


def _parse_datetime(value: str | None) -> datetime:
    """Parse an ISO-8601 datetime string to a UTC-aware datetime.
    Handles strings with or without the trailing 'Z'."""
    if not value:
        raise ValueError("Empty datetime string")
    # Normalise trailing Z → +00:00 for fromisoformat compatibility
    normalised = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalised)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _cents_to_prob(value: int | float | None) -> float:
    """Convert Kalshi cent-price (0–99) or fractional (0–1) to probability 0–1."""
    if value is None:
        return 0.0
    v = float(value)
    # Kalshi orderbook prices are in cents (0-99); mid-market quotes may be floats 0-1
    if v > 1.0:
        return v / 100.0
    return v


def parse_kalshi_market(raw: dict) -> KalshiMarket:
    """Parse a raw dict from KalshiSource into a KalshiMarket.

    Handles:
    - Extracting strike from ticker (e.g., 'T62500' → 62500.0)
    - Deriving side from ticker prefix ('T' = above, 'B' = below)
    - Normalising bid/ask from cent prices (0-99) or fractions (0-1)
    - Parsing close_time from raw['close_time'] ISO string
    """
    ticker = raw["ticker"]
    strike, side = _parse_strike_and_side(ticker)

    # Prices: Kalshi REST returns yes_bid/yes_ask in cents (integer 0-99)
    # or as floats in 0-1 range — normalise both
    yes_bid = _cents_to_prob(raw.get("yes_bid") or raw.get("last_price") or 0)
    yes_ask = _cents_to_prob(raw.get("yes_ask") or 0)
    no_bid = _cents_to_prob(raw.get("no_bid") or 0)
    no_ask = _cents_to_prob(raw.get("no_ask") or 0)

    # Expiration and close time
    close_time_raw = raw.get("close_time") or raw.get("expiration_time") or ""
    expiration_raw = raw.get("expiration_time") or raw.get("close_time") or ""
    close_time = _parse_datetime(close_time_raw)
    expiration = _parse_datetime(expiration_raw)

    # Volume and open interest
    volume_24h = int(raw.get("volume", 0) or 0)
    open_interest = int(raw.get("open_interest", 0) or 0)

    # Status
    raw_status = raw.get("status", "open").lower()
    if raw_status not in ("open", "closed", "settled"):
        raw_status = "open"
    status: Literal["open", "closed", "settled"] = raw_status  # type: ignore[assignment]

    # Resolution (only present on settled markets)
    resolution_raw = raw.get("result") or raw.get("resolution")
    resolution: Optional[Literal["yes", "no"]] = None
    if resolution_raw and resolution_raw.lower() in ("yes", "no"):
        resolution = resolution_raw.lower()  # type: ignore[assignment]

    return KalshiMarket(
        ticker=ticker,
        event_ticker=raw.get("event_ticker", ""),
        title=raw.get("title", ""),
        strike=strike,
        side=side,
        expiration=expiration,
        close_time=close_time,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume_24h=volume_24h,
        open_interest=open_interest,
        status=status,
        resolution=resolution,
    )
