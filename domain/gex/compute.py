"""Dealer Gamma Exposure (GEX) computation for index options.

SpotGamma convention (industry standard):
-  Dealers are assumed SHORT calls (customers buy calls).
-  Dealers are assumed LONG puts (customers buy puts).

Per-strike GEX formula:
    gex(K) = (call_OI(K) * call_gamma(K) - put_OI(K) * put_gamma(K)) * 100 * spot²

Sign interpretation:
    gex(K) > 0  →  call-OI dominates at this strike → dealers are net short gamma
                   here (resistance/wall when above spot).
    gex(K) < 0  →  put-OI dominates at this strike → dealers are net long gamma here
                   (support/wall when below spot is less common for puts).

Total GEX across all strikes:
    total_gex > 0  →  positive regime: vol-suppressed, mean-reverting, dealers
                       sell rallies / buy dips.
    total_gex < 0  →  negative regime: vol-amplified, trending, dealers sell dips /
                       buy rallies.

Gamma Flip Level (GFL):
    The spot price at which total_gex crosses zero.  Acts as the major intraday
    regime dividing line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

from py_vollib.black_scholes.greeks import analytical as bs_greeks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrikeGex:
    strike: float
    gex_dollars: float  # signed; positive = call-dominated (dealers short gamma)


@dataclass(frozen=True)
class GexZone:
    strike: float
    gex_dollars: float
    side: Literal['above', 'below']
    distance_pct: float  # signed: positive if above spot, negative if below


@dataclass(frozen=True)
class GexContext:
    ticker: str
    spot: float
    total_gex_dollars: float
    gamma_flip_level: float | None
    supply_zones: list[GexZone]   # strikes above spot, sorted by abs(gex) desc
    demand_zones: list[GexZone]   # strikes below spot, sorted by abs(gex) desc
    chain_size: int               # number of contracts considered


# ---------------------------------------------------------------------------
# Internal gamma helper
# ---------------------------------------------------------------------------

def _safe_gamma(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Compute Black-Scholes gamma; return 0.0 on any invalid input or error."""
    if t <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    try:
        g = float(bs_greeks.gamma(flag, S, K, t, r, sigma))
        return g if math.isfinite(g) else 0.0
    except Exception:
        return 0.0


def _t_years(expiration: date, today: date) -> float:
    """Calendar days from today to expiration, expressed as fraction of a year."""
    days = (expiration - today).days
    return max(days / 365.0, 0.0)


# ---------------------------------------------------------------------------
# Core compute functions
# ---------------------------------------------------------------------------

def gex_per_strike(
    chain: list,               # list[RawContract]
    spot: float,
    risk_free_rate: float = 0.053,
    today=None,                # datetime.date; defaults to date.today()
) -> dict[float, float]:
    """Sum signed dealer-side GEX across all expirations per strike.

    Returns {strike: gex_dollars} where gex_dollars follows the SpotGamma
    convention:
        gex(K) = (call_OI * call_gamma - put_OI * put_gamma) * 100 * spot²
    """
    today = today or date.today()
    per_strike: dict[float, float] = {}

    for contract in chain:
        iv = contract.implied_volatility
        if iv is None or iv <= 0.0:
            continue
        oi = contract.open_interest
        if oi is None or oi <= 0:
            continue

        expiration = contract.expiration
        if isinstance(expiration, str):
            from datetime import date as _date
            expiration = _date.fromisoformat(expiration)

        t = _t_years(expiration, today)
        if t <= 0.0:
            continue

        flag = 'c' if contract.option_type == 'call' else 'p'
        gamma = _safe_gamma(flag, spot, contract.strike, t, risk_free_rate, iv)
        if gamma == 0.0:
            continue

        # SpotGamma formula: call contribution positive, put contribution negative
        if contract.option_type == 'call':
            contribution = oi * gamma * 100.0 * (spot ** 2)
        else:
            contribution = -(oi * gamma * 100.0 * (spot ** 2))

        k = contract.strike
        per_strike[k] = per_strike.get(k, 0.0) + contribution

    return per_strike


def total_gex(per_strike: dict[float, float]) -> float:
    """Sum of all per-strike GEX values."""
    return sum(per_strike.values())


def gamma_flip_level(
    chain: list,
    spot: float,
    risk_free_rate: float = 0.053,
    today=None,
    search_range_pct: float = 0.10,   # search ±10% around spot
    search_step_pct: float = 0.002,   # step 0.2%
) -> float | None:
    """Find the spot price at which total_gex crosses zero.

    Method:
        1. Build a grid of candidate spot prices from spot*(1-range) to
           spot*(1+range), stepping by search_step_pct.
        2. At each candidate spot, recompute gamma using each contract's IV
           and that candidate price (gamma depends on S via Black-Scholes).
        3. Find adjacent candidate spots where total GEX changes sign.
        4. Linearly interpolate between the bracketing points for precision.

    Returns None if no crossing is found within the search range.
    """
    today = today or date.today()

    lo = spot * (1.0 - search_range_pct)
    hi = spot * (1.0 + search_range_pct)
    step = spot * search_step_pct

    # Pre-filter chain to only valid contracts (saves recomputing failures)
    valid_contracts = []
    for c in chain:
        iv = c.implied_volatility
        if iv is None or iv <= 0.0:
            continue
        oi = c.open_interest
        if oi is None or oi <= 0:
            continue
        expiration = c.expiration
        if isinstance(expiration, str):
            from datetime import date as _date
            expiration = _date.fromisoformat(expiration)
        t = _t_years(expiration, today)
        if t <= 0.0:
            continue
        valid_contracts.append((c, expiration, t))

    if not valid_contracts:
        return None

    def _total_gex_at(s: float) -> float:
        acc = 0.0
        for contract, expiration, t in valid_contracts:
            flag = 'c' if contract.option_type == 'call' else 'p'
            gamma = _safe_gamma(flag, s, contract.strike, t, risk_free_rate,
                                contract.implied_volatility)
            if gamma == 0.0:
                continue
            oi = contract.open_interest
            if contract.option_type == 'call':
                acc += oi * gamma * 100.0 * (s ** 2)
            else:
                acc -= oi * gamma * 100.0 * (s ** 2)
        return acc

    # Build (candidate_spot, total_gex) list
    candidates: list[tuple[float, float]] = []
    s = lo
    while s <= hi + 1e-9:
        g = _total_gex_at(s)
        candidates.append((s, g))
        s += step

    if len(candidates) < 2:
        return None

    # Find sign change(s); return the first crossing via linear interpolation
    for i in range(len(candidates) - 1):
        s0, g0 = candidates[i]
        s1, g1 = candidates[i + 1]
        if g0 == 0.0:
            return s0
        if g1 == 0.0:
            return s1
        if (g0 > 0.0) != (g1 > 0.0):
            # Linear interpolation: find x where line from (s0,g0)→(s1,g1) = 0
            frac = g0 / (g0 - g1)
            return s0 + frac * (s1 - s0)

    return None


def top_zones(
    per_strike: dict[float, float],
    spot: float,
    n: int = 5,
    max_distance_pct: float = 0.05,   # only zones within 5% of spot
) -> tuple[list[GexZone], list[GexZone]]:
    """Return (supply_zones, demand_zones).

    supply_zones = strikes above spot, sorted by abs(gex_dollars) descending.
    demand_zones = strikes below spot, sorted by abs(gex_dollars) descending.
    Only includes strikes within abs(distance_pct) <= max_distance_pct.
    """
    supply: list[GexZone] = []
    demand: list[GexZone] = []

    for strike, gex_val in per_strike.items():
        dist_pct = (strike - spot) / spot
        if abs(dist_pct) > max_distance_pct:
            continue
        if strike > spot:
            supply.append(GexZone(
                strike=strike,
                gex_dollars=gex_val,
                side='above',
                distance_pct=dist_pct,
            ))
        elif strike < spot:
            demand.append(GexZone(
                strike=strike,
                gex_dollars=gex_val,
                side='below',
                distance_pct=dist_pct,
            ))
        # strikes exactly at spot are skipped (neither supply nor demand)

    supply.sort(key=lambda z: abs(z.gex_dollars), reverse=True)
    demand.sort(key=lambda z: abs(z.gex_dollars), reverse=True)

    return supply[:n], demand[:n]


def build_context(
    ticker: str,
    chain: list,
    spot: float,
    risk_free_rate: float = 0.053,
    today=None,
    max_dte: int | None = None,
    n_zones: int = 5,
    zone_max_distance_pct: float = 0.05,
) -> GexContext:
    """Build a full GexContext for a ticker.

    Filters chain by max_dte if given.  Skips contracts with IV <= 0 or
    invalid greeks.  Returns a GexContext with total GEX, gamma flip level,
    and top supply/demand zones.
    """
    today = today or date.today()

    # Filter by max_dte
    if max_dte is not None:
        filtered_chain = [c for c in chain if c.dte <= max_dte]
    else:
        filtered_chain = list(chain)

    # Compute per-strike GEX (already skips IV<=0 / invalid)
    per_strike = gex_per_strike(filtered_chain, spot, risk_free_rate, today)

    tgex = total_gex(per_strike)

    gfl = gamma_flip_level(filtered_chain, spot, risk_free_rate, today)

    supply_zones, demand_zones = top_zones(
        per_strike, spot, n=n_zones, max_distance_pct=zone_max_distance_pct
    )

    return GexContext(
        ticker=ticker,
        spot=spot,
        total_gex_dollars=tgex,
        gamma_flip_level=gfl,
        supply_zones=supply_zones,
        demand_zones=demand_zones,
        chain_size=len(filtered_chain),
    )
