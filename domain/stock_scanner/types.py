from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from datetime import date


@dataclass(frozen=True)
class StockSetup:
    """A single detector's finding on a ticker."""
    ticker: str
    setup_type: str       # 'sma_stack' | 'compression' | 'proximity_52w' | 'volume_surge'
    direction: str        # 'bullish' | 'bearish'
    strength: float       # 0.0 - 1.0
    spot: float
    notes: dict           # detector-specific details


@dataclass(frozen=True)
class StockCandidate:
    """Aggregated view: one ticker with all its detected setups + composite score."""
    ticker: str
    spot: float
    setups: list[StockSetup]                    # list of all matching setups
    bullish_score: float                        # 0-100, weighted bullish signal strength
    bearish_score: float                        # 0-100, weighted bearish signal strength
    net_score: float                            # bullish - bearish (can be negative)
    label: str                                  # 'STRONG_BULL' | 'BULL' | 'NEUTRAL' | 'BEAR' | 'STRONG_BEAR'
    snapshot: dict                              # last price/volume/52w levels for reference
