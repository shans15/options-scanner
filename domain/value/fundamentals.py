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
