"""Batch fundamentals fetcher over yahooquery.

Pulls the fields our value scanner needs into plain dicts ready for
domain.value.fundamentals.normalize_fundamentals.
"""
from __future__ import annotations
import time
import logging
from typing import Optional

from yahooquery import Ticker


log = logging.getLogger(__name__)


def _safe_get(container, key):
    """dict.get that survives None / non-dict containers."""
    if container is None or not isinstance(container, dict):
        return None
    return container.get(key)


def _first_ttm_row(income_statement_rows) -> Optional[dict]:
    """Return the TTM row if present, else the most recent 12M row."""
    if not income_statement_rows or not isinstance(income_statement_rows, list):
        return None
    for row in income_statement_rows:
        if row.get('periodType') == 'TTM':
            return row
    for row in income_statement_rows:
        if row.get('periodType') == '12M':
            return row
    return income_statement_rows[0]


def _second_12m_row(income_statement_rows) -> Optional[dict]:
    """Return the 12M row from 1 year ago (index 1 among 12M rows)."""
    if not income_statement_rows or not isinstance(income_statement_rows, list):
        return None
    twelve_months = [r for r in income_statement_rows if r.get('periodType') == '12M']
    if len(twelve_months) >= 2:
        return twelve_months[1]
    if len(twelve_months) == 1:
        return twelve_months[0]
    return None


def extract_ticker_row(
    yq: Ticker,
    ticker: str,
    volume_5d_avg: Optional[float],
    volume_20d_avg: Optional[float],
    price_1y_ago: Optional[float],
) -> dict:
    """Pull all fields for one ticker into a normalisation-ready dict."""
    ap = _safe_get(yq.asset_profile, ticker) or {}
    ks = _safe_get(yq.key_stats, ticker) or {}
    inc = _safe_get(yq.income_statement, ticker)
    sd = _safe_get(yq.summary_detail, ticker) or {}

    ttm = _first_ttm_row(inc) or {}
    prior = _second_12m_row(inc) or {}

    return {
        'ticker': ticker,
        'sector': ap.get('sector') if isinstance(ap, dict) else None,
        'price': sd.get('regularMarketPrice') if isinstance(sd, dict) else None,
        'forward_eps': ks.get('forwardEps') if isinstance(ks, dict) else None,
        'revenue_ttm': ttm.get('TotalRevenue'),
        'revenue_ttm_1y_ago': prior.get('TotalRevenue'),
        'eps_ttm': ttm.get('DilutedEPS'),
        'eps_ttm_1y_ago': prior.get('DilutedEPS'),
        'book_value_per_share': ks.get('bookValue') if isinstance(ks, dict) else None,
        'enterprise_value': ks.get('enterpriseValue') if isinstance(ks, dict) else None,
        'ebitda_ttm': ttm.get('EBITDA'),
        'shares_short': ks.get('sharesShort') if isinstance(ks, dict) else None,
        'float_shares': ks.get('floatShares') if isinstance(ks, dict) else None,
        'avg_daily_volume_30d': sd.get('averageVolume') if isinstance(sd, dict) else None,
        'volume_5d_avg': volume_5d_avg,
        'volume_20d_avg': volume_20d_avg,
        'price_1y_ago': price_1y_ago,
    }


def _compute_volume_and_price_history(yq: Ticker, tickers: list[str]) -> dict[str, tuple]:
    """Return {ticker: (volume_5d_avg, volume_20d_avg, price_1y_ago)}."""
    result: dict[str, tuple] = {}
    try:
        hist = yq.history(period='1y', interval='1d')
    except Exception as exc:
        log.warning("history fetch failed for batch: %s", exc)
        return {t: (None, None, None) for t in tickers}

    for t in tickers:
        try:
            df = hist.xs(t) if hasattr(hist, 'xs') else hist
            closes = df['close'].dropna()
            volumes = df['volume'].dropna()
            v5 = float(volumes.tail(5).mean()) if len(volumes) >= 5 else None
            v20 = float(volumes.tail(20).mean()) if len(volumes) >= 20 else None
            p1y = float(closes.iloc[0]) if len(closes) > 0 else None
            result[t] = (v5, v20, p1y)
        except Exception:
            result[t] = (None, None, None)
    return result


def fetch_fundamentals_batch(
    tickers: list[str],
    batch_size: int = 20,
    sleep_seconds: float = 1.0,
) -> dict[str, dict]:
    """Fetch fundamentals for `tickers`, one batch at a time.

    Returns {ticker: raw_dict}.  Missing tickers get a ticker-only stub
    dict so downstream normalisation can still record a skip reason.
    """
    all_rows: dict[str, dict] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            yq = Ticker(batch, asynchronous=True, progress=False)
            hist_map = _compute_volume_and_price_history(yq, batch)
            for t in batch:
                v5, v20, p1y = hist_map.get(t, (None, None, None))
                all_rows[t] = extract_ticker_row(yq, t, v5, v20, p1y)
        except Exception as exc:
            log.warning("batch %d-%d failed: %s", i, i + batch_size, exc)
            for t in batch:
                all_rows.setdefault(t, {'ticker': t})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return all_rows
