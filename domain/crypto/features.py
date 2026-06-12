from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from a merged price+funding DataFrame.

    Parameters
    ----------
    merged : pd.DataFrame
        Columns: ['open', 'high', 'low', 'close', 'volume', 'funding_rate',
        'mark_price']. Index: UTC datetime at 15-minute intervals, ascending.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with the same index as *merged*. All features are
        strictly backward-looking (no lookahead): at row T only data up to and
        including T is used.

    Feature groups
    --------------
    PRICE / VOLUME
        ret_1, ret_4, ret_24           log returns over 1, 4, 24 bars
        rv_24, rv_96                   realized vol (std of log returns) over 24/96 bars
        rv_24_pct_rank_7d              7-day percentile rank of rv_24 (rolling)
        vol_zscore_24                  close volume / 24-bar rolling mean volume

    FUNDING
        fund_current                   current funding rate value
        fund_delta_1h                  change in funding rate over 4 bars (1 h)
        fund_delta_24h                 change in funding rate over 96 bars (24 h)
        fund_pct_rank_7d               7-day percentile rank of funding rate
        fund_zscore_7d                 z-score of funding vs 7-day rolling window
        fund_x_price_div               fund_delta_24h x -ret_24 (divergence signal)

    CONTEXT
        hour_sin, hour_cos             sin/cos encoding of hour of day (UTC)
        dow_sin, dow_cos               sin/cos encoding of day of week
        bars_since_funding_settlement  0–31 (funding settles every 32 bars = 8h)
    """
    close = merged["close"]
    volume = merged["volume"]
    funding = merged["funding_rate"]

    log_close = np.log(close)

    # ------------------------------------------------------------------
    # PRICE / VOLUME
    # ------------------------------------------------------------------
    ret_1 = log_close.diff(1)
    ret_4 = log_close.diff(4)
    ret_24 = log_close.diff(24)

    # Realized vol: rolling std of 1-bar log returns
    log_ret_1 = log_close.diff(1)
    rv_24 = log_ret_1.rolling(24).std()
    rv_96 = log_ret_1.rolling(96).std()

    # 7-day percentile rank of rv_24: 7 days * 96 bars/day = 672 bars
    rv_24_pct_rank_7d = rv_24.rolling(672).rank(pct=True)

    # Volume z-score: current volume / 24-bar rolling mean
    vol_mean_24 = volume.rolling(24).mean()
    vol_zscore_24 = volume / vol_mean_24

    # ------------------------------------------------------------------
    # FUNDING
    # ------------------------------------------------------------------
    fund_current = funding

    # Funding changes: 4 bars = 1h, 96 bars = 24h
    fund_delta_1h = funding.diff(4)
    fund_delta_24h = funding.diff(96)

    # 7-day rolling window: 7 * 96 = 672 bars
    fund_roll_672 = funding.rolling(672)
    fund_pct_rank_7d = fund_roll_672.rank(pct=True)
    fund_mean_7d = fund_roll_672.mean()
    fund_std_7d = fund_roll_672.std()
    fund_zscore_7d = (funding - fund_mean_7d) / fund_std_7d.replace(0, np.nan)

    # Cross-product divergence: high positive funding + negative price → bearish signal
    fund_x_price_div = fund_delta_24h * (-ret_24)

    # ------------------------------------------------------------------
    # CONTEXT (cyclical encodings)
    # ------------------------------------------------------------------
    idx = merged.index
    hour = idx.hour
    dow = idx.dayofweek  # 0=Monday … 6=Sunday

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    # bars_since_funding_settlement: funding settles every 8h = 32 bars at 15m
    # We encode position within the current 32-bar cycle.
    # Use the integer bar position mod 32 so there is no lookahead.
    bar_number = pd.RangeIndex(len(merged))
    bars_since_settlement = bar_number % 32

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    features = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_4": ret_4,
            "ret_24": ret_24,
            "rv_24": rv_24,
            "rv_96": rv_96,
            "rv_24_pct_rank_7d": rv_24_pct_rank_7d,
            "vol_zscore_24": vol_zscore_24,
            "fund_current": fund_current,
            "fund_delta_1h": fund_delta_1h,
            "fund_delta_24h": fund_delta_24h,
            "fund_pct_rank_7d": fund_pct_rank_7d,
            "fund_zscore_7d": fund_zscore_7d,
            "fund_x_price_div": fund_x_price_div,
            "hour_sin": pd.Series(hour_sin, index=merged.index),
            "hour_cos": pd.Series(hour_cos, index=merged.index),
            "dow_sin": pd.Series(dow_sin, index=merged.index),
            "dow_cos": pd.Series(dow_cos, index=merged.index),
            "bars_since_funding_settlement": pd.Series(
                bars_since_settlement.values, index=merged.index
            ),
        },
        index=merged.index,
    )

    return features
