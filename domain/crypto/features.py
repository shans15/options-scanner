from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(merged: pd.DataFrame, settlement_period_bars: int = 32) -> pd.DataFrame:
    """Build ML features from a merged price+funding DataFrame.

    Parameters
    ----------
    merged : pd.DataFrame
        Required columns: ['open', 'high', 'low', 'close', 'volume',
        'funding_rate', 'mark_price'].
        Optional columns (auto-detected, skipped if absent):
            'btc_dominance'   — BTC dominance % (daily, forward-filled to 15m)
            'hv_30d'          — Deribit 30-day realized volatility (daily, ffill)
        Index: UTC datetime at 15-minute intervals, ascending.
    settlement_period_bars : int, default 32
        Number of 15-minute bars per funding settlement cycle.
        32 = 8h (Binance cadence), 4 = 1h (Hyperliquid cadence).

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with the same index as *merged*. All features are
        strictly backward-looking (no lookahead): at row T only data up to and
        including T is used.

    Feature groups
    --------------
    PRICE / VOLUME (7)
        ret_1, ret_4, ret_24           log returns over 1, 4, 24 bars
        rv_24, rv_96                   realized vol (std of log returns) over 24/96 bars
        rv_24_pct_rank_7d              7-day percentile rank of rv_24 (rolling)
        vol_zscore_24                  close volume / 24-bar rolling mean volume

    FUNDING (6)
        fund_current                   current funding rate value
        fund_delta_1h                  change in funding rate over 4 bars (1 h)
        fund_delta_24h                 change in funding rate over 96 bars (24 h)
        fund_pct_rank_7d               7-day percentile rank of funding rate
        fund_zscore_7d                 z-score of funding vs 7-day rolling window
        fund_x_price_div               fund_delta_24h x -ret_24 (divergence signal)

    CONTEXT (5)
        hour_sin, hour_cos             sin/cos encoding of hour of day (UTC)
        dow_sin, dow_cos               sin/cos encoding of day of week
        bars_since_funding_settlement  0–(settlement_period_bars-1), position within
                                       the current funding cycle (no lookahead)

    SESSION INDICATORS (4) — always present
        session_asia                   hour_utc in [0, 8)
        session_eu                     hour_utc in [8, 14)
        session_us                     hour_utc in [14, 21)
        session_overnight              hour_utc in [21, 24)

    INTRA-BAR (3)
        intrabar_range                 (high - low) / close
        body_pct                       abs(close - open) / (high - low + 1e-9)
        upper_shadow_pct               (high - max(open, close)) / (high - low + 1e-9)

    VOLUME ZSCORE (1)
        vol_zscore_96                  current volume vs 96-bar (24h) z-score

    BTC DOMINANCE (2) — present only when 'btc_dominance' column is in merged
        btc_dom_current                current dominance %
        btc_dom_delta_24h              change over 96 bars (24h, forward-filled daily → ~constant)

    REALIZED VOL from Deribit (1) — present only when 'hv_30d' column is in merged
        hv_30d                         Deribit's 30-day realized volatility (forward-filled)

    CROSS-PRODUCTS (3) — always present
        fund_x_vol                     fund_zscore_7d * rv_24_pct_rank_7d
        fund_x_ret                     fund_delta_24h * ret_24
        vol_x_ret                      rv_24 * ret_24
    """
    close = merged["close"]
    open_ = merged["open"]
    high = merged["high"]
    low = merged["low"]
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

    # bars_since_funding_settlement: position within the current funding cycle.
    # settlement_period_bars=32 → 8h Binance cadence (0–31)
    # settlement_period_bars=4  → 1h Hyperliquid cadence (0–3)
    # Use integer bar position mod settlement_period_bars — no lookahead.
    bar_number = pd.RangeIndex(len(merged))
    bars_since_settlement = bar_number % settlement_period_bars

    # ------------------------------------------------------------------
    # SESSION INDICATORS
    # ------------------------------------------------------------------
    session_asia = ((hour >= 0) & (hour < 8)).astype(int)
    session_eu = ((hour >= 8) & (hour < 14)).astype(int)
    session_us = ((hour >= 14) & (hour < 21)).astype(int)
    session_overnight = (hour >= 21).astype(int)

    # ------------------------------------------------------------------
    # INTRA-BAR
    # ------------------------------------------------------------------
    hl_range = high - low
    intrabar_range = hl_range / close
    body_pct = (close - open_).abs() / (hl_range + 1e-9)
    upper_shadow_pct = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0) / (hl_range + 1e-9)

    # ------------------------------------------------------------------
    # VOLUME ZSCORE (96-bar = 24h window)
    # ------------------------------------------------------------------
    vol_mean_96 = volume.rolling(96).mean()
    vol_std_96 = volume.rolling(96).std()
    vol_zscore_96 = (volume - vol_mean_96) / vol_std_96.replace(0, np.nan)

    # ------------------------------------------------------------------
    # CROSS-PRODUCTS
    # ------------------------------------------------------------------
    fund_x_vol = fund_zscore_7d * rv_24_pct_rank_7d
    fund_x_ret = fund_delta_24h * ret_24
    vol_x_ret = rv_24 * ret_24

    # ------------------------------------------------------------------
    # Assemble base features (always present)
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
            # Session indicators
            "session_asia": pd.Series(session_asia, index=merged.index),
            "session_eu": pd.Series(session_eu, index=merged.index),
            "session_us": pd.Series(session_us, index=merged.index),
            "session_overnight": pd.Series(session_overnight, index=merged.index),
            # Intra-bar
            "intrabar_range": intrabar_range,
            "body_pct": body_pct,
            "upper_shadow_pct": upper_shadow_pct,
            # Volume z-score (96-bar)
            "vol_zscore_96": vol_zscore_96,
            # Cross-products
            "fund_x_vol": fund_x_vol,
            "fund_x_ret": fund_x_ret,
            "vol_x_ret": vol_x_ret,
        },
        index=merged.index,
    )

    # ------------------------------------------------------------------
    # Optional: BTC dominance features
    # ------------------------------------------------------------------
    if "btc_dominance" in merged.columns:
        btc_dom = merged["btc_dominance"]
        features["btc_dom_current"] = btc_dom
        features["btc_dom_delta_24h"] = btc_dom.diff(96)

    # ------------------------------------------------------------------
    # Optional: Deribit realized volatility
    # ------------------------------------------------------------------
    if "hv_30d" in merged.columns:
        features["hv_30d"] = merged["hv_30d"]

    return features
