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
            'eth_close'       — ETH-USD close price (15m, forward-filled)
            'eth_volume'      — ETH-USD volume (15m, forward-filled)
            'sol_close'       — SOL-USD close price (15m, forward-filled)
            'sol_volume'      — SOL-USD volume (15m, forward-filled)
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

    PRICE ACTION (7) — new
        close_to_high_20               close / rolling 20-bar max(high) — distance from recent high
        close_to_low_20                close / rolling 20-bar min(low)  — distance from recent low
        range_pct_20                   (max(high[-20:]) - min(low[-20:])) / close — 20-bar range
        consec_up_bars                 consecutive prior bars where close > open (capped at 10)
        consec_down_bars               consecutive prior bars where close < open (capped at 10)
        gap_pct                        (open - prior_close) / prior_close
        range_expansion                current bar range / 20-bar avg range

    VOLUME (4) — new
        vol_ratio_4_24                 mean(volume[-4:]) / mean(volume[-24:])
        cvd_proxy_24                   sum(volume * sign(close-open)) / sum(volume) over 24 bars
        vol_breakout                   current volume / 96-bar avg, clipped to [0, 5]
        up_vol_pct_24                  fraction of last 24 bars' volume on green candles

    CROSS-ASSET (8) — new, only when eth_close/sol_close present
        eth_ret_24                     ETH 24-bar log return
        sol_ret_24                     SOL 24-bar log return
        eth_btc_ratio                  ETH price / BTC price
        eth_btc_ratio_delta_24         24h change in ETH/BTC ratio
        correl_btc_eth_96              rolling 96-bar correlation of BTC/ETH log returns
        btc_eth_div                    ret_24(BTC) - ret_24(ETH) divergence
        crypto_breadth_24              fraction of (BTC, ETH, SOL) with positive 24-bar return
        eth_vol_zscore_24              ETH volume z-score over 24 bars

    CROSS-PRODUCTS (2) — new
        fund_x_dom                     fund_zscore_7d * btc_dom_current (only when btc_dom present)
        vol_x_breadth                  rv_24_pct_rank_7d * crypto_breadth_24 (only when cross-asset present)
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

    # ------------------------------------------------------------------
    # PRICE ACTION (new)
    # ------------------------------------------------------------------
    # close / 20-bar rolling max(high): 0-to-1 ratio (1 = at recent high)
    high_20 = high.rolling(20).max()
    features["close_to_high_20"] = close / high_20

    # close / 20-bar rolling min(low): ≥1 (1 = at recent low)
    low_20 = low.rolling(20).min()
    features["close_to_low_20"] = close / low_20

    # 20-bar range as % of close
    features["range_pct_20"] = (high_20 - low_20) / close

    # Consecutive bars where close > open (capped at 10), strictly backward-looking.
    # We compute this as: for each bar T, count how many consecutive bars ending at T-1
    # had close > open. The current bar T is NOT included (no lookahead).
    is_up = (close > open_).astype(int).values
    consec_up = np.zeros(len(is_up), dtype=float)
    consec_down_arr = np.zeros(len(is_up), dtype=float)
    run_up = 0
    run_down = 0
    for i in range(len(is_up)):
        # consec at position i reflects the run of bars ending at i-1
        consec_up[i] = min(run_up, 10)
        consec_down_arr[i] = min(run_down, 10)
        if is_up[i] == 1:
            run_up += 1
            run_down = 0
        elif is_up[i] == 0:
            run_down += 1
            run_up = 0
        else:
            run_up = 0
            run_down = 0
    features["consec_up_bars"] = pd.Series(consec_up, index=merged.index)
    features["consec_down_bars"] = pd.Series(consec_down_arr, index=merged.index)

    # Gap: (open - prior close) / prior close
    features["gap_pct"] = (open_ - close.shift(1)) / close.shift(1)

    # Range expansion: current bar range vs 20-bar average range
    bar_range = high - low
    avg_range_20 = bar_range.shift(1).rolling(20).mean()  # shift(1): exclude current bar
    features["range_expansion"] = bar_range / avg_range_20.replace(0, np.nan)

    # ------------------------------------------------------------------
    # VOLUME (new)
    # ------------------------------------------------------------------
    # Short-term vs medium-term volume ratio
    vol_mean_4 = volume.rolling(4).mean()
    features["vol_ratio_4_24"] = vol_mean_4 / vol_mean_24.replace(0, np.nan)

    # CVD proxy: sum(vol * sign(close - open)) / sum(vol) over 24 bars
    signed_vol = volume * np.sign(close - open_)
    cvd_num = signed_vol.rolling(24).sum()
    cvd_den = volume.rolling(24).sum()
    features["cvd_proxy_24"] = cvd_num / cvd_den.replace(0, np.nan)

    # Vol breakout: current volume vs 96-bar mean, clipped to [0, 5]
    features["vol_breakout"] = (volume / vol_mean_96.replace(0, np.nan)).clip(lower=0, upper=5)

    # Fraction of last 24 bars' volume on green candles
    green_vol = (volume * (close > open_).astype(float)).rolling(24).sum()
    total_vol_24 = volume.rolling(24).sum()
    features["up_vol_pct_24"] = green_vol / total_vol_24.replace(0, np.nan)

    # ------------------------------------------------------------------
    # CROSS-ASSET (new) — only when eth_close and sol_close are present
    # ------------------------------------------------------------------
    _has_eth = "eth_close" in merged.columns
    _has_sol = "sol_close" in merged.columns
    _has_cross = _has_eth and _has_sol

    if _has_cross:
        eth_close = merged["eth_close"]
        sol_close = merged["sol_close"]

        log_eth = np.log(eth_close)
        log_sol = np.log(sol_close)
        log_btc = log_close  # already computed above

        eth_ret_24 = log_eth.diff(24)
        sol_ret_24 = log_sol.diff(24)
        btc_ret_24 = ret_24  # same as log_btc.diff(24)

        features["eth_ret_24"] = eth_ret_24
        features["sol_ret_24"] = sol_ret_24

        # ETH/BTC price ratio
        eth_btc_ratio = eth_close / close
        features["eth_btc_ratio"] = eth_btc_ratio
        features["eth_btc_ratio_delta_24"] = eth_btc_ratio.diff(24)

        # Rolling correlation of 1-bar log returns over 96 bars
        btc_ret_1bar = log_btc.diff(1)
        eth_ret_1bar = log_eth.diff(1)
        features["correl_btc_eth_96"] = btc_ret_1bar.rolling(96).corr(eth_ret_1bar)

        # BTC-ETH return divergence
        features["btc_eth_div"] = btc_ret_24 - eth_ret_24

        # Crypto breadth: fraction of (BTC, ETH, SOL) with positive 24-bar return
        btc_pos = (btc_ret_24 > 0).astype(float)
        eth_pos = (eth_ret_24 > 0).astype(float)
        sol_pos = (sol_ret_24 > 0).astype(float)
        features["crypto_breadth_24"] = (btc_pos + eth_pos + sol_pos) / 3.0

        # ETH volume z-score over 24 bars
        if "eth_volume" in merged.columns:
            eth_vol = merged["eth_volume"]
            eth_vol_mean_24 = eth_vol.rolling(24).mean()
            eth_vol_std_24 = eth_vol.rolling(24).std()
            features["eth_vol_zscore_24"] = (eth_vol - eth_vol_mean_24) / eth_vol_std_24.replace(0, np.nan)

        # Cross-products involving cross-asset
        features["vol_x_breadth"] = rv_24_pct_rank_7d * features["crypto_breadth_24"]

    # ------------------------------------------------------------------
    # Optional cross-product: funding × dominance
    # ------------------------------------------------------------------
    if "btc_dom_current" in features.columns:
        features["fund_x_dom"] = fund_zscore_7d * features["btc_dom_current"]

    return features
