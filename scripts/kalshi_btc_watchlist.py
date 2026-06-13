"""Live watchlist of mispriced Kalshi BTC prediction markets.

Polls Kalshi for currently open BTC markets, computes fair probabilities
via Strategy A+B (combined ensemble), filters by minimum edge + agreement
threshold + volume, ranks by expected value, and outputs to
output/kalshi/latest.json + console table.

Usage
-----
    python3 -m scripts.kalshi_btc_watchlist
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.kalshi_source import KalshiSource
from data.sources.coinbase_source import CoinbaseSource
from domain.prediction.market import parse_kalshi_market
from domain.prediction.fair_probability import detect_edge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_EDGE = 0.05              # Minimum absolute edge after fees
MIN_VOLUME_24H = 500         # Minimum 24h dollar volume
MIN_AGREEMENT = 3            # Minimum of 4 estimators that must agree
MAX_POSITION_PCT = 0.05      # Max fraction of account per trade
ACCOUNT_SIZE = 1_000         # Default account size (user configures)
SUGGESTED_POS_DOLLARS = ACCOUNT_SIZE * MAX_POSITION_PCT

OUTPUT_DIR = _ROOT / "output" / "kalshi"


# ---------------------------------------------------------------------------
# Vol model helper — loads artifact and returns P(vol expansion)
# ---------------------------------------------------------------------------

def _get_vol_proba_and_quantiles(
    ohlcv_1h: pd.DataFrame,
) -> tuple[float, dict]:
    """Load vol model artifact and predict proba on last bar.

    Falls back to 0.5 if model not found or features can't be built.
    Returns (vol_proba, historical_vol_quantiles).
    """
    hourly_returns = np.log(ohlcv_1h["close"]).diff().dropna()
    hourly_vol = hourly_returns.rolling(24).std() * np.sqrt(24)
    quantiles = {
        0.25: float(hourly_vol.quantile(0.25)),
        0.5:  float(hourly_vol.quantile(0.5)),
        0.75: float(hourly_vol.quantile(0.75)),
    }

    model_path = _ROOT / "cache" / "models" / "btc_vol_model.pkl"
    if not model_path.exists():
        print(f"  WARN: Vol model not found at {model_path}. Using vol_proba=0.5")
        return 0.5, quantiles

    try:
        from domain.crypto.vol_model import load_model, predict_proba
        from data.sources.hyperliquid_source import HyperliquidSource
        from domain.crypto.features import build_features

        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_15m = now - pd.Timedelta(days=30)
        start_ms = int(start_15m.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        cb = CoinbaseSource()
        hl = HyperliquidSource()
        ohlcv_15m = cb.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
        funding = hl.fetch_perp_funding("BTC", start_ms, end_ms)

        merged = ohlcv_15m.copy()
        merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
        merged["mark_price"] = merged["close"]
        merged = merged.dropna(subset=["open", "high", "low", "close", "volume", "funding_rate"])

        artifact = load_model(model_path)
        features_all = build_features(merged, settlement_period_bars=4)
        model_cols = [f for f in artifact.feature_names if f in features_all.columns]
        feat_df = features_all[model_cols].copy()
        for col in [c for c in artifact.feature_names if c not in model_cols]:
            feat_df[col] = np.nan
        feat_df = feat_df[artifact.feature_names]
        valid = ~feat_df.isna().any(axis=1)
        feat_valid = feat_df[valid]
        if len(feat_valid) == 0:
            return 0.5, quantiles
        last_row = feat_valid.iloc[[-1]]
        proba_series = predict_proba(artifact, last_row)
        return float(proba_series.iloc[0]), quantiles

    except Exception as exc:
        print(f"  WARN: Vol model inference failed ({exc}). Using vol_proba=0.5")
        return 0.5, quantiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    now = datetime.now(timezone.utc)
    print(f"\n{'='*65}")
    print(f"=== Kalshi BTC Watchlist — {now.strftime('%Y-%m-%d %H:%M UTC')} ===")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Fetch live Kalshi markets
    # ------------------------------------------------------------------
    print("Fetching open Kalshi BTC markets …")
    try:
        ks = KalshiSource()
        raw_markets = ks.list_btc_markets(status="open")
    except Exception as exc:
        print(f"ERROR: Kalshi fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    markets = [parse_kalshi_market(m) for m in raw_markets]
    print(f"  {len(markets)} open BTC markets found")

    # ------------------------------------------------------------------
    # Fetch BTC spot + recent price action
    # ------------------------------------------------------------------
    print("Fetching BTC spot + recent price action …")
    try:
        cb = CoinbaseSource()
        end_ts = int(now.timestamp() * 1000)
        start_1h_ms = int((now - pd.Timedelta(hours=1)).timestamp() * 1000)
        start_30d_ms = int((now - pd.Timedelta(days=30)).timestamp() * 1000)

        ohlcv_30d = cb.fetch_ohlcv("BTC-USD", "1h", start_30d_ms, end_ts)
        ohlcv_1min = cb.fetch_ohlcv("BTC-USD", "1m", start_1h_ms, end_ts)

    except Exception as exc:
        print(f"ERROR: Coinbase fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    if ohlcv_30d.empty:
        print("ERROR: No 30d OHLCV data returned", file=sys.stderr)
        return 1

    btc_spot = float(ohlcv_30d["close"].iloc[-1])
    print(f"  BTC spot: ${btc_spot:,.0f}")

    hourly_returns_30d = np.log(ohlcv_30d["close"]).diff().dropna()
    minute_returns_last_60 = (
        np.log(ohlcv_1min["close"]).diff().dropna()
        if not ohlcv_1min.empty
        else pd.Series(dtype=float)
    )
    hourly_vol_24 = hourly_returns_30d.rolling(24).std() * np.sqrt(24)

    # Historical vol quantiles (from 30-day rolling hourly vol)
    historical_vol_quantiles = {
        0.25: float(hourly_vol_24.quantile(0.25)),
        0.5:  float(hourly_vol_24.quantile(0.5)),
        0.75: float(hourly_vol_24.quantile(0.75)),
    }

    last_hour_return = float(hourly_returns_30d.iloc[-1]) if len(hourly_returns_30d) > 0 else 0.0
    hourly_sigma = float(hourly_vol_24.iloc[-1]) if len(hourly_vol_24) > 0 and not np.isnan(hourly_vol_24.iloc[-1]) else 0.02

    # ------------------------------------------------------------------
    # Vol model inference
    # ------------------------------------------------------------------
    print("Running vol model inference …")
    vol_proba, _ = _get_vol_proba_and_quantiles(ohlcv_30d)
    print(f"  P(vol expansion): {vol_proba:.2%}")

    # ------------------------------------------------------------------
    # Detect edges
    # ------------------------------------------------------------------
    print(f"\nAnalyzing {len(markets)} markets for edge > {MIN_EDGE:.0%} …")
    edges = []
    skipped_volume = 0
    skipped_expired = 0

    for m in markets:
        if m.volume_24h < MIN_VOLUME_24H:
            skipped_volume += 1
            continue

        hours_to_expiry = (m.expiration - now).total_seconds() / 3600.0
        if hours_to_expiry <= 0:
            skipped_expired += 1
            continue

        try:
            edge = detect_edge(
                market=m,
                btc_spot=btc_spot,
                btc_minute_returns_last_60=minute_returns_last_60,
                btc_hourly_returns_last_30d=hourly_returns_30d,
                vol_proba=vol_proba,
                historical_vol_quantiles=historical_vol_quantiles,
                last_hour_return=last_hour_return,
                hourly_sigma=hourly_sigma,
                suggested_position_dollars=SUGGESTED_POS_DOLLARS,
                min_edge=MIN_EDGE,
            )
        except Exception as exc:
            print(f"  WARN: detect_edge failed for {m.ticker}: {exc}")
            continue

        if edge is None:
            continue

        n_agree, n_total = edge.estimator_agreement
        if n_agree < MIN_AGREEMENT:
            continue

        edges.append(edge)

    # Sort by absolute edge × suggested position size (expected value proxy)
    edges.sort(key=lambda e: -abs(e.edge) * e.suggested_position_dollars)

    print(f"  Skipped (low volume): {skipped_volume}")
    print(f"  Skipped (expired): {skipped_expired}")
    print(f"  Markets with edge >{int(MIN_EDGE*100)}%: {len(edges)}")

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print(f"BTC spot: ${btc_spot:,.0f}  |  Vol regime P(expansion): {vol_proba:.2%}")
    print(f"Markets analyzed: {len(markets)}  |  Mispriced (edge>{int(MIN_EDGE*100)}%, agreement≥{MIN_AGREEMENT}/4): {len(edges)}")
    print(f"{'─'*65}\n")

    for e in edges[:15]:
        m = e.market
        direction = "BUY YES" if e.edge > 0 else "BUY NO (SELL YES)"
        print(f"  {m.ticker}")
        print(f"    {m.title}")
        print(f"    Market: {m.market_implied_prob:.0%}  Fair: {e.canonical_fair_prob:.0%}  "
              f"Edge: {e.edge:+.1%}  ({direction})")
        print(f"    Pos: ${e.suggested_position_dollars:.0f}  "
              f"EV=${e.expected_value_per_dollar * e.suggested_position_dollars:.2f}  "
              f"Agreement: {e.estimator_agreement[0]}/{e.estimator_agreement[1]}")
        if e.adjustments_applied:
            print(f"    Adjustments: {', '.join(e.adjustments_applied)}")
        print()

    if not edges:
        print("  No mispriced markets found matching all filters.\n")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": now.isoformat(),
        "btc_spot": btc_spot,
        "vol_regime": {
            "probability": vol_proba,
            "direction": "expansion" if vol_proba > 0.65 else ("contraction" if vol_proba < 0.35 else "neutral"),
        },
        "markets_analyzed": len(markets),
        "markets_with_edge": len(edges),
        "recommendations": [
            {
                "ticker": e.market.ticker,
                "title": e.market.title,
                "expiration": e.market.expiration.isoformat(),
                "strike": e.market.strike,
                "side": e.market.side,
                "market_implied_prob": e.market.market_implied_prob,
                "fair_prob_A": e.fair_prob_A,
                "fair_prob_B": e.fair_prob_B,
                "fair_prob_AB": e.fair_prob_AB,
                "canonical_fair_prob": e.canonical_fair_prob,
                "edge": e.edge,
                "expected_value_per_dollar": e.expected_value_per_dollar,
                "suggested_position_dollars": e.suggested_position_dollars,
                "estimator_agreement": list(e.estimator_agreement),
                "adjustments_applied": e.adjustments_applied,
                "volume_24h": e.market.volume_24h,
                "spread": e.market.spread,
            }
            for e in edges
        ],
    }
    out_path = OUTPUT_DIR / "latest.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Output saved → {out_path}")
    print(f"{'='*65}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
