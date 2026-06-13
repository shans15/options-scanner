"""Generate a crypto vol trading watchlist.

Steps
-----
1. Load the saved BTC vol model
2. Fetch latest 30 days of BTC data
3. Run inference on the most recent bar
4. If signal is EXPANSION:
     - Fetch option chains for BTC proxies (MSTR, IBIT, GBTC, BITO, COIN)
     - Fetch option chains for ETH proxies (ETHA, ETHE)
     - For each proxy, find the closest-to-ATM strikes
     - Build straddle/strangle suggestions
5. If signal is NEUTRAL: report 'no trade'
6. If signal is CONTRACTION: v2 placeholder

Output: structured JSON to output/crypto_vol/latest.json + console report

Usage
-----
    python3 -m scripts.crypto_vol_watchlist
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.coinbase_source import CoinbaseSource
from data.sources.hyperliquid_source import HyperliquidSource
from data.sources.yahoo_macro_source import YahooMacroSource
from data.sources.equity_options_source import CRYPTO_PROXIES, fetch_proxy_chains, get_proxy_spot
from domain.crypto.features import build_features
from domain.crypto.vol_model import load_model, predict_proba, VolModelArtifact
from domain.crypto.vol_signal import classify_signal, VolSignal

MODEL_PATH = _ROOT / "cache" / "models" / "btc_vol_model.pkl"
OUTPUT_DIR = _ROOT / "output" / "crypto_vol"
OUTPUT_FILE = OUTPUT_DIR / "latest.json"

REQUIRED_FEATURE_COLS = [
    "ret_1", "ret_4", "ret_24",
    "rv_24", "rv_96", "rv_24_pct_rank_7d",
    "vol_zscore_24",
    "fund_current", "fund_delta_1h", "fund_delta_24h",
    "fund_pct_rank_7d", "fund_zscore_7d", "fund_x_price_div",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "upper_shadow_pct",
    "vol_zscore_96",
    "fund_x_vol", "fund_x_ret", "vol_x_ret",
    "close_to_high_20", "close_to_low_20", "range_pct_20",
    "consec_up_bars", "range_expansion",
    "vol_ratio_4_24", "cvd_proxy_24", "vol_breakout", "up_vol_pct_24",
]


# ---------------------------------------------------------------------------
# Straddle / Strangle helpers
# ---------------------------------------------------------------------------

def _find_atm_strike(contracts, spot: float, option_type: str) -> float | None:
    """Return the strike closest to spot for the given option type."""
    candidates = [c for c in contracts if c.option_type == option_type]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.strike - spot)).strike


def _find_nearest_exp(contracts) -> str | None:
    """Return the nearest expiration date string (closest to today, DTE >= 7)."""
    if not contracts:
        return None
    exps = sorted({c.expiration for c in contracts}, key=lambda d: d.toordinal())
    # Prefer at least 7 DTE to avoid pin risk
    for exp in exps:
        if min(c.dte for c in contracts if c.expiration == exp) >= 7:
            return str(exp)
    return str(exps[0]) if exps else None


def _mid_price(contracts, strike: float, option_type: str, exp_str: str) -> float:
    """Mid-price for a specific contract."""
    for c in contracts:
        if (
            c.strike == strike
            and c.option_type == option_type
            and str(c.expiration) == exp_str
        ):
            mid = (c.bid + c.ask) / 2 if c.bid > 0 or c.ask > 0 else 0.0
            return round(mid, 4)
    return 0.0


def _build_straddle(ticker: str, contracts, spot: float, strong: bool) -> dict[str, Any]:
    """Build a straddle or strangle suggestion for a proxy ticker."""
    exp_str = _find_nearest_exp(contracts)
    if exp_str is None or spot == 0.0:
        return {"ticker": ticker, "error": "no_option_data"}

    atm = _find_atm_strike(contracts, spot, "call")
    if atm is None:
        return {"ticker": ticker, "error": "no_call_contracts"}

    if strong:
        # Straddle: same strike for call and put
        call_strike = atm
        put_strike = atm
        structure = "straddle"
    else:
        # Strangle: one strike OTM each side
        # Find the next OTM call strike (above spot)
        otm_calls = sorted(
            [c for c in contracts if c.option_type == "call" and c.strike > spot],
            key=lambda c: c.strike,
        )
        otm_puts = sorted(
            [c for c in contracts if c.option_type == "put" and c.strike < spot],
            key=lambda c: -c.strike,
        )
        call_strike = otm_calls[0].strike if otm_calls else atm
        put_strike = otm_puts[0].strike if otm_puts else atm
        structure = "strangle"

    call_mid = _mid_price(contracts, call_strike, "call", exp_str)
    put_mid = _mid_price(contracts, put_strike, "put", exp_str)
    total_cost = round(call_mid + put_mid, 4)
    break_even_pct = round(total_cost / spot * 100, 2) if spot > 0 else 0.0

    return {
        "ticker": ticker,
        "spot": round(spot, 2),
        "structure": structure,
        "expiration": exp_str,
        "call_strike": call_strike,
        "put_strike": put_strike,
        "call_mid": call_mid,
        "put_mid": put_mid,
        "total_cost": total_cost,
        "break_even_pct": break_even_pct,
    }


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def _print_trade(t: dict[str, Any]) -> None:
    if "error" in t:
        print(f"  {t['ticker']:<6}  [no data: {t['error']}]")
        return
    be_str = f"±{t['break_even_pct']:.1f}%"
    print(
        f"  {t['ticker']:<6}  (${t['spot']:>8.2f})   "
        f"ATM {t['structure']:<8}  "
        f"${t['call_strike']}c + ${t['put_strike']}p  "
        f"exp {t['expiration']}   "
        f"cost ${t['total_cost']:.2f}   "
        f"break-even {be_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"=== Crypto Vol Watchlist — {today} ===")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\nModel artifact:        {MODEL_PATH}")
    try:
        artifact: VolModelArtifact = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Run `python3 -m scripts.train_btc_vol_model` first.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"\nERROR: Failed to load model: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    trained_date = artifact.train_window_end.strftime("%Y-%m-%d")
    days_old = (datetime.now(timezone.utc) - artifact.train_window_end.tz_localize("UTC")
                if artifact.train_window_end.tzinfo is None
                else datetime.now(timezone.utc) - artifact.train_window_end).days
    print(f"Trained:               {trained_date} ({days_old} day(s) ago)")
    print(f"Train CV AUC:          {artifact.cv_mean_auc:.3f}")
    print(f"Train samples:         {artifact.n_train_samples:,}")
    print(f"Feature count:         {len(artifact.feature_names)}")

    # ------------------------------------------------------------------
    # 2. Fetch latest BTC data (30 days for feature warmup)
    # ------------------------------------------------------------------
    print("\nFetching latest BTC data (30 days) …")
    try:
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=30)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        cb = CoinbaseSource()
        hl = HyperliquidSource()
        yh = YahooMacroSource()

        ohlcv = cb.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
        funding = hl.fetch_perp_funding("BTC", start_ms, end_ms)

        merged = ohlcv.copy()
        merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
        merged["mark_price"] = merged["close"]

        for product, col_close, col_vol in [
            ("ETH-USD", "eth_close", "eth_volume"),
            ("SOL-USD", "sol_close", "sol_volume"),
        ]:
            try:
                df = cb.fetch_ohlcv(product, "15m", start_ms, end_ms)
                merged[col_close] = df["close"].reindex(merged.index, method="ffill")
                merged[col_vol] = df["volume"].reindex(merged.index, method="ffill")
            except Exception:
                pass

        for symbol, col in [("DX-Y.NYB", "dxy_close"), ("^VIX", "vix_close")]:
            try:
                df = yh.fetch_history(symbol, period="1mo")
                merged[col] = df["close"].reindex(merged.index, method="ffill")
            except Exception:
                pass

        merged = merged.dropna(subset=["open", "high", "low", "close", "volume", "funding_rate"])
        btc_spot = float(merged["close"].iloc[-1])
        current_bar = str(merged.index[-1])

        print(f"Current bar:           {current_bar}")
        print(f"BTC spot:              ${btc_spot:,.0f}")

    except Exception as exc:
        print(f"\nERROR: Data fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 3. Build features and run inference
    # ------------------------------------------------------------------
    print("\nBuilding features and running inference …")
    try:
        features_all = build_features(merged, settlement_period_bars=4)

        # We only need the last valid bar for inference
        # Align to artifact's feature names; only keep columns the model knows about
        model_feats = [f for f in artifact.feature_names if f in features_all.columns]
        missing_feats = [f for f in artifact.feature_names if f not in features_all.columns]

        if missing_feats:
            print(
                f"  WARN: {len(missing_feats)} model feature(s) not in current data "
                f"(will be filled with NaN): {missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}"
            )

        feat_df = features_all[model_feats].copy()
        # Add missing columns as NaN so predict_proba can handle them
        for col in missing_feats:
            feat_df[col] = np.nan

        # Reorder to match artifact order
        feat_df = feat_df[artifact.feature_names]

        # Drop rows with NaN in required features (warm-up rows)
        req_cols = [c for c in REQUIRED_FEATURE_COLS if c in feat_df.columns]
        valid = ~feat_df[req_cols].isna().any(axis=1)
        feat_valid = feat_df[valid]

        if len(feat_valid) == 0:
            print("ERROR: No valid feature rows after warmup. Need more data.", file=sys.stderr)
            return 1

        # Inference on most recent bar only
        last_row = feat_valid.iloc[[-1]]
        proba_series = predict_proba(artifact, last_row)
        proba = float(proba_series.iloc[0])

    except Exception as exc:
        print(f"\nERROR: Feature / inference failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 4. Classify signal
    # ------------------------------------------------------------------
    signal: VolSignal = classify_signal(proba)

    print(f"\nVOL SIGNAL: {signal.strength.upper()} {signal.direction.upper()} "
          f"(probability {proba:.2f})")
    print(f"Suggested action: {signal.suggested_action}")

    # ------------------------------------------------------------------
    # 5. Generate trade recommendations
    # ------------------------------------------------------------------
    trades: dict[str, list[dict[str, Any]]] = {"BTC": [], "ETH": []}
    output: dict[str, Any] = {
        "generated_at": today,
        "model_path": str(MODEL_PATH),
        "trained_date": trained_date,
        "cv_auc": artifact.cv_mean_auc,
        "current_bar": current_bar,
        "btc_spot": btc_spot,
        "probability": proba,
        "signal": {
            "direction": signal.direction,
            "strength": signal.strength,
            "suggested_action": signal.suggested_action,
            "notes": signal.notes,
        },
        "trades": trades,
    }

    if signal.direction == "expansion":
        strong = signal.strength == "strong"
        print("\nFetching proxy option chains …")

        for crypto in ["BTC", "ETH"]:
            proxy_label = "BTC proxies" if crypto == "BTC" else "ETH proxies"
            print(f"\n{proxy_label}:")
            chains = fetch_proxy_chains(crypto)
            for ticker, contracts in chains.items():
                spot = get_proxy_spot(ticker) if not contracts else contracts[0].spot_price
                suggestion = _build_straddle(ticker, contracts, spot, strong=strong)
                trades[crypto].append(suggestion)
                _print_trade(suggestion)

        # Risk / sizing guidance
        print("\nRisk: max loss per straddle = total premium paid.")
        print("Suggested position size:")
        print("  $20k account → 2-3% per straddle → 1-2 contracts depending on premium")

    elif signal.direction == "neutral":
        print("\nNo trade — model is not decisive. Monitor for signal change.")

    else:
        print(
            f"\nContraction signal ({signal.strength}). "
            f"Short-vol strategies are deferred to v2 (require margin). "
            "No trade generated."
        )

    # ------------------------------------------------------------------
    # 6. Save JSON output
    # ------------------------------------------------------------------
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as fh:
            json.dump(output, fh, indent=2, default=str)
        print(f"\nOutput saved → {OUTPUT_FILE}")
    except Exception as exc:
        print(f"\nWARN: Could not save JSON output: {exc}", file=sys.stderr)

    print(f"\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
