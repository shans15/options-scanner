"""Train the BTC vol prediction model and save to disk.

Run once when you want to refresh the model artifact. Should be run weekly
or after any significant feature change.

Usage
-----
    python3 -m scripts.train_btc_vol_model

Exit codes
----------
  0  success — model saved to cache/models/btc_vol_model.pkl
  1  training failed (data fetch error, feature build error, etc.)
  2  CV AUC fell below 0.65 — model not deployed; alert emitted to stderr
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path (needed when run as `python3 -m scripts.train_btc_vol_model`)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.sources.coinbase_source import CoinbaseSource
from data.sources.hyperliquid_source import HyperliquidSource
from data.sources.yahoo_macro_source import YahooMacroSource
from domain.crypto.features import build_features
from domain.crypto.vol_model import (
    VolModelArtifact,
    train_vol_model,
    save_model,
)
from engine.ml.walkforward import walk_forward_splits
from sklearn.metrics import roc_auc_score  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = _ROOT / "cache" / "models" / "btc_vol_model.pkl"
AUC_FLOOR = 0.65          # minimum acceptable CV AUC before we refuse to deploy
LOOKBACK_DAYS = 365        # days of OHLCV history to fetch for training
N_CV_SPLITS = 6            # walk-forward folds
TRAIN_FRAC = 0.5           # fraction of pool used in first CV fold

# Required features from the Phase 1 audit (base set, always present)
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


def _fetch_data() -> pd.DataFrame:
    """Fetch 365 days of BTC OHLCV + funding + optional macro/cross-asset.

    Returns a merged DataFrame with:
        open, high, low, close, volume, funding_rate, mark_price
        (optional) eth_close, eth_volume, sol_close, dxy_close, vix_close
    """
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=LOOKBACK_DAYS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    cb = CoinbaseSource()
    hl = HyperliquidSource()
    yh = YahooMacroSource()

    print(f"  Fetching BTC-USD 15m OHLCV: {start.date()} → {end.date()} …")
    ohlcv = cb.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
    print(f"  OHLCV rows: {len(ohlcv):,}")

    print("  Fetching BTC perp funding from Hyperliquid …")
    funding = hl.fetch_perp_funding("BTC", start_ms, end_ms)
    print(f"  Funding rows: {len(funding):,}")

    merged = ohlcv.copy()
    merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
    merged["mark_price"] = merged["close"]

    # Cross-asset — optional
    for product, col_close, col_vol in [
        ("ETH-USD", "eth_close", "eth_volume"),
        ("SOL-USD", "sol_close", "sol_volume"),
    ]:
        try:
            df = cb.fetch_ohlcv(product, "15m", start_ms, end_ms)
            merged[col_close] = df["close"].reindex(merged.index, method="ffill")
            merged[col_vol] = df["volume"].reindex(merged.index, method="ffill")
            print(f"  {product} rows: {len(df):,}")
        except Exception as exc:
            print(f"  WARN: {product} fetch failed ({exc}), skipping cross-asset features")

    # Macro — optional
    for symbol, col in [("DX-Y.NYB", "dxy_close"), ("^VIX", "vix_close")]:
        try:
            df = yh.fetch_history(symbol, period="2y")
            merged[col] = df["close"].reindex(merged.index, method="ffill")
            print(f"  {symbol} rows: {len(df)}")
        except Exception as exc:
            print(f"  WARN: {symbol} fetch failed ({exc}), skipping macro feature")

    required_cols = ["open", "high", "low", "close", "volume", "funding_rate"]
    merged = merged.dropna(subset=required_cols)
    print(f"  Merged rows after dropna: {len(merged):,}")
    return merged


def _build_vol_target(merged: pd.DataFrame) -> pd.Series:
    """Build binary vol expansion target.

    Target = 1 when forward 4-bar realized vol > rolling 7-day median of realized vol.
    This mirrors the Phase 1 audit target construction exactly.
    """
    future_rv = (
        pd.Series(
            np.log(merged["close"]).diff().rolling(4).std().shift(-4) * np.sqrt(96 * 252),
            index=merged.index,
        )
    )
    rolling_median = future_rv.rolling(96 * 7).median()
    target = (future_rv > rolling_median).astype(int)
    return target


def _walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    """Walk-forward CV on the training pool. Returns (mean_auc, mean_lift)."""
    try:
        from lightgbm import early_stopping  # type: ignore
        _has_early_stopping = True
    except ImportError:
        _has_early_stopping = False

    splits = walk_forward_splits(len(X), n_splits=N_CV_SPLITS, train_frac=TRAIN_FRAC)
    aucs: list[float] = []
    lifts: list[float] = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        if len(X_tr) < 50 or len(X_te) < 20:
            continue

        if _has_early_stopping:
            import lightgbm as lgb  # type: ignore
            model = train_vol_model(X_tr, y_tr, n_estimators=500)
        else:
            model = train_vol_model(X_tr, y_tr, n_estimators=500)

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba > 0.5).astype(int)

        try:
            aucs.append(float(roc_auc_score(y_te, proba)))
        except Exception:
            pass

        acc = (pred == y_te).mean()
        majority = max(float(y_te.mean()), 1.0 - float(y_te.mean()))
        lifts.append(float(acc - majority))
        print(f"    Fold {fold}: AUC={aucs[-1]:.3f}  Lift={lifts[-1]:+.3f}  test_n={len(X_te):,}")

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    mean_lift = float(np.mean(lifts)) if lifts else float("nan")
    return mean_auc, mean_lift


def main() -> int:
    print("=" * 60)
    print("BTC Vol Model — Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Fetch data
    # ------------------------------------------------------------------
    print("\n[1/5] Fetching data …")
    try:
        merged = _fetch_data()
    except Exception as exc:
        print(f"\nERROR: Data fetch failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 2: Build features
    # ------------------------------------------------------------------
    print("\n[2/5] Building features …")
    try:
        # Hyperliquid uses 1h funding cadence (4 bars per settlement)
        features_all = build_features(merged, settlement_period_bars=4)
    except Exception as exc:
        print(f"\nERROR: Feature build failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 3: Build target and align
    # ------------------------------------------------------------------
    print("\n[3/5] Building vol expansion target …")
    try:
        target_raw = _build_vol_target(merged)

        # Determine which feature columns are present (base + optional)
        present_cols = [c for c in REQUIRED_FEATURE_COLS if c in features_all.columns]
        optional_cols = [c for c in features_all.columns if c not in REQUIRED_FEATURE_COLS]
        all_cols = present_cols + optional_cols

        req_present = features_all[present_cols]
        target_aligned = target_raw.reindex(features_all.index)
        valid = ~target_aligned.isna() & ~req_present.isna().any(axis=1)

        features = features_all[valid][all_cols].copy()
        target = target_aligned[valid].copy()

        print(f"  Valid samples: {len(features):,}   Base rate: {target.mean():.3f}")
        print(f"  Features: {len(features.columns)} columns")
        print(f"  Date range: {features.index[0]} → {features.index[-1]}")
    except Exception as exc:
        print(f"\nERROR: Target build failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    if len(features) < 200:
        print(
            f"\nERROR: Only {len(features)} valid samples — too few to train reliably.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Step 4: Walk-forward CV to validate
    # ------------------------------------------------------------------
    print("\n[4/5] Walk-forward CV …")
    try:
        # Use 80% of data as the training pool for CV
        cv_end = int(len(features) * 0.8)
        X_pool = features.iloc[:cv_end]
        y_pool = target.iloc[:cv_end]

        cv_auc, cv_lift = _walk_forward_cv(X_pool, y_pool)
        print(f"\n  CV mean AUC:  {cv_auc:.3f}   CV mean Lift: {cv_lift:+.3f}")

        if cv_auc < AUC_FLOOR:
            print(
                f"\nALERT: CV AUC {cv_auc:.3f} is below floor {AUC_FLOOR:.2f}. "
                "Model NOT saved. Re-check features or data quality.",
                file=sys.stderr,
            )
            return 2
    except Exception as exc:
        print(f"\nERROR: CV failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 5: Train final model on full dataset and save
    # ------------------------------------------------------------------
    print("\n[5/5] Training final model on full dataset …")
    try:
        final_model = train_vol_model(features, target, n_estimators=500)

        artifact = VolModelArtifact(
            model=final_model,
            feature_names=list(features.columns),
            train_window_start=features.index[0],
            train_window_end=features.index[-1],
            n_train_samples=len(features),
            cv_mean_auc=cv_auc,
            cv_mean_lift=cv_lift,
        )

        save_model(artifact, MODEL_PATH)
        print(f"  Saved → {MODEL_PATH}")
        print(f"  Training samples: {len(features):,}")
        print(f"  Features:         {len(features.columns)}")
        print(f"  CV AUC:           {cv_auc:.3f}")
        print(f"  CV Lift:          {cv_lift:+.3f}")
    except Exception as exc:
        print(f"\nERROR: Final training or save failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
