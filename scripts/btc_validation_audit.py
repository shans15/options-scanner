# %%
"""Phase 1 robustness audit + direction prediction extensions.

Phases
------
A — Volatility model audit (4 sub-tests)
    A.1  Drop top feature (rv_24_pct_rank_7d), retrain, compare AUC.
    A.2  Out-of-time split: train on first half, test on second half.
    A.3  Calibration analysis: decile bins, Brier score.
    A.4  Simulated P&L for a vol straddle proxy.

B — Direction prediction at multiple horizons (1h, 4h, 1d, 3d)
    Walk-forward 6-fold CV + holdout at each horizon.

C — Direction conditional on vol regime
    Split dataset by vol model's predicted regime; train separate direction
    models for high-vol and low-vol subsets; compare AUC.
"""

# ---------------------------------------------------------------------------
# %% Data prep — mirrors btc_funding_validation.py
# ---------------------------------------------------------------------------
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from data.sources.coinbase_source import CoinbaseSource
from data.sources.hyperliquid_source import HyperliquidSource
from data.sources.coingecko_source import CoinGeckoSource
from data.sources.deribit_source import DeribitSource
from data.sources.yahoo_macro_source import YahooMacroSource
from domain.crypto.features import build_features
from engine.ml.walkforward import walk_forward_splits

end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=365)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

cb = CoinbaseSource()
hl = HyperliquidSource()
yh = YahooMacroSource()

print(f"Fetching BTC-USD 15m OHLCV: {start.date()} to {end.date()} …")
ohlcv = cb.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
print(f"  OHLCV rows: {len(ohlcv):,}")

print("Fetching BTC perp hourly funding from Hyperliquid …")
funding = hl.fetch_perp_funding("BTC", start_ms, end_ms)
print(f"  Funding rows: {len(funding):,}")

# Cross-asset (optional)
try:
    ohlcv_eth = cb.fetch_ohlcv("ETH-USD", "15m", start_ms, end_ms)
    print(f"  ETH rows: {len(ohlcv_eth):,}")
except Exception as e:
    print(f"  WARN: ETH-USD fetch failed ({e}), skipping")
    ohlcv_eth = None

try:
    ohlcv_sol = cb.fetch_ohlcv("SOL-USD", "15m", start_ms, end_ms)
    print(f"  SOL rows: {len(ohlcv_sol):,}")
except Exception as e:
    print(f"  WARN: SOL-USD fetch failed ({e}), skipping")
    ohlcv_sol = None

# Macro (optional)
try:
    dxy = yh.fetch_history("DX-Y.NYB", period="2y")
    print(f"  DXY rows: {len(dxy)}")
except Exception as e:
    print(f"  WARN: DXY fetch failed ({e}), skipping")
    dxy = pd.DataFrame()

try:
    vix = yh.fetch_history("^VIX", period="2y")
    print(f"  VIX rows: {len(vix)}")
except Exception as e:
    print(f"  WARN: VIX fetch failed ({e}), skipping")
    vix = pd.DataFrame()

# Merge onto 15m grid
merged = ohlcv.copy()
merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
merged["mark_price"] = merged["close"]

if ohlcv_eth is not None and not ohlcv_eth.empty:
    merged["eth_close"] = ohlcv_eth["close"].reindex(merged.index, method="ffill")
    merged["eth_volume"] = ohlcv_eth["volume"].reindex(merged.index, method="ffill")

if ohlcv_sol is not None and not ohlcv_sol.empty:
    merged["sol_close"] = ohlcv_sol["close"].reindex(merged.index, method="ffill")
    merged["sol_volume"] = ohlcv_sol["volume"].reindex(merged.index, method="ffill")

if not dxy.empty:
    merged["dxy_close"] = dxy["close"].reindex(merged.index, method="ffill")
if not vix.empty:
    merged["vix_close"] = vix["close"].reindex(merged.index, method="ffill")

required_cols = ["open", "high", "low", "close", "volume", "funding_rate"]
merged = merged.dropna(subset=required_cols)
print(f"\nMerged rows after dropna: {len(merged):,}")

# Feature matrix (full)
features_all = build_features(merged, settlement_period_bars=4)

# Required feature columns (from Phase 1 v3)
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
# Helper: build_lgbm  ——  single model factory to keep params consistent
# ---------------------------------------------------------------------------
def build_lgbm(n_estimators: int = 500) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Helper: run_holdout_eval  ——  train on pool, eval on holdout, return dict
# ---------------------------------------------------------------------------
def run_holdout_eval(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    n_estimators: int = 200,
) -> dict:
    """Train a LightGBM on X_pool / y_pool, evaluate on X_holdout / y_holdout."""
    if len(X_pool) < 50 or len(X_holdout) < 20:
        return {"auc": float("nan"), "lift": float("nan"), "f1": float("nan"), "n": len(X_holdout)}

    model = build_lgbm(n_estimators)
    model.fit(X_pool, y_pool)
    proba = model.predict_proba(X_holdout)[:, 1]
    pred = (proba > 0.5).astype(int)

    try:
        auc = roc_auc_score(y_holdout, proba)
    except Exception:
        auc = float("nan")
    f1 = f1_score(y_holdout, pred, zero_division=0)
    acc = (pred == y_holdout).mean()
    majority = max(float(y_holdout.mean()), 1.0 - float(y_holdout.mean()))
    lift = acc - majority

    return {
        "model": model,
        "proba": proba,
        "pred": pred,
        "auc": auc,
        "lift": lift,
        "f1": f1,
        "n": len(y_holdout),
    }


# ---------------------------------------------------------------------------
# Helper: run_walkforward_auc  ——  6-fold WF CV, return mean holdout AUC
# ---------------------------------------------------------------------------
def run_walkforward_auc(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 6,
) -> tuple[float, float, float]:
    """Walk-forward 6-fold CV. Returns (mean_auc, mean_lift, mean_f1)."""
    holdout_start = int(len(X) * 0.8)
    X_pool = X.iloc[:holdout_start]
    y_pool = y.iloc[:holdout_start]
    X_holdout = X.iloc[holdout_start:]
    y_holdout = y.iloc[holdout_start:]

    if len(X_pool) < 50:
        return float("nan"), float("nan"), float("nan"), X_holdout, y_holdout, None

    splits = walk_forward_splits(len(X_pool), n_splits=n_splits)
    aucs, lifts, f1s = [], [], []

    for train_idx, val_idx, test_idx in splits:
        X_tr = X_pool.iloc[train_idx]
        y_tr = y_pool.iloc[train_idx]
        X_val = X_pool.iloc[val_idx]
        y_val = y_pool.iloc[val_idx]
        X_te = X_pool.iloc[test_idx]
        y_te = y_pool.iloc[test_idx]

        model = build_lgbm(500)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba > 0.5).astype(int)

        try:
            aucs.append(roc_auc_score(y_te, proba))
        except Exception:
            pass
        f1s.append(f1_score(y_te, pred, zero_division=0))
        acc = (pred == y_te).mean()
        majority = max(float(y_te.mean()), 1.0 - float(y_te.mean()))
        lifts.append(acc - majority)

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    mean_lift = float(np.mean(lifts)) if lifts else float("nan")
    mean_f1 = float(np.mean(f1s)) if f1s else float("nan")

    # Train final model on full pool for holdout eval
    best_iter = getattr(model, "best_iteration_", None)
    n_est = int(best_iter) if best_iter and best_iter > 0 else 200
    final = build_lgbm(n_est)
    final.fit(X_pool, y_pool)
    proba_h = final.predict_proba(X_holdout)[:, 1]
    pred_h = (proba_h > 0.5).astype(int)

    try:
        holdout_auc = roc_auc_score(y_holdout, proba_h)
    except Exception:
        holdout_auc = float("nan")
    holdout_f1 = f1_score(y_holdout, pred_h, zero_division=0)
    acc_h = (pred_h == y_holdout).mean()
    majority_h = max(float(y_holdout.mean()), 1.0 - float(y_holdout.mean()))
    holdout_lift = acc_h - majority_h

    return holdout_auc, holdout_lift, holdout_f1, X_holdout, y_holdout, final


# ---------------------------------------------------------------------------
# Build vol target (reused across Phase A, B, C)
# ---------------------------------------------------------------------------
future_rv = pd.Series(
    np.log(merged["close"]).diff().rolling(4).std().shift(-4) * np.sqrt(96 * 252),
    index=merged.index,
)
rolling_vol_median = future_rv.rolling(96 * 7).median()
target_vol_raw = (future_rv > rolling_vol_median).astype(int)

# Align to features index and drop NaN required features
features_vol_raw = features_all.copy()
target_vol_raw = target_vol_raw.reindex(features_vol_raw.index)

req_present_vol = features_vol_raw[[c for c in REQUIRED_FEATURE_COLS if c in features_vol_raw.columns]]
valid_vol = ~target_vol_raw.isna() & ~req_present_vol.isna().any(axis=1)
features_vol = features_vol_raw[valid_vol]
target_vol = target_vol_raw[valid_vol]
print(f"\nVol target valid samples: {len(features_vol):,}  base rate: {target_vol.mean():.3f}")

# Vol train/holdout split
holdout_start_vol = int(len(features_vol) * 0.8)
X_pool_vol = features_vol.iloc[:holdout_start_vol]
y_pool_vol = target_vol.iloc[:holdout_start_vol]
X_holdout_vol = features_vol.iloc[holdout_start_vol:]
y_holdout_vol = target_vol.iloc[holdout_start_vol:]

# ---------------------------------------------------------------------------
# %% Reference vol model: full features (baseline — AUC from Phase 1 = 0.751)
# ---------------------------------------------------------------------------
print("\n--- [REFERENCE] Vol model: full feature set (replicate Phase 1) ---")
ref_result = run_holdout_eval(X_pool_vol, y_pool_vol, X_holdout_vol, y_holdout_vol, n_estimators=200)
auc_ref = ref_result["auc"]
lift_ref = ref_result["lift"]
f1_ref = ref_result["f1"]
final_model_vol_ref = ref_result["model"]
proba_holdout_vol_ref = ref_result["proba"]
print(f"  AUC={auc_ref:.3f}  Lift={lift_ref:+.3f}  F1={f1_ref:.3f}")

# ---------------------------------------------------------------------------
# %% Phase A — Volatility model audit
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE A — Volatility model audit")
print("=" * 60)

# ---- A.1  Drop top feature (rv_24_pct_rank_7d), retrain ----
print("\n[A.1] Drop rv_24_pct_rank_7d, retrain vol model …")
feats_a1 = features_vol.drop(columns=["rv_24_pct_rank_7d"], errors="ignore")
X_pool_a1 = feats_a1.iloc[:holdout_start_vol]
X_hold_a1 = feats_a1.iloc[holdout_start_vol:]

res_a1 = run_holdout_eval(X_pool_a1, y_pool_vol, X_hold_a1, y_holdout_vol, n_estimators=200)
auc_a1 = res_a1["auc"]
lift_a1 = res_a1["lift"]
f1_a1 = res_a1["f1"]
print(f"  AUC={auc_a1:.3f}  Lift={lift_a1:+.3f}  F1={f1_a1:.3f}")
if auc_a1 < 0.65:
    print("  VERDICT A.1: AUC < 0.65 — model is GARCH-equivalent (one-feature dependent)")
elif auc_a1 >= 0.70:
    print("  VERDICT A.1: AUC >= 0.70 — multi-signal redundancy confirmed")
else:
    print("  VERDICT A.1: AUC in 0.65-0.70 — partial dependence on rv_24_pct_rank_7d")

# ---- A.2  Out-of-time split test ----
print("\n[A.2] Out-of-time split: train on first half, test on second half …")
n_vol = len(features_vol)
oot_split = n_vol // 2
X_oot_train = features_vol.iloc[:oot_split]
y_oot_train = target_vol.iloc[:oot_split]
X_oot_test = features_vol.iloc[oot_split:]
y_oot_test = target_vol.iloc[oot_split:]

res_a2 = run_holdout_eval(X_oot_train, y_oot_train, X_oot_test, y_oot_test, n_estimators=200)
auc_a2 = res_a2["auc"]
lift_a2 = res_a2["lift"]
f1_a2 = res_a2["f1"]
print(f"  Train n={len(X_oot_train):,}  Test n={len(X_oot_test):,}")
print(f"  AUC={auc_a2:.3f}  Lift={lift_a2:+.3f}  F1={f1_a2:.3f}")
if auc_a2 >= 0.65:
    print("  VERDICT A.2: AUC >= 0.65 — pattern is regime-stable across time halves")
else:
    print("  VERDICT A.2: AUC < 0.65 — regime drift detected; pattern not stable")

# ---- A.3  Calibration analysis ----
print("\n[A.3] Calibration analysis (decile bins) …")
proba_h = proba_holdout_vol_ref
y_h = y_holdout_vol.values

brier = float(brier_score_loss(y_h, proba_h))
decile_labels = pd.qcut(proba_h, q=10, labels=False, duplicates="drop")

print(f"\n  {'Decile':<8} {'Mean pred prob':>14} {'Actual freq':>12} {'N':>6}")
print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*6}")
for d in sorted(set(decile_labels)):
    mask = decile_labels == d
    mean_pred = proba_h[mask].mean()
    actual_freq = y_h[mask].mean()
    n_bin = int(mask.sum())
    print(f"  {d:<8} {mean_pred:>14.3f} {actual_freq:>12.3f} {n_bin:>6}")

print(f"\n  Brier score: {brier:.4f}  (lower is better; 0.25 = random, 0 = perfect)")

# ---- A.4  Simulated trading P&L (vol straddle proxy) ----
print("\n[A.4] Simulated vol straddle P&L …")

# Realized vol over the 4-bar forward window (annualised), aligned to holdout.
# Annualise consistently with how vol target was built.
future_rv_aligned = future_rv.reindex(features_vol.index)
holdout_rv = future_rv_aligned.iloc[holdout_start_vol:]

# Implied vol proxy: rolling 7-day median of CURRENT rv_24 (strictly backward-looking,
# no lookahead — rv_24 is already in the feature set as a lagged feature).
# Annualise rv_24 (which is std of log-returns) to match future_rv units.
rv24_annualised = features_vol["rv_24"] * np.sqrt(96 * 252)
implied_vol_proxy = rv24_annualised.rolling(96 * 7).median()
holdout_impl = implied_vol_proxy.iloc[holdout_start_vol:]

# Align lengths and drop any rows where realized or implied is NaN
min_len = min(len(proba_h), len(holdout_rv), len(holdout_impl))
proba_strat = proba_h[:min_len]
realized = holdout_rv.values[:min_len]
implied = holdout_impl.values[:min_len]
y_strat = y_h[:min_len]

# Mask out NaN rows (edges of rolling windows)
valid_mask = ~np.isnan(realized) & ~np.isnan(implied)
proba_strat = proba_strat[valid_mask]
realized = realized[valid_mask]
implied = implied[valid_mask]
y_strat = y_strat[valid_mask]

slippage_bps = 0.5 / 10_000  # 0.5 bps in vol-point terms

# Long vol if model predicts high vol (proba > 0.65)
# Short vol if model predicts low vol (proba < 0.35)
long_mask = proba_strat > 0.65
short_mask = proba_strat < 0.35

n_valid = len(proba_strat)  # after NaN mask applied
pnl = np.zeros(n_valid)
pnl[long_mask] = (realized[long_mask] - implied[long_mask]) - slippage_bps
pnl[short_mask] = (implied[short_mask] - realized[short_mask]) - slippage_bps

n_trades = int(long_mask.sum() + short_mask.sum())
n_wins = int((pnl[long_mask | short_mask] > 0).sum())
total_pnl = float(pnl.sum())
trade_pnl = pnl[long_mask | short_mask]
win_rate = n_wins / n_trades if n_trades > 0 else float("nan")

std_pnl = float(trade_pnl.std()) if n_trades > 1 else float("nan")
bars_per_year = 96 * 252
sharpe_strat = (
    float(trade_pnl.mean()) / std_pnl * (bars_per_year ** 0.5)
    if std_pnl and std_pnl > 0 else float("nan")
)

print(f"  Trades: {n_trades}  (long vol: {long_mask.sum()}, short vol: {short_mask.sum()})")
print(f"  Win rate: {win_rate:.1%}")
print(f"  Total P&L (vol points): {total_pnl:.4f}")
print(f"  Sharpe proxy: {sharpe_strat:.2f}")

# Store for summary
auc_vol_full = auc_ref
lift_vol_full = lift_ref

# ---------------------------------------------------------------------------
# %% Phase B — Direction prediction at multiple horizons
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE B — Direction prediction: multi-horizon test")
print("=" * 60)

# Direction feature matrix: use full features, drop rows where required cols NaN
future_log_ret_base = np.log(merged["close"].shift(-4) / merged["close"])  # 1h baseline

HORIZONS = {
    "1h (4 bars)": 4,
    "4h (16 bars)": 16,
    "1d (96 bars)": 96,
    "3d (288 bars)": 288,
}

horizon_results: dict[str, dict] = {}

for horizon_name, n_bars in HORIZONS.items():
    print(f"\n--- Horizon: {horizon_name} ---")
    future_ret_h = np.log(merged["close"].shift(-n_bars) / merged["close"])
    target_h = (future_ret_h > 0).astype(int)

    # Use same full feature set; align to features_all
    target_h = target_h.reindex(features_all.index)
    req_present = features_all[[c for c in REQUIRED_FEATURE_COLS if c in features_all.columns]]
    valid_h = ~target_h.isna() & ~req_present.isna().any(axis=1)
    feat_h = features_all[valid_h]
    tgt_h = target_h[valid_h]

    print(f"  Valid samples: {len(feat_h):,}  base rate: {tgt_h.mean():.3f}")

    auc_h, lift_h, f1_h, X_hold_h, y_hold_h, model_h = run_walkforward_auc(feat_h, tgt_h)
    print(f"  AUC={auc_h:.3f}  Lift={lift_h:+.3f}  F1={f1_h:.3f}")

    # SHAP top 5
    shap_results = {}
    if model_h is not None and len(X_hold_h) > 0:
        try:
            sample_h = X_hold_h.sample(n=min(500, len(X_hold_h)), random_state=42)
            expl_h = shap.TreeExplainer(model_h)
            sv_h = expl_h.shap_values(sample_h)
            if isinstance(sv_h, list):
                sv_h = sv_h[1]
            mean_sv_h = pd.Series(
                np.abs(sv_h).mean(axis=0), index=sample_h.columns
            ).sort_values(ascending=False)
            shap_results = mean_sv_h.head(5).to_dict()
            print(f"  Top 5 SHAP features: {list(mean_sv_h.head(5).index.tolist())}")
        except Exception as e:
            print(f"  WARN: SHAP failed ({e})")

    horizon_results[horizon_name] = {
        "auc": auc_h,
        "lift": lift_h,
        "f1": f1_h,
        "n": len(feat_h),
        "shap_top5": shap_results,
    }

# ---------------------------------------------------------------------------
# %% Phase C — Direction conditional on vol regime
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE C — Direction conditional on vol regime")
print("=" * 60)

# Build direction target (4-bar horizon, same as Phase 1)
future_log_ret_4 = np.log(merged["close"].shift(-4) / merged["close"])

# Use single chronological train/test split (80/20) for both vol and direction
# Step 1: Get vol model predictions for the entire feature set using in-sample model
# (We already have proba from the reference vol model for the holdout portion.
#  For the full dataset we need to predict on all rows — we use the already-trained
#  final_model_vol_ref fitted on the training pool, and predict on the full features_vol)
print("\nStep 1: Generating vol regime predictions on full dataset …")
proba_vol_full = final_model_vol_ref.predict_proba(features_vol)[:, 1]

# Step 2: Build direction target aligned to features_vol index
target_dir_4 = (future_log_ret_4 > 0).astype(int).reindex(features_vol.index)
valid_dir = ~target_dir_4.isna()
features_dir = features_vol[valid_dir]
target_dir = target_dir_4[valid_dir]
proba_vol_dir = proba_vol_full[valid_dir.values]  # align

print(f"  Direction samples (aligned to vol): {len(features_dir):,}")

# Step 3: Chronological 80/20 split
split_c = int(len(features_dir) * 0.8)
X_train_c = features_dir.iloc[:split_c]
y_train_c = target_dir.iloc[:split_c]
X_test_c = features_dir.iloc[split_c:]
y_test_c = target_dir.iloc[split_c:]
proba_vol_test = proba_vol_dir[split_c:]
proba_vol_train = proba_vol_dir[:split_c]

print(f"  Train: {len(X_train_c):,}  Test: {len(X_test_c):,}")

# --- COMBINED (baseline direction) ---
res_combined = run_holdout_eval(X_train_c, y_train_c, X_test_c, y_test_c, n_estimators=200)
auc_combined = res_combined["auc"]
lift_combined = res_combined["lift"]
n_combined = len(X_test_c)
print(f"\n  COMBINED (all regimes):  AUC={auc_combined:.3f}  Lift={lift_combined:+.3f}  n={n_combined}")

# --- HIGH-VOL regime (proba_vol > 0.65) ---
high_vol_mask_train = proba_vol_train > 0.65
high_vol_mask_test = proba_vol_test > 0.65

X_tr_hv = X_train_c[high_vol_mask_train]
y_tr_hv = y_train_c[high_vol_mask_train]
X_te_hv = X_test_c[high_vol_mask_test]
y_te_hv = y_test_c[high_vol_mask_test]

print(f"\n  HIGH_VOL train n={len(X_tr_hv):,}  test n={len(X_te_hv):,}")
if len(X_te_hv) >= 20 and len(X_tr_hv) >= 50:
    res_hv = run_holdout_eval(X_tr_hv, y_tr_hv, X_te_hv, y_te_hv, n_estimators=200)
    auc_hv = res_hv["auc"]
    lift_hv = res_hv["lift"]
    n_hv = len(X_te_hv)
    print(f"  HIGH_VOL:  AUC={auc_hv:.3f}  Lift={lift_hv:+.3f}")
else:
    auc_hv, lift_hv, n_hv = float("nan"), float("nan"), len(X_te_hv)
    print("  HIGH_VOL: insufficient samples, skipping")

# --- LOW-VOL regime (proba_vol < 0.35) ---
low_vol_mask_train = proba_vol_train < 0.35
low_vol_mask_test = proba_vol_test < 0.35

X_tr_lv = X_train_c[low_vol_mask_train]
y_tr_lv = y_train_c[low_vol_mask_train]
X_te_lv = X_test_c[low_vol_mask_test]
y_te_lv = y_test_c[low_vol_mask_test]

print(f"\n  LOW_VOL  train n={len(X_tr_lv):,}  test n={len(X_te_lv):,}")
if len(X_te_lv) >= 20 and len(X_tr_lv) >= 50:
    res_lv = run_holdout_eval(X_tr_lv, y_tr_lv, X_te_lv, y_te_lv, n_estimators=200)
    auc_lv = res_lv["auc"]
    lift_lv = res_lv["lift"]
    n_lv = len(X_te_lv)
    print(f"  LOW_VOL:  AUC={auc_lv:.3f}  Lift={lift_lv:+.3f}")
else:
    auc_lv, lift_lv, n_lv = float("nan"), float("nan"), len(X_te_lv)
    print("  LOW_VOL: insufficient samples, skipping")


# ---------------------------------------------------------------------------
# %% Summary table
# ---------------------------------------------------------------------------
def _fmt(val: float, fmt: str = ".3f") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return format(val, fmt)


print("\n\n" + "=" * 60)
print("=== AUDIT SUMMARY ===")
print("=" * 60)

print("\nVolatility model:")
print(f"  Full features:               AUC={_fmt(auc_vol_full)}, Lift={_fmt(lift_vol_full, '+.3f')}   [reference]")
print(f"  Without rv_24_pct_rank_7d:   AUC={_fmt(auc_a1)},      Lift={_fmt(lift_a1, '+.3f')}   [A.1]")
print(f"  Out-of-time test:            AUC={_fmt(auc_a2)},      Lift={_fmt(lift_a2, '+.3f')}   [A.2]")
print(f"  Brier score:                 {_fmt(brier, '.4f')}                       [A.3]")
print(f"  Simulated Sharpe:            {_fmt(sharpe_strat, '.2f')}                          [A.4]")

print("\nDirection prediction by horizon:")
for horizon_name, res in horizon_results.items():
    print(f"  {horizon_name:<14}: AUC={_fmt(res['auc'])},  Lift={_fmt(res['lift'], '+.3f')}")

print("\nDirection conditional on vol regime:")
print(f"  HIGH_VOL:  AUC={_fmt(auc_hv)}, Lift={_fmt(lift_hv, '+.3f')} (test n={n_hv})")
print(f"  LOW_VOL:   AUC={_fmt(auc_lv)}, Lift={_fmt(lift_lv, '+.3f')} (test n={n_lv})")
print(f"  COMBINED:  AUC={_fmt(auc_combined)}, Lift={_fmt(lift_combined, '+.3f')} (test n={n_combined})")

print("\n" + "=" * 60)
