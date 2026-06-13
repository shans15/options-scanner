# %%
"""BTC funding-rate edge validation (Phase 1 v3 — Path 1+2+3).

OHLCV:      Coinbase BTC-USD spot 15m (US-accessible, full year of history)
Funding:    Hyperliquid BTC perp hourly funding (US-accessible DEX)
Cross-asset:ETH-USD, SOL-USD OHLCV (graceful degradation on rate-limit)
Macro:      Yahoo Finance DXY + VIX daily (Path 3)

Target 1:   sign(future_log_return over next 4 bars) — direction prediction
Target 2:   realized-vol regime (above/below 7-day rolling median) — Path 2

Changes vs Phase 1 v2
---------------------
Path 3 (macro):
  - New YahooMacroSource for DXY ('DX-Y.NYB') and VIX ('^VIX') history
  - 6 new features: dxy_ret_24, dxy_vs_btc_24, vix_current, vix_zscore_30d,
    correl_btc_vix_96, correl_btc_dxy_96

Path 1 (prune):
  - Removed 14 dead-weight features per SHAP audit:
    session_asia/eu/us/overnight, btc_dom_current, btc_dom_delta_24h,
    hv_30d, bars_since_funding_settlement, consec_down_bars, gap_pct,
    body_pct, intrabar_range, fund_x_dom, crypto_breadth_24

Path 2 (vol target):
  - Second validation predicting realized-vol regime instead of direction
  - Same walk-forward + holdout + SHAP pipeline; results shown side-by-side
"""

# %% Fetch all data
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from data.sources.coinbase_source import CoinbaseSource
from data.sources.hyperliquid_source import HyperliquidSource
from data.sources.coingecko_source import CoinGeckoSource
from data.sources.deribit_source import DeribitSource
from data.sources.yahoo_macro_source import YahooMacroSource

end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=365)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

cb = CoinbaseSource()
hl = HyperliquidSource()
cg = CoinGeckoSource()
dr = DeribitSource()
yh = YahooMacroSource()

print(f"Fetching BTC-USD 15m OHLCV from Coinbase: {start.date()} to {end.date()} …")
ohlcv = cb.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
print(f"  OHLCV rows: {len(ohlcv):,}")

print(f"Fetching BTC perp hourly funding from Hyperliquid …")
funding = hl.fetch_perp_funding("BTC", start_ms, end_ms)
print(f"  Funding rows: {len(funding):,}")

# Cross-asset: ETH-USD and SOL-USD OHLCV (graceful degradation on rate-limit)
try:
    print(f"Fetching ETH-USD 15m OHLCV …")
    ohlcv_eth = cb.fetch_ohlcv("ETH-USD", "15m", start_ms, end_ms)
    print(f"  ETH rows: {len(ohlcv_eth):,}")
except Exception as e:
    print(f"  WARN: ETH-USD fetch failed ({e}), skipping cross-asset ETH features")
    ohlcv_eth = None

try:
    print(f"Fetching SOL-USD 15m OHLCV …")
    ohlcv_sol = cb.fetch_ohlcv("SOL-USD", "15m", start_ms, end_ms)
    print(f"  SOL rows: {len(ohlcv_sol):,}")
except Exception as e:
    print(f"  WARN: SOL-USD fetch failed ({e}), skipping cross-asset SOL features")
    ohlcv_sol = None

# Macro indicators (Path 3)
print("Fetching DXY history from Yahoo …")
try:
    dxy = yh.fetch_history('DX-Y.NYB', period='2y')
    print(f"  DXY rows: {len(dxy)}")
except Exception as e:
    print(f"  WARN: DXY fetch failed ({e}), skipping")
    dxy = pd.DataFrame()

print("Fetching VIX history from Yahoo …")
try:
    vix = yh.fetch_history('^VIX', period='2y')
    print(f"  VIX rows: {len(vix)}")
except Exception as e:
    print(f"  WARN: VIX fetch failed ({e}), skipping")
    vix = pd.DataFrame()

# %% Merge all into single 15m grid
merged = ohlcv.copy()
# Forward-fill funding rate into the 15m price grid.
# Hyperliquid's funding endpoint doesn't return mark_price (all NaN), so we
# fill it with the close as a proxy and drop only rows with missing
# OHLCV/funding_rate (the features we actually use).
merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
merged["mark_price"] = merged["close"]  # proxy; not used in features

# Optional: ETH and SOL cross-asset columns
if ohlcv_eth is not None and not ohlcv_eth.empty:
    merged["eth_close"] = ohlcv_eth["close"].reindex(merged.index, method="ffill")
    merged["eth_volume"] = ohlcv_eth["volume"].reindex(merged.index, method="ffill")

if ohlcv_sol is not None and not ohlcv_sol.empty:
    merged["sol_close"] = ohlcv_sol["close"].reindex(merged.index, method="ffill")
    merged["sol_volume"] = ohlcv_sol["volume"].reindex(merged.index, method="ffill")

# Macro indicators (Path 3): merge into 15m grid via forward-fill of daily data
if not dxy.empty:
    merged['dxy_close'] = dxy['close'].reindex(merged.index, method='ffill')
if not vix.empty:
    merged['vix_close'] = vix['close'].reindex(merged.index, method='ffill')

required_cols = ["open", "high", "low", "close", "volume", "funding_rate"]
merged = merged.dropna(subset=required_cols)
print(f"\nMerged rows after dropna: {len(merged):,}")

from domain.crypto.features import build_features

features = build_features(merged, settlement_period_bars=4)  # Hyperliquid: 1h = 4 bars at 15m

# Step 1: NaN audit
print(f"Feature columns ({len(features.columns)}): {list(features.columns)}")
print("\nNaN audit per feature column:")
nan_counts = features.isna().sum().sort_values(ascending=False)
for col, n in nan_counts.items():
    pct = n / len(features) * 100
    if n > 0:
        print(f"  {col:<32} {n:>6,} NaN ({pct:.1f}%)")
print(f"\n  Rows where ANY feature is NaN: {features.isna().any(axis=1).sum():,} ({features.isna().any(axis=1).mean()*100:.1f}%)")
print(f"  Rows where ALL features are non-NaN: {(~features.isna().any(axis=1)).sum():,}")

# %% Build target — symmetric ~50/50 split (PATH 1: direction)
# Previous version used > +10 bps which created a 35/65 imbalance, causing the
# model to learn "always predict 0" and achieve 65% accuracy with zero lift.
future_log_return = np.log(merged["close"].shift(-4) / merged["close"])
target = (future_log_return > 0).astype(int)  # sign of next 4-bar return — true direction

# Step 2: REQUIRED feature columns (pruned per Path 1 SHAP audit)
# Removed: session_*, bars_since_funding_settlement, intrabar_range, body_pct,
#          consec_down_bars, gap_pct (all dropped in features.py)
REQUIRED_FEATURE_COLS = [
    'ret_1', 'ret_4', 'ret_24',
    'rv_24', 'rv_96', 'rv_24_pct_rank_7d',
    'vol_zscore_24',
    'fund_current', 'fund_delta_1h', 'fund_delta_24h',
    'fund_pct_rank_7d', 'fund_zscore_7d', 'fund_x_price_div',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'upper_shadow_pct',
    'vol_zscore_96',
    'fund_x_vol', 'fund_x_ret', 'vol_x_ret',
    'close_to_high_20', 'close_to_low_20', 'range_pct_20',
    'consec_up_bars', 'range_expansion',
    'vol_ratio_4_24', 'cvd_proxy_24', 'vol_breakout', 'up_vol_pct_24',
]

# Optional: only present when data source was available
optional_cols = [c for c in features.columns if c not in REQUIRED_FEATURE_COLS]
print(f"\n  Required features: {len(REQUIRED_FEATURE_COLS)}")
print(f"  Optional features: {len(optional_cols)}  ({optional_cols})")

# Drop rows where target is undefined OR any REQUIRED feature is NaN
# Optional features keep NaN; LightGBM handles them natively
required_present = features[[c for c in REQUIRED_FEATURE_COLS if c in features.columns]]
valid = ~target.isna() & ~required_present.isna().any(axis=1)
features = features[valid]
target = target[valid]
print(f"\nValid samples: {len(features):,}  (base rate: {target.mean():.3f})")

# Step 3: Confirm LightGBM NaN handling
import lightgbm as lgb
print(f"\nLightGBM version: {lgb.__version__}  (NaN handled natively via use_missing=True)")
print(f"  Optional-feature NaN in training set: {features[optional_cols].isna().sum().sum():,} total cells" if optional_cols else "  No optional features present.")

# %% Hold out final 20% as untouched test set
holdout_start = int(len(features) * 0.8)
X_train_pool = features.iloc[:holdout_start]
y_train_pool = target.iloc[:holdout_start]
X_holdout = features.iloc[holdout_start:]
y_holdout = target.iloc[holdout_start:]
print(f"\nTrain pool: {len(X_train_pool):,}  Holdout: {len(X_holdout):,}")

# %% Walk-forward CV on the training pool
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from engine.ml.walkforward import walk_forward_splits

splits = walk_forward_splits(len(X_train_pool), n_splits=6)
fold_results = []

print("\n--- Walk-forward CV (Direction) ---")
for i, (train_idx, val_idx, test_idx) in enumerate(splits):
    X_tr = X_train_pool.iloc[train_idx]
    y_tr = y_train_pool.iloc[train_idx]
    X_val = X_train_pool.iloc[val_idx]
    y_val = y_train_pool.iloc[val_idx]
    X_te = X_train_pool.iloc[test_idx]
    y_te = y_train_pool.iloc[test_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",   # forces model to learn minority class
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba > 0.5).astype(int)

    acc = (pred == y_te).mean()
    auc = roc_auc_score(y_te, proba)
    f1 = f1_score(y_te, pred, zero_division=0)
    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred, zero_division=0)
    base_rate = float(y_te.mean())
    majority_baseline = max(base_rate, 1 - base_rate)
    lift = acc - majority_baseline

    fold_results.append({
        "fold": i, "acc": acc, "auc": auc, "f1": f1,
        "precision": prec, "recall": rec, "lift": lift,
        "n_test": len(y_te), "base_rate": base_rate,
    })
    print(
        f"Fold {i}: acc={acc:.3f}  auc={auc:.3f}  f1={f1:.3f}  "
        f"prec={prec:.3f}  rec={rec:.3f}  lift={lift:+.3f}  "
        f"base_rate={base_rate:.3f}  n={len(y_te)}"
    )

mean_cv_acc = sum(r["acc"] for r in fold_results) / len(fold_results)
mean_cv_auc = sum(r["auc"] for r in fold_results) / len(fold_results)
mean_cv_lift = sum(r["lift"] for r in fold_results) / len(fold_results)
print(f"\nMean CV accuracy: {mean_cv_acc:.3f}  AUC: {mean_cv_auc:.3f}  Lift: {mean_cv_lift:+.3f}")

# %% Final fit on all training pool and evaluate on untouched holdout
best_iter = getattr(model, "best_iteration_", None)
n_estimators_final = int(best_iter) if best_iter and best_iter > 0 else 200

final_model = lgb.LGBMClassifier(
    n_estimators=n_estimators_final,
    learning_rate=0.03,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    class_weight="balanced",
    verbose=-1,
)
final_model.fit(X_train_pool, y_train_pool)
proba_holdout = final_model.predict_proba(X_holdout)[:, 1]
pred_holdout = (proba_holdout > 0.5).astype(int)

oos_acc = (pred_holdout == y_holdout).mean()
auc_holdout = roc_auc_score(y_holdout, proba_holdout)
f1_holdout = f1_score(y_holdout, pred_holdout, zero_division=0)
prec_holdout = precision_score(y_holdout, pred_holdout, zero_division=0)
rec_holdout = recall_score(y_holdout, pred_holdout, zero_division=0)
cm = confusion_matrix(y_holdout, pred_holdout)

majority_baseline = max(float(y_holdout.mean()), 1.0 - float(y_holdout.mean()))
lift = oos_acc - majority_baseline

print(f"\n[PATH 1] Direction prediction — Final OOS metrics (untouched holdout):")
print(f"  Accuracy:          {oos_acc:.3f}")
print(f"  AUC:               {auc_holdout:.3f}")
print(f"  F1:                {f1_holdout:.3f}")
print(f"  Precision:         {prec_holdout:.3f}")
print(f"  Recall:            {rec_holdout:.3f}")
print(f"  Majority baseline: {majority_baseline:.3f}")
print(f"  Lift over majority:{lift:+.3f}")
print(f"  Confusion matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

# %% Feature importance (direction)
imp_dir = pd.Series(
    final_model.feature_importances_, index=features.columns
).sort_values(ascending=False)
print("\n[PATH 1] Top 10 features by gain (direction):")
print(imp_dir.head(10))

# %% SHAP feature importance (direction)
import shap

shap_sample_dir = X_holdout.sample(n=min(1000, len(X_holdout)), random_state=42)
explainer_dir = shap.TreeExplainer(final_model)
shap_values_dir = explainer_dir.shap_values(shap_sample_dir)

if isinstance(shap_values_dir, list):
    shap_arr_dir = shap_values_dir[1]
else:
    shap_arr_dir = shap_values_dir

mean_abs_shap_dir = pd.Series(
    np.abs(shap_arr_dir).mean(axis=0),
    index=shap_sample_dir.columns,
).sort_values(ascending=False)

print("\n--- [PATH 1] SHAP feature importance (direction — mean |SHAP| on holdout) ---")
print(mean_abs_shap_dir.to_string())

cut_threshold_dir = mean_abs_shap_dir.quantile(0.25)
recommended_drops_dir = mean_abs_shap_dir[mean_abs_shap_dir <= cut_threshold_dir].index.tolist()
print(f"\nRecommended drops (bottom quartile, mean|SHAP| <= {cut_threshold_dir:.5f}):")
for col in recommended_drops_dir:
    print(f"  - {col}: {mean_abs_shap_dir[col]:.5f}")

# %% Simulated trading P&L (educational, not production)
fee_bps = 7.5
threshold = 0.55
positions = (proba_holdout > threshold).astype(int)

realized_4bar = np.log(merged["close"].shift(-4) / merged["close"])
holdout_realized = realized_4bar[valid].iloc[holdout_start : holdout_start + len(positions)]

trade_returns = (
    holdout_realized.values * positions
    - (fee_bps / 10_000) * positions
)
n_trades = int(positions.sum())
total_return = float(trade_returns.sum())
std = float(trade_returns.std())
bars_per_year = 24 * 4 * 252
sharpe_proxy = (
    float(trade_returns.mean()) / std * (bars_per_year ** 0.5) if std > 0 else 0.0
)
print(
    f"\nSimulated trading [direction] (threshold={threshold}): {n_trades} trades, "
    f"total log-return {total_return:.4f}, "
    f"Sharpe proxy {sharpe_proxy:.2f}"
)

# %% Verdict — direction
print("\n" + "=" * 60)
print("[PATH 1] Direction prediction verdict:")
if auc_holdout < 0.52:
    verdict_dir = "FAIL: No edge over random. Features need work."
elif lift < 0.02:
    verdict_dir = f"FAIL: No lift over majority-class baseline (lift={lift:+.2%}, AUC={auc_holdout:.3f})."
elif auc_holdout >= 0.65:
    verdict_dir = f"WARNING: AUC={auc_holdout:.3f} >= 0.65 — audit for lookahead bias."
elif lift >= 0.05 and auc_holdout >= 0.55:
    verdict_dir = f"PASS: Real edge detected (lift={lift:+.2%}, AUC={auc_holdout:.3f}). Build Phase 2 infrastructure."
else:
    verdict_dir = f"MARGINAL: lift={lift:+.2%}, AUC={auc_holdout:.3f}. More features needed."
print(verdict_dir)
print("=" * 60)

# ============================================================
# PATH 2: Volatility-regime prediction
# ============================================================
print("\n" + "=" * 60)
print("PATH 2: Volatility-regime prediction")
print("=" * 60)

# Realized vol over next 4 bars (15m → 1hr forward window)
# Annualised: std of 1-bar log returns over 4 bars * sqrt(96 * 252)
future_realized_vol = pd.Series(
    np.log(merged["close"]).diff().rolling(4).std().shift(-4) * np.sqrt(96 * 252),
    index=merged.index,
)

# Symmetric vol target: above or below 7-day rolling median
rolling_vol_median = future_realized_vol.rolling(96 * 7).median()
target_vol = (future_realized_vol > rolling_vol_median).astype(int)

# Build feature matrix for vol target (same merged, same settlement_period_bars)
features_vol = build_features(merged, settlement_period_bars=4)

# Align target_vol to features_vol index
target_vol = target_vol.reindex(features_vol.index)

# REQUIRED columns for vol target: same pruned list (no dropped features)
REQUIRED_FEATURE_COLS_VOL = REQUIRED_FEATURE_COLS  # identical pruned list

required_present_vol = features_vol[[c for c in REQUIRED_FEATURE_COLS_VOL if c in features_vol.columns]]
valid_vol = ~target_vol.isna() & ~required_present_vol.isna().any(axis=1)
features_vol = features_vol[valid_vol]
target_vol = target_vol[valid_vol]
print(f"Volatility-target valid samples: {len(features_vol):,}  base rate: {target_vol.mean():.3f}")

# Walk-forward CV — vol target
holdout_start_vol = int(len(features_vol) * 0.8)
X_train_pool_vol = features_vol.iloc[:holdout_start_vol]
y_train_pool_vol = target_vol.iloc[:holdout_start_vol]
X_holdout_vol = features_vol.iloc[holdout_start_vol:]
y_holdout_vol = target_vol.iloc[holdout_start_vol:]
print(f"Train pool: {len(X_train_pool_vol):,}  Holdout: {len(X_holdout_vol):,}")

splits_vol = walk_forward_splits(len(X_train_pool_vol), n_splits=6)
fold_results_vol = []

print("\n--- Walk-forward CV (Volatility) ---")
for i, (train_idx, val_idx, test_idx) in enumerate(splits_vol):
    X_tr_v = X_train_pool_vol.iloc[train_idx]
    y_tr_v = y_train_pool_vol.iloc[train_idx]
    X_val_v = X_train_pool_vol.iloc[val_idx]
    y_val_v = y_train_pool_vol.iloc[val_idx]
    X_te_v = X_train_pool_vol.iloc[test_idx]
    y_te_v = y_train_pool_vol.iloc[test_idx]

    model_vol = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        verbose=-1,
    )
    model_vol.fit(
        X_tr_v, y_tr_v,
        eval_set=[(X_val_v, y_val_v)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    proba_v = model_vol.predict_proba(X_te_v)[:, 1]
    pred_v = (proba_v > 0.5).astype(int)

    acc_v = (pred_v == y_te_v).mean()
    auc_v = roc_auc_score(y_te_v, proba_v)
    f1_v = f1_score(y_te_v, pred_v, zero_division=0)
    prec_v = precision_score(y_te_v, pred_v, zero_division=0)
    rec_v = recall_score(y_te_v, pred_v, zero_division=0)
    base_rate_v = float(y_te_v.mean())
    majority_baseline_v = max(base_rate_v, 1 - base_rate_v)
    lift_v = acc_v - majority_baseline_v

    fold_results_vol.append({
        "fold": i, "acc": acc_v, "auc": auc_v, "f1": f1_v,
        "precision": prec_v, "recall": rec_v, "lift": lift_v,
        "n_test": len(y_te_v), "base_rate": base_rate_v,
    })
    print(
        f"Fold {i}: acc={acc_v:.3f}  auc={auc_v:.3f}  f1={f1_v:.3f}  "
        f"prec={prec_v:.3f}  rec={rec_v:.3f}  lift={lift_v:+.3f}  "
        f"base_rate={base_rate_v:.3f}  n={len(y_te_v)}"
    )

mean_cv_auc_vol = sum(r["auc"] for r in fold_results_vol) / len(fold_results_vol)
mean_cv_lift_vol = sum(r["lift"] for r in fold_results_vol) / len(fold_results_vol)
print(f"\nMean CV AUC (vol): {mean_cv_auc_vol:.3f}  Lift: {mean_cv_lift_vol:+.3f}")

# Final vol model on full training pool
best_iter_vol = getattr(model_vol, "best_iteration_", None)
n_est_vol = int(best_iter_vol) if best_iter_vol and best_iter_vol > 0 else 200

final_model_vol = lgb.LGBMClassifier(
    n_estimators=n_est_vol,
    learning_rate=0.03,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    class_weight="balanced",
    verbose=-1,
)
final_model_vol.fit(X_train_pool_vol, y_train_pool_vol)
proba_holdout_vol = final_model_vol.predict_proba(X_holdout_vol)[:, 1]
pred_holdout_vol = (proba_holdout_vol > 0.5).astype(int)

oos_acc_vol = (pred_holdout_vol == y_holdout_vol).mean()
auc_holdout_vol = roc_auc_score(y_holdout_vol, proba_holdout_vol)
f1_holdout_vol = f1_score(y_holdout_vol, pred_holdout_vol, zero_division=0)
prec_holdout_vol = precision_score(y_holdout_vol, pred_holdout_vol, zero_division=0)
rec_holdout_vol = recall_score(y_holdout_vol, pred_holdout_vol, zero_division=0)
cm_vol = confusion_matrix(y_holdout_vol, pred_holdout_vol)
majority_baseline_vol = max(float(y_holdout_vol.mean()), 1.0 - float(y_holdout_vol.mean()))
lift_vol = oos_acc_vol - majority_baseline_vol

print(f"\n[PATH 2] Volatility-regime prediction — Final OOS metrics (untouched holdout):")
print(f"  Accuracy:          {oos_acc_vol:.3f}")
print(f"  AUC:               {auc_holdout_vol:.3f}")
print(f"  F1:                {f1_holdout_vol:.3f}")
print(f"  Precision:         {prec_holdout_vol:.3f}")
print(f"  Recall:            {rec_holdout_vol:.3f}")
print(f"  Majority baseline: {majority_baseline_vol:.3f}")
print(f"  Lift over majority:{lift_vol:+.3f}")
print(f"  Confusion matrix:")
print(f"    TN={cm_vol[0,0]}  FP={cm_vol[0,1]}")
print(f"    FN={cm_vol[1,0]}  TP={cm_vol[1,1]}")

# Feature importance (vol)
imp_vol = pd.Series(
    final_model_vol.feature_importances_, index=features_vol.columns
).sort_values(ascending=False)
print("\n[PATH 2] Top 10 features by gain (volatility):")
print(imp_vol.head(10))

# SHAP for vol target
shap_sample_vol = X_holdout_vol.sample(n=min(1000, len(X_holdout_vol)), random_state=42)
explainer_vol = shap.TreeExplainer(final_model_vol)
shap_values_vol = explainer_vol.shap_values(shap_sample_vol)

if isinstance(shap_values_vol, list):
    shap_arr_vol = shap_values_vol[1]
else:
    shap_arr_vol = shap_values_vol

mean_abs_shap_vol = pd.Series(
    np.abs(shap_arr_vol).mean(axis=0),
    index=shap_sample_vol.columns,
).sort_values(ascending=False)

print("\n--- [PATH 2] SHAP feature importance (volatility — mean |SHAP| on holdout) ---")
print(mean_abs_shap_vol.to_string())

# %% Verdict — vol target
print("\n" + "=" * 60)
print("[PATH 2] Volatility-regime prediction verdict:")
if auc_holdout_vol < 0.52:
    verdict_vol = "FAIL: No edge over random."
elif lift_vol < 0.02:
    verdict_vol = f"FAIL: No lift over majority baseline (lift={lift_vol:+.2%}, AUC={auc_holdout_vol:.3f})."
elif auc_holdout_vol >= 0.65:
    verdict_vol = f"WARNING: AUC={auc_holdout_vol:.3f} >= 0.65 — audit for lookahead bias."
elif lift_vol >= 0.05 and auc_holdout_vol >= 0.55:
    verdict_vol = f"PASS: Real vol-regime edge detected (lift={lift_vol:+.2%}, AUC={auc_holdout_vol:.3f})."
else:
    verdict_vol = f"MARGINAL: lift={lift_vol:+.2%}, AUC={auc_holdout_vol:.3f}. More features needed."
print(verdict_vol)
print("=" * 60)

# %% Side-by-side summary
print("\n" + "=" * 60)
print("SUMMARY — Direction vs Volatility-Regime Prediction")
print("=" * 60)
print(f"{'Target':<30} {'AUC':>6} {'Lift':>8} {'F1':>6} {'Verdict'}")
print("-" * 70)
print(f"{'Direction (sign next 4 bars)':<30} {auc_holdout:>6.3f} {lift:>+8.3f} {f1_holdout:>6.3f}  {verdict_dir}")
print(f"{'Vol regime (7d median)':<30} {auc_holdout_vol:>6.3f} {lift_vol:>+8.3f} {f1_holdout_vol:>6.3f}  {verdict_vol}")
print("=" * 60)
