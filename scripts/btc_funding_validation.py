# %%
"""BTC funding-rate edge validation (Phase 1 v2).

OHLCV:      Coinbase BTC-USD spot 15m (US-accessible, full year of history)
Funding:    Hyperliquid BTC perp hourly funding (US-accessible DEX)
Optional:   CoinGecko BTC dominance history (daily, forward-filled)
Optional:   Deribit BTC realized volatility (daily, forward-filled)

Target:   sign(future_log_return over next 4 bars) — symmetric ~50/50 split

Changes vs Phase 1 v1
---------------------
Path A:
  - Symmetric target: (future_log_return > 0) instead of > +10 bps
    → avoids majority-class collapse (~50/50 split, true direction signal)
  - class_weight='balanced' in LightGBM
    → forces the model to learn the minority class rather than always predict 0
  - Proper metrics: AUC, F1, precision, recall (not just accuracy)
  - Fixed verdict logic: lift over majority baseline + AUC thresholds
    (previous logic incorrectly flagged no-edge as lookahead)

Path B:
  - CoinGeckoSource: BTC dominance history (graceful fallback)
  - DeribitSource: BTC 30-day realized volatility (graceful fallback)
  - 14 new features: session indicators, intra-bar, vol z-score,
    BTC dominance, Deribit HV, cross-products
"""

# %% Fetch all data
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from data.sources.coinbase_source import CoinbaseSource
from data.sources.hyperliquid_source import HyperliquidSource
from data.sources.coingecko_source import CoinGeckoSource
from data.sources.deribit_source import DeribitSource

end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=365)
start_ms = int(start.timestamp() * 1000)
end_ms = int(end.timestamp() * 1000)

cb = CoinbaseSource()
hl = HyperliquidSource()
cg = CoinGeckoSource()
dr = DeribitSource()

print(f"Fetching BTC-USD 15m OHLCV from Coinbase: {start.date()} to {end.date()} …")
ohlcv = cb.fetch_ohlcv("BTC-USD", "15m", start_ms, end_ms)
print(f"  OHLCV rows: {len(ohlcv):,}")

print(f"Fetching BTC perp hourly funding from Hyperliquid …")
funding = hl.fetch_perp_funding("BTC", start_ms, end_ms)
print(f"  Funding rows: {len(funding):,}")

# Optional: BTC dominance (CoinGecko)
try:
    print("Fetching BTC dominance history from CoinGecko …")
    btc_dom = cg.fetch_btc_dominance_history(days=365)
    print(f"  BTC dominance rows: {len(btc_dom)}")
except Exception as e:
    print(f"  WARN: BTC dominance fetch failed ({e}), skipping that feature group")
    btc_dom = pd.DataFrame(columns=["btc_dominance"])

# Optional: Deribit realized volatility
try:
    print("Fetching BTC realized volatility from Deribit …")
    hv = dr.fetch_historical_volatility("BTC")
    print(f"  Deribit HV rows: {len(hv)}")
except Exception as e:
    print(f"  WARN: Deribit HV fetch failed ({e}), skipping that feature")
    hv = pd.DataFrame(columns=["hv_30d"])

# %% Merge all into single 15m grid
merged = ohlcv.copy()
# Forward-fill funding rate into the 15m price grid.
# Hyperliquid's funding endpoint doesn't return mark_price (all NaN), so we
# fill it with the close as a proxy and drop only rows with missing
# OHLCV/funding_rate (the features we actually use).
merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
merged["mark_price"] = merged["close"]  # proxy; not used in features

# Optional: BTC dominance (daily → forward-fill to 15m grid)
if not btc_dom.empty and "btc_dominance" in btc_dom.columns:
    merged["btc_dominance"] = btc_dom["btc_dominance"].reindex(merged.index, method="ffill")

# Optional: Deribit HV (daily → forward-fill to 15m grid)
if not hv.empty and "hv_30d" in hv.columns:
    merged["hv_30d"] = hv["hv_30d"].reindex(merged.index, method="ffill")

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

# %% Build target — symmetric ~50/50 split
# Previous version used > +10 bps which created a 35/65 imbalance, causing the
# model to learn "always predict 0" and achieve 65% accuracy with zero lift.
future_log_return = np.log(merged["close"].shift(-4) / merged["close"])
target = (future_log_return > 0).astype(int)  # sign of next 4-bar return — true direction

# Step 2: Split features into required vs optional
# LightGBM handles NaN natively (use_missing=True by default since 3.0).
# Only drop rows where REQUIRED features are NaN; optional columns keep NaN.
REQUIRED_FEATURE_COLS = [
    'ret_1', 'ret_4', 'ret_24',
    'rv_24', 'rv_96', 'rv_24_pct_rank_7d',
    'vol_zscore_24',
    'fund_current', 'fund_delta_1h', 'fund_delta_24h',
    'fund_pct_rank_7d', 'fund_zscore_7d', 'fund_x_price_div',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'bars_since_funding_settlement',
    'session_asia', 'session_eu', 'session_us', 'session_overnight',
    'intrabar_range', 'body_pct', 'upper_shadow_pct',
    'vol_zscore_96',
    'fund_x_vol', 'fund_x_ret', 'vol_x_ret',
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

print("\n--- Walk-forward CV ---")
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

print(f"\nFinal OOS metrics (untouched holdout):")
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

# %% Feature importance
imp = pd.Series(
    final_model.feature_importances_, index=features.columns
).sort_values(ascending=False)
print("\nTop 10 features by gain:")
print(imp.head(10))

# %% Simulated trading P&L (educational, not production)
fee_bps = 7.5
threshold = 0.55
positions = (proba_holdout > threshold).astype(int)

# Realized 4-bar log returns for the holdout window
realized_4bar = np.log(merged["close"].shift(-4) / merged["close"])
# Align to the holdout slice of valid rows
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
    f"\nSimulated trading (threshold={threshold}): {n_trades} trades, "
    f"total log-return {total_return:.4f}, "
    f"Sharpe proxy {sharpe_proxy:.2f}"
)

# %% Verdict — fixed logic: lift over majority baseline + AUC
# Previous logic: "if OOS >= 60% → lookahead" was wrong for imbalanced targets.
# A model that always predicts majority can get 65% accuracy with zero skill.
# Correct logic uses lift over majority baseline and AUC (rank-based, not threshold-based).
print("\n" + "=" * 60)
if auc_holdout < 0.52:
    verdict = "FAIL: No edge over random. Features need work."
elif lift < 0.02:
    verdict = f"FAIL: No lift over majority-class baseline (lift={lift:+.2%}, AUC={auc_holdout:.3f})."
elif auc_holdout >= 0.65:
    verdict = f"WARNING: AUC={auc_holdout:.3f} >= 0.65 — audit for lookahead bias."
elif lift >= 0.05 and auc_holdout >= 0.55:
    verdict = f"PASS: Real edge detected (lift={lift:+.2%}, AUC={auc_holdout:.3f}). Build Phase 2 infrastructure."
else:
    verdict = f"MARGINAL: lift={lift:+.2%}, AUC={auc_holdout:.3f}. More features needed."

print(verdict)
print("=" * 60)
