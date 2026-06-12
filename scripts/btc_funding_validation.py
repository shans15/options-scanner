# %%
"""BTC funding-rate edge validation.

Goal: prove or disprove that funding-rate features predict short-horizon BTC direction.
Target: P(BTC log-return > +10 bps in next 4 bars of 15-min data).

Decision rule:
  OOS accuracy >= 53%: edge exists, proceed to Phase 2.
  OOS accuracy < 53%:  no edge, more feature work needed before building infra.
  OOS accuracy >= 60%: very likely lookahead bias, audit features.

Data source: Hyperliquid (US-accessible DEX perp exchange, public REST API).
Binance global API is geo-blocked from US (HTTP 451); Binance.US has no perp futures.
Hyperliquid funding settles HOURLY (every 4 bars at 15m cadence), vs Binance's 8h.
settlement_period_bars=4 is passed to build_features accordingly.

Run with:
    python -m scripts.btc_funding_validation
"""

# %% Fetch data
from datetime import datetime, timezone, timedelta

from data.sources.hyperliquid_source import HyperliquidSource

src = HyperliquidSource()
end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=365)

print(f"Fetching BTC 15m OHLCV from Hyperliquid: {start.date()} to {end.date()} …")
ohlcv = src.fetch_ohlcv(
    "BTC", "15m",
    int(start.timestamp() * 1000),
    int(end.timestamp() * 1000),
)
print(f"  OHLCV rows: {len(ohlcv):,}")

print("Fetching BTC perpetual funding rates (hourly) from Hyperliquid …")
funding = src.fetch_perp_funding(
    "BTC",
    int(start.timestamp() * 1000),
    int(end.timestamp() * 1000),
)
print(f"  Funding rows: {len(funding):,}")

# %% Merge + feature build
import pandas as pd

merged = ohlcv.copy()
# Forward-fill funding rate into the 15m price grid
merged["funding_rate"] = funding["funding_rate"].reindex(merged.index, method="ffill")
merged["mark_price"] = funding["mark_price"].reindex(merged.index, method="ffill")
merged = merged.dropna()
print(f"\nMerged rows after dropna: {len(merged):,}")

from domain.crypto.features import build_features

features = build_features(merged, settlement_period_bars=4)  # Hyperliquid: 1h = 4 bars at 15m
print(f"Feature columns ({len(features.columns)}): {list(features.columns)}")

# %% Build target
import numpy as np

future_log_return = np.log(merged["close"].shift(-4) / merged["close"])
target = (future_log_return > 0.001).astype(int)  # > +10 bps in next 4 bars

# Drop rows where target is undefined or any feature is NaN
valid = ~target.isna() & ~features.isna().any(axis=1)
features = features[valid]
target = target[valid]
print(f"\nValid samples: {len(features):,}  (base rate: {target.mean():.3f})")

# %% Hold out final 20% as untouched test set
holdout_start = int(len(features) * 0.8)
X_train_pool = features.iloc[:holdout_start]
y_train_pool = target.iloc[:holdout_start]
X_holdout = features.iloc[holdout_start:]
y_holdout = target.iloc[holdout_start:]
print(f"\nTrain pool: {len(X_train_pool):,}  Holdout: {len(X_holdout):,}")

# %% Walk-forward CV on the training pool
import lightgbm as lgb

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
    fold_results.append({"fold": i, "acc": acc, "n_test": len(y_te), "base_rate": y_te.mean()})
    print(f"Fold {i}: acc={acc:.3f}  base_rate={y_te.mean():.3f}  n={len(y_te)}")

mean_cv_acc = sum(r["acc"] for r in fold_results) / len(fold_results)
print(f"\nMean CV accuracy: {mean_cv_acc:.3f}")

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
    verbose=-1,
)
final_model.fit(X_train_pool, y_train_pool)
proba_holdout = final_model.predict_proba(X_holdout)[:, 1]
pred_holdout = (proba_holdout > 0.5).astype(int)
oos_acc = (pred_holdout == y_holdout).mean()

print(f"\nFinal OOS accuracy (untouched holdout): {oos_acc:.3f}")
print(f"Base rate on holdout:                   {y_holdout.mean():.3f}")
naive_acc = max(float(y_holdout.mean()), 1.0 - float(y_holdout.mean()))
print(f"Lift over base rate:                    {oos_acc - naive_acc:+.3f}")

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
n_trades = positions.sum()
total_return = float(trade_returns.sum())
std = float(trade_returns.std())
bars_per_year = 24 * 4 * 252
sharpe_proxy = (
    float(trade_returns.mean()) / std * (bars_per_year ** 0.5) if std > 0 else 0.0
)
print(
    f"\nSimulated trading: {n_trades} trades, "
    f"total log-return {total_return:.4f}, "
    f"Sharpe proxy {sharpe_proxy:.2f}"
)

# %% Verdict
print("\n" + "=" * 60)
if oos_acc >= 0.60:
    print("WARNING  OOS accuracy >= 60% — likely lookahead bias, audit features.")
elif oos_acc >= 0.53:
    print("PASS  Edge exists. Build Phase 2 infrastructure.")
else:
    print("FAIL  No clear edge. More feature engineering needed before Phase 2.")
print("=" * 60)
