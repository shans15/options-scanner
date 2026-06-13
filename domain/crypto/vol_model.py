from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolModelArtifact:
    """Trained model + metadata. Persists to disk."""

    model: lgb.LGBMClassifier
    feature_names: list[str]
    train_window_start: pd.Timestamp
    train_window_end: pd.Timestamp
    n_train_samples: int
    cv_mean_auc: float
    cv_mean_lift: float


def train_vol_model(
    features: pd.DataFrame,
    target: pd.Series,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with class_weight='balanced' on the given features+target.

    Used for both training the production artifact and for inline validation.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix. Rows must correspond to target.
    target : pd.Series
        Binary target: 1 = vol expansion, 0 = contraction.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        LightGBM learning rate.
    num_leaves : int
        Number of leaves per tree.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted classifier.
    """
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        verbose=-1,
    )
    model.fit(features, target)
    return model


def save_model(artifact: VolModelArtifact, path: Path) -> None:
    """Pickle the artifact to disk. Creates parent directory if needed.

    Parameters
    ----------
    artifact : VolModelArtifact
        The trained model artifact to persist.
    path : Path
        Destination file path (e.g. cache/models/btc_vol_model.pkl).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> VolModelArtifact:
    """Load a previously saved artifact.

    Parameters
    ----------
    path : Path
        Path to the .pkl file written by save_model().

    Returns
    -------
    VolModelArtifact

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run `python3 -m scripts.train_btc_vol_model` to train the model first."
        )
    with open(path, "rb") as fh:
        artifact = pickle.load(fh)
    if not isinstance(artifact, VolModelArtifact):
        raise TypeError(
            f"Expected VolModelArtifact, got {type(artifact).__name__}. "
            "The file may be corrupt or from an incompatible version."
        )
    return artifact


def predict_proba(artifact: VolModelArtifact, features: pd.DataFrame) -> pd.Series:
    """Run inference. Returns a Series of P(vol expansion) indexed by features.index.

    Validates that features.columns match artifact.feature_names.

    Parameters
    ----------
    artifact : VolModelArtifact
        Loaded model artifact.
    features : pd.DataFrame
        Feature matrix with the same column set as the training data.

    Returns
    -------
    pd.Series
        Probability of vol expansion for each row, indexed by features.index.

    Raises
    ------
    ValueError
        If feature columns do not match the artifact's expected feature names.
    """
    expected = set(artifact.feature_names)
    provided = set(features.columns)

    missing = expected - provided
    extra = provided - expected

    if missing:
        raise ValueError(
            f"predict_proba: {len(missing)} feature(s) missing from input DataFrame: "
            f"{sorted(missing)}"
        )
    if extra:
        # Warn but don't raise — extra columns are silently dropped so the model
        # only sees its training features in the correct order.
        import warnings
        warnings.warn(
            f"predict_proba: {len(extra)} extra feature column(s) found and ignored: "
            f"{sorted(extra)}",
            stacklevel=2,
        )

    # Reorder to match training column order
    X = features[artifact.feature_names]
    proba = artifact.model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=features.index, name="vol_expansion_proba")
