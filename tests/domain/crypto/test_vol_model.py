from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domain.crypto.vol_model import (
    VolModelArtifact,
    load_model,
    predict_proba,
    save_model,
    train_vol_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n: int = 400, n_features: int = 10, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a small synthetic feature matrix and binary target."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.standard_normal((n, n_features)), index=idx, columns=feature_names)
    y = pd.Series((rng.random(n) > 0.5).astype(int), index=idx, name="target")
    return X, y


def _make_artifact(n: int = 200) -> VolModelArtifact:
    """Build and return a VolModelArtifact fitted on synthetic data."""
    X, y = _make_synthetic_data(n=n)
    model = train_vol_model(X, y, n_estimators=10)
    return VolModelArtifact(
        model=model,
        feature_names=list(X.columns),
        train_window_start=X.index[0],
        train_window_end=X.index[-1],
        n_train_samples=len(X),
        cv_mean_auc=0.71,
        cv_mean_lift=0.05,
    )


# ---------------------------------------------------------------------------
# train_vol_model
# ---------------------------------------------------------------------------

class TestTrainVolModel:
    def test_returns_fitted_classifier(self):
        X, y = _make_synthetic_data(n=200)
        model = train_vol_model(X, y, n_estimators=10)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_probabilities_in_unit_interval(self):
        X, y = _make_synthetic_data(n=200)
        model = train_vol_model(X, y, n_estimators=10)
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_different_hyperparams_accepted(self):
        X, y = _make_synthetic_data(n=100)
        model = train_vol_model(X, y, n_estimators=5, learning_rate=0.05, num_leaves=15)
        assert model is not None

    def test_class_weight_balanced_via_attribute(self):
        X, y = _make_synthetic_data(n=200)
        model = train_vol_model(X, y, n_estimators=10)
        assert model.class_weight == "balanced"


# ---------------------------------------------------------------------------
# save_model / load_model roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_feature_names(self):
        artifact = _make_artifact()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "model.pkl"
            save_model(artifact, path)
            loaded = load_model(path)
        assert loaded.feature_names == artifact.feature_names

    def test_roundtrip_preserves_metadata(self):
        artifact = _make_artifact()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_model(artifact, path)
            loaded = load_model(path)
        assert loaded.n_train_samples == artifact.n_train_samples
        assert loaded.cv_mean_auc == pytest.approx(artifact.cv_mean_auc)
        assert loaded.cv_mean_lift == pytest.approx(artifact.cv_mean_lift)

    def test_save_creates_parent_dir(self):
        artifact = _make_artifact()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a" / "b" / "c" / "model.pkl"
            assert not path.parent.exists()
            save_model(artifact, path)
            assert path.exists()

    def test_load_model_inference_matches(self):
        """Loaded model must produce same predict_proba as original model."""
        X, y = _make_synthetic_data(n=200)
        artifact = _make_artifact(n=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_model(artifact, path)
            loaded = load_model(path)

        proba_orig = artifact.model.predict_proba(X)[:, 1]
        proba_load = loaded.model.predict_proba(X)[:, 1]
        np.testing.assert_array_almost_equal(proba_orig, proba_load)

    def test_load_nonexistent_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/path/model.pkl"))

    def test_load_wrong_type_raises_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pkl"
            with open(path, "wb") as fh:
                pickle.dump({"not": "an artifact"}, fh)
            with pytest.raises(TypeError):
                load_model(path)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_returns_series_indexed_by_input(self):
        artifact = _make_artifact(n=200)
        X, _ = _make_synthetic_data(n=50, n_features=10)
        X.columns = artifact.feature_names
        result = predict_proba(artifact, X)
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.index.equals(X.index)

    def test_values_in_unit_interval(self):
        artifact = _make_artifact(n=200)
        X, _ = _make_synthetic_data(n=50, n_features=10)
        X.columns = artifact.feature_names
        result = predict_proba(artifact, X)
        assert (result >= 0.0).all() and (result <= 1.0).all()

    def test_missing_features_raises_value_error(self):
        artifact = _make_artifact(n=200)
        # Provide only 5 of 10 features
        X, _ = _make_synthetic_data(n=20, n_features=5)
        X.columns = artifact.feature_names[:5]
        with pytest.raises(ValueError, match="feature"):
            predict_proba(artifact, X)

    def test_extra_features_warn_but_succeed(self):
        """Extra columns should trigger a warning, not an error."""
        import warnings
        artifact = _make_artifact(n=200)
        X, _ = _make_synthetic_data(n=20, n_features=10)
        X.columns = artifact.feature_names
        # Add extra column
        X["extra_col"] = 0.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = predict_proba(artifact, X)
        assert len(result) == 20
        assert any("extra" in str(warning.message).lower() for warning in w)

    def test_column_order_does_not_matter(self):
        """Shuffled column order should give same result as canonical order."""
        artifact = _make_artifact(n=200)
        X, _ = _make_synthetic_data(n=30, n_features=10)
        X.columns = artifact.feature_names

        # Shuffle columns
        shuffled_cols = list(reversed(artifact.feature_names))
        X_shuffled = X[shuffled_cols]

        proba_canonical = predict_proba(artifact, X)
        proba_shuffled = predict_proba(artifact, X_shuffled)

        pd.testing.assert_series_equal(
            proba_canonical, proba_shuffled, check_exact=False, atol=1e-10
        )


# ---------------------------------------------------------------------------
# VolModelArtifact dataclass
# ---------------------------------------------------------------------------

class TestVolModelArtifact:
    def test_is_frozen(self):
        artifact = _make_artifact()
        with pytest.raises((AttributeError, TypeError)):
            artifact.cv_mean_auc = 0.99  # type: ignore[misc]

    def test_stores_timestamps(self):
        artifact = _make_artifact(n=100)
        assert isinstance(artifact.train_window_start, pd.Timestamp)
        assert isinstance(artifact.train_window_end, pd.Timestamp)
        assert artifact.train_window_end > artifact.train_window_start

    def test_n_train_samples_positive(self):
        artifact = _make_artifact(n=100)
        assert artifact.n_train_samples == 100


# ---------------------------------------------------------------------------
# Smoke test: scripts fail gracefully when model file is absent
# ---------------------------------------------------------------------------

class TestScriptGracefulFailure:
    def test_watchlist_fails_without_model(self, monkeypatch):
        """crypto_vol_watchlist main() must return non-zero when model is missing."""
        import importlib
        import scripts.crypto_vol_watchlist as wl
        importlib.reload(wl)

        # Point MODEL_PATH to a guaranteed non-existent path
        monkeypatch.setattr(wl, "MODEL_PATH", Path("/nonexistent/btc_vol_model.pkl"))
        result = wl.main()
        assert result != 0, "Expected non-zero exit code when model file is missing"
