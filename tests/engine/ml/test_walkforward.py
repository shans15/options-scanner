from __future__ import annotations

import pytest

from engine.ml.walkforward import walk_forward_splits


# ---------------------------------------------------------------------------
# test_walk_forward_splits_count
# ---------------------------------------------------------------------------

def test_walk_forward_splits_count():
    """walk_forward_splits must return exactly n_splits tuples."""
    for n_splits in (1, 3, 6, 10):
        splits = walk_forward_splits(1000, n_splits=n_splits)
        assert len(splits) == n_splits, (
            f"Expected {n_splits} splits, got {len(splits)}"
        )


# ---------------------------------------------------------------------------
# test_walk_forward_splits_train_before_test
# ---------------------------------------------------------------------------

def test_walk_forward_splits_train_before_test():
    """Every fold's test indices must come strictly after its train indices."""
    splits = walk_forward_splits(1000, n_splits=6)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_max = max(train_idx)
        test_min = min(test_idx)
        assert test_min > train_max, (
            f"Fold {i}: test starts at {test_min} but train ends at {train_max}"
        )


# ---------------------------------------------------------------------------
# test_walk_forward_splits_no_overlap
# ---------------------------------------------------------------------------

def test_walk_forward_splits_no_overlap():
    """train_idx and test_idx must be disjoint in every fold."""
    splits = walk_forward_splits(1000, n_splits=6)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set
        assert not overlap, (
            f"Fold {i}: train/test overlap at indices {overlap}"
        )

        # Also check val and test are disjoint
        val_set = set(val_idx)
        assert not (val_set & test_set), (
            f"Fold {i}: val/test overlap"
        )


# ---------------------------------------------------------------------------
# test_walk_forward_splits_expanding_train
# ---------------------------------------------------------------------------

def test_walk_forward_splits_expanding_train():
    """The training window must grow (or stay the same) across folds."""
    splits = walk_forward_splits(1000, n_splits=6)
    train_sizes = [len(train_idx) for train_idx, _, _ in splits]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] >= train_sizes[i - 1], (
            f"Train window shrank from fold {i-1} ({train_sizes[i-1]}) "
            f"to fold {i} ({train_sizes[i]})"
        )


# ---------------------------------------------------------------------------
# test_walk_forward_splits_indices_within_bounds
# ---------------------------------------------------------------------------

def test_walk_forward_splits_indices_within_bounds():
    """All indices must fall within [0, n-1]."""
    n = 500
    splits = walk_forward_splits(n, n_splits=4)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        for label, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            assert min(idx) >= 0, f"Fold {i} {label}: negative index"
            assert max(idx) < n, (
                f"Fold {i} {label}: index {max(idx)} >= n={n}"
            )


# ---------------------------------------------------------------------------
# test_walk_forward_splits_val_between_train_and_test
# ---------------------------------------------------------------------------

def test_walk_forward_splits_val_between_train_and_test():
    """val_idx must be between train and test in each fold."""
    splits = walk_forward_splits(1000, n_splits=6)
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_max = max(train_idx)
        val_min = min(val_idx)
        val_max = max(val_idx)
        test_min = min(test_idx)

        assert val_min > train_max, (
            f"Fold {i}: val starts ({val_min}) before train ends ({train_max})"
        )
        assert test_min > val_max, (
            f"Fold {i}: test starts ({test_min}) before val ends ({val_max})"
        )


# ---------------------------------------------------------------------------
# test_walk_forward_splits_invalid_inputs
# ---------------------------------------------------------------------------

def test_walk_forward_splits_invalid_n_splits():
    """n_splits=0 must raise ValueError."""
    with pytest.raises(ValueError, match="n_splits"):
        walk_forward_splits(1000, n_splits=0)


def test_walk_forward_splits_invalid_train_frac():
    """train_frac outside (0, 1) must raise ValueError."""
    with pytest.raises(ValueError, match="train_frac"):
        walk_forward_splits(1000, train_frac=1.0)

    with pytest.raises(ValueError, match="train_frac"):
        walk_forward_splits(1000, train_frac=0.0)


# ---------------------------------------------------------------------------
# test_walk_forward_splits_default_parameters
# ---------------------------------------------------------------------------

def test_walk_forward_splits_default_parameters():
    """Default call (6 splits, train_frac=0.5) must work for a realistic n."""
    n = 28_000  # ~12 months of 15m data after warm-up burn
    splits = walk_forward_splits(n)
    assert len(splits) == 6

    # First fold: train covers first ~50%
    train0, _, _ = splits[0]
    approx_50pct = int(n * 0.5)
    # Allow 1% tolerance for rounding
    assert abs(len(train0) - approx_50pct) <= int(n * 0.01)
