from __future__ import annotations


def walk_forward_splits(
    n: int,
    n_splits: int = 6,
    train_frac: float = 0.5,
) -> list[tuple[range, range, range]]:
    """Return walk-forward cross-validation index splits.

    Each fold uses an expanding training window followed by equal-length
    validation and test windows. The test set always comes strictly after
    the training set (no shuffling or random sampling).

    Parameters
    ----------
    n : int
        Total number of samples in the data pool (e.g. len(X_train_pool)).
    n_splits : int
        Number of folds. Default 6.
    train_frac : float
        Fraction of the total pool used as train data in the *first* fold.
        For subsequent folds the train window expands by one step.

    Returns
    -------
    list of (train_idx, val_idx, test_idx)
        Each element is a tuple of three ``range`` objects.

    Example (n_splits=6, train_frac=0.5, n=1000)
    -----------------------------------------------
    Fold 0: train [0:500]   val [500:583]  test [583:667]
    Fold 1: train [0:583]   val [583:667]  test [667:750]
    Fold 2: train [0:667]   val [667:750]  test [750:833]
    ...
    Fold 5: train [0:833]   val [833:917]  test [917:1000]

    Note: The step size is (1 - train_frac) / (n_splits + 1) * n, so that
    after n_splits folds the test set ends exactly at n.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    # The remaining fraction is divided into (n_splits + 1) equal steps:
    #   one step per fold for the test window, plus the initial gap already
    #   covered by train_frac.
    #
    # Fold k (0-indexed):
    #   train_end  = train_frac * n  + k * step
    #   val_end    = train_end + step
    #   test_end   = val_end   + step
    #
    # After n_splits folds:  train_frac + (n_splits + 1) * step = 1.0
    #   => step = (1 - train_frac) / (n_splits + 1)

    step = (1.0 - train_frac) / (n_splits + 1)

    splits: list[tuple[range, range, range]] = []
    for k in range(n_splits):
        train_end = int(round((train_frac + k * step) * n))
        val_end = int(round((train_frac + (k + 1) * step) * n))
        test_end = int(round((train_frac + (k + 2) * step) * n))

        # Clamp to [0, n]
        train_end = max(1, min(train_end, n))
        val_end = max(train_end + 1, min(val_end, n))
        test_end = max(val_end + 1, min(test_end, n))

        train_idx = range(0, train_end)
        val_idx = range(train_end, val_end)
        test_idx = range(val_end, test_end)

        splits.append((train_idx, val_idx, test_idx))

    return splits
