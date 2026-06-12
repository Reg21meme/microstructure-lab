"""
dataset.py
Purged and embargoed walk-forward cross-validation for time series data.

Follows López de Prado (2018) "Advances in Financial Machine Learning"
Chapter 7: Cross-Validation in Finance.

Key concepts:
  - Purging  : remove training samples whose labels overlap with test period
  - Embargo  : add a gap after test period to prevent autocorrelation leakage
  - Walk-forward: always train on past, test on future — never shuffle
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Iterator


class PurgedWalkForwardCV:
    """
    Walk-forward cross-validator with purging and embargo.

    Parameters
    ----------
    n_splits      : number of train/test folds
    label_horizon : number of rows the label looks forward (= purge length)
    embargo_pct   : fraction of test fold to use as embargo gap (default 0.01)
                    embargo_rows = max(1, int(n_test * embargo_pct))
    """

    def __init__(self,
                 n_splits: int = 5,
                 label_horizon: int = 5,
                 embargo_pct: float = 0.01):
        self.n_splits      = n_splits
        self.label_horizon = label_horizon
        self.embargo_pct   = embargo_pct

    def split(self, X: np.ndarray,
              y=None,
              groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index pairs.

        For each fold:
          1. Split data into train and test windows (time order)
          2. Purge: remove last label_horizon rows from train
             (their labels use data from the test window)
          3. Embargo: remove first embargo_rows rows from test
             (they're autocorrelated with the end of train)

        Yields
        ------
        train_idx : np.ndarray of int indices for training
        test_idx  : np.ndarray of int indices for testing
        """
        n       = len(X)
        indices = np.arange(n)

        # Size of each test fold
        test_size = n // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Test window: moves forward with each fold
            test_start = (fold + 1) * test_size
            test_end   = test_start + test_size
            if test_end > n:
                test_end = n

            # Train window: everything before test start
            train_end = test_start

            # ── Purge ────────────────────────────────────────────────────────
            # Remove last label_horizon rows from train — their labels
            # were computed using data that falls inside the test window
            purge_start = max(0, train_end - self.label_horizon)
            train_idx   = indices[:purge_start]

            # ── Embargo ───────────────────────────────────────────────────────
            # Skip first embargo_rows of test window — they're autocorrelated
            # with the end of the training window
            embargo_rows = max(1, int(test_size * self.embargo_pct))
            embargo_rows = max(embargo_rows, self.label_horizon)
            test_idx     = indices[test_start + embargo_rows: test_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def run_walk_forward(df: pd.DataFrame,
                     feature_cols: list[str],
                     label_col: str,
                     model: BaseEstimator,
                     scaler_class,
                     n_splits: int = 5,
                     label_horizon: int = 5) -> dict:
    """
    Run purged walk-forward CV and collect per-fold metrics.

    Parameters
    ----------
    df            : feature DataFrame, sorted by time
    feature_cols  : list of feature column names
    label_col     : name of label column
    model         : sklearn-compatible model (unfitted)
    scaler_class  : sklearn scaler class (e.g. StandardScaler)
    n_splits      : number of CV folds
    label_horizon : label lookahead in rows (for purging)

    Returns
    -------
    dict with per-fold IC, rank-IC, and aggregate statistics
    """
    from scipy.stats import spearmanr, pearsonr
    import copy

    # Drop NaNs in features or label
    cols    = feature_cols + [label_col]
    df_clean = df[cols].dropna().reset_index(drop=True)

    X = df_clean[feature_cols].values
    y = df_clean[label_col].values

    cv = PurgedWalkForwardCV(
        n_splits      = n_splits,
        label_horizon = label_horizon,
        embargo_pct   = 0.01,
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # Fit scaler on train only — never fit on test
        scaler  = scaler_class()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Fit model on train only
        m = copy.deepcopy(model)
        m.fit(X_train, y_train)

        # Predict on test
        y_pred = m.predict(X_test)

        # IC = Pearson correlation
        ic = pearsonr(y_pred, y_test).statistic if len(y_test) > 2 else np.nan

        # Rank-IC = Spearman correlation
        rank_ic = spearmanr(y_pred, y_test).statistic if len(y_test) > 2 else np.nan

        fold_results.append({
            "fold"        : fold + 1,
            "train_size"  : len(train_idx),
            "test_size"   : len(test_idx),
            "ic"          : ic,
            "rank_ic"     : rank_ic,
        })

        print(f"  Fold {fold+1}: train={len(train_idx)}, "
              f"test={len(test_idx)}, "
              f"IC={ic:.4f}, Rank-IC={rank_ic:.4f}")

    results_df = pd.DataFrame(fold_results)

    # Aggregate statistics
    ic_mean   = results_df["ic"].mean()
    ic_std    = results_df["ic"].std()
    ic_t_stat = ic_mean / (ic_std / np.sqrt(len(results_df))) if ic_std > 0 else np.nan

    rank_ic_mean = results_df["rank_ic"].mean()
    rank_ic_std  = results_df["rank_ic"].std()

    print(f"\n  Mean IC      : {ic_mean:.4f} ± {ic_std:.4f} (t={ic_t_stat:.2f})")
    print(f"  Mean Rank-IC : {rank_ic_mean:.4f} ± {rank_ic_std:.4f}")
    print(f"  IC > 0 folds : {(results_df['ic'] > 0).sum()}/{len(results_df)}")

    return {
        "fold_results" : results_df,
        "ic_mean"      : ic_mean,
        "ic_std"       : ic_std,
        "ic_t_stat"    : ic_t_stat,
        "rank_ic_mean" : rank_ic_mean,
        "rank_ic_std"  : rank_ic_std,
    }