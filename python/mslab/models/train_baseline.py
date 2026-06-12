"""
train_baseline.py
Trains logistic and ridge regression baselines on microstructure features.

Usage:
    python3 -m mslab.models.train_baseline
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle

ROOT       = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR   = ROOT / "data" / "features"
OUTPUT_DIR = ROOT / "data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature and label configuration ──────────────────────────────────────────
FEATURE_COLS = [
    "depth_imbalance_5",
    "micro_price_deviation",
    "ofi",
    "mlofi_pc1",
    "realized_vol",
    "spread",
]

LABEL_REGRESSION    = "future_mid_move_5"   # continuous target for ridge
LABEL_CLASSIFICATION = "future_mid_move_5"  # binarized for logistic


def load_clean_data(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Load feature Parquet and drop rows with any NaN in features or label."""
    df = pd.read_parquet(DATA_DIR / f"{symbol}_features.parquet")

    # Drop rows where any feature or label is NaN
    cols_needed = FEATURE_COLS + [LABEL_REGRESSION]
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    print(f"Loaded {len(df)} clean rows after dropping NaNs")
    return df


def time_split(df: pd.DataFrame,
               train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test by time — never shuffle.
    First train_frac of rows = train. Rest = test.
    """
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


def make_binary_label(series: pd.Series) -> pd.Series:
    """
    Convert continuous mid-move to binary direction label.
    1 = price went up, 0 = price went down.
    Rows where move == 0 are dropped (no signal).
    """
    return series.apply(lambda x: 1 if x > 0 else (0 if x < 0 else np.nan))


def train_ridge(train: pd.DataFrame,
                test: pd.DataFrame) -> dict:
    """
    Train ridge regression to predict future mid-price move magnitude.
    Returns dict with model, scaler, and evaluation metrics.
    """
    X_train = train[FEATURE_COLS].values
    y_train = train[LABEL_REGRESSION].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[LABEL_REGRESSION].values

    # Standardize features — ridge is sensitive to scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train ridge with cross-validated alpha
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Information Coefficient = Pearson correlation between predicted and actual
    ic_train = np.corrcoef(y_pred_train, y_train)[0, 1]
    ic_test  = np.corrcoef(y_pred_test,  y_test)[0, 1]

    # Rank IC = Spearman correlation (more robust to outliers)
    from scipy.stats import spearmanr
    rank_ic_train = spearmanr(y_pred_train, y_train).statistic
    rank_ic_test  = spearmanr(y_pred_test,  y_test).statistic

    print(f"\nRidge Regression Results:")
    print(f"  Train IC      : {ic_train:.4f}")
    print(f"  Test  IC      : {ic_test:.4f}")
    print(f"  Train Rank-IC : {rank_ic_train:.4f}")
    print(f"  Test  Rank-IC : {rank_ic_test:.4f}")
    print(f"  Coefficients  :")
    for name, coef in zip(FEATURE_COLS, model.coef_):
        print(f"    {name:30s}: {coef:+.4f}")

    return {
        "model"         : model,
        "scaler"        : scaler,
        "ic_train"      : ic_train,
        "ic_test"       : ic_test,
        "rank_ic_train" : rank_ic_train,
        "rank_ic_test"  : rank_ic_test,
        "feature_cols"  : FEATURE_COLS,
        "y_pred_test"   : y_pred_test,
        "y_test"        : y_test,
    }


def train_logistic(train: pd.DataFrame,
                   test: pd.DataFrame) -> dict:
    """
    Train logistic regression to predict price direction (up vs down).
    Returns dict with model, scaler, and evaluation metrics.
    """
    # Binarize labels — drop zero moves
    train = train.copy()
    test  = test.copy()

    train["label"] = make_binary_label(train[LABEL_CLASSIFICATION])
    test["label"]  = make_binary_label(test[LABEL_CLASSIFICATION])

    train = train.dropna(subset=["label"])
    test  = test.dropna(subset=["label"])

    X_train = train[FEATURE_COLS].values
    y_train = train["label"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["label"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    # Predictions
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]
    y_pred_test  = model.predict(X_test)

    # Metrics
    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test  = roc_auc_score(y_test,  y_prob_test)
    acc_test  = (y_pred_test == y_test).mean()
    baseline  = max(y_test.mean(), 1 - y_test.mean())

    print(f"\nLogistic Regression Results:")
    print(f"  Train AUC         : {auc_train:.4f}")
    print(f"  Test  AUC         : {auc_test:.4f}")
    print(f"  Test  Accuracy    : {acc_test:.4f}")
    print(f"  Naive baseline    : {baseline:.4f} (majority class)")
    print(f"  Beats baseline    : {acc_test > baseline}")
    print(f"  Coefficients      :")
    for name, coef in zip(FEATURE_COLS, model.coef_[0]):
        print(f"    {name:30s}: {coef:+.4f}")

    return {
        "model"        : model,
        "scaler"       : scaler,
        "auc_train"    : auc_train,
        "auc_test"     : auc_test,
        "acc_test"     : acc_test,
        "baseline"     : baseline,
        "feature_cols" : FEATURE_COLS,
        "y_prob_test"  : y_prob_test,
        "y_test"       : y_test,
    }


def run(symbol: str = "BTCUSDT") -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from mslab.research.dataset import run_walk_forward

    print(f"Training baseline models for {symbol}")
    print("=" * 50)

    df          = load_clean_data(symbol)
    train, test = time_split(df)

    ridge_results    = train_ridge(train, test)
    logistic_results = train_logistic(train, test)

    # ── Purged walk-forward CV ────────────────────────────────────────────────
    print("\nPurged Walk-Forward CV (Ridge, label=future_mid_move_5):")
    wf_results = run_walk_forward(
        df            = df,
        feature_cols  = FEATURE_COLS,
        label_col     = LABEL_REGRESSION,
        model         = Ridge(alpha=1.0),
        scaler_class  = StandardScaler,
        n_splits      = 5,
        label_horizon = 5,
    )

# Save results for validate.py
    results = {
        "ridge"      : ridge_results,
        "logistic"   : logistic_results,
        "walk_forward": wf_results,
        "train"      : train,
        "test"       : test,
    }
    out_path = OUTPUT_DIR / f"{symbol}_baseline_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Ridge  test IC          : {ridge_results['ic_test']:.4f}")
    print(f"Ridge  test Rank-IC     : {ridge_results['rank_ic_test']:.4f}")
    print(f"Logistic test AUC       : {logistic_results['auc_test']:.4f}")
    print(f"Logistic test Acc       : {logistic_results['acc_test']:.4f}")
    print(f"Naive baseline          : {logistic_results['baseline']:.4f}")
    print(f"Walk-forward mean IC    : {wf_results['ic_mean']:.4f} ± {wf_results['ic_std']:.4f}")
    print(f"Walk-forward mean Rank-IC: {wf_results['rank_ic_mean']:.4f}")
    print(f"Walk-forward IC t-stat  : {wf_results['ic_t_stat']:.2f}")

if __name__ == "__main__":
    run()