"""
validate.py
IC decay curve and Newey-West corrected t-statistics.

Runs walk-forward CV at multiple horizons to build the IC decay curve,
then applies Newey-West HAC correction to get honest standard errors.

Usage:
    python3 -m mslab.models.validate
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

ROOT       = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR   = ROOT / "data" / "features"
OUTPUT_DIR = ROOT / "data" / "results"
FIGURES    = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "depth_imbalance_5",
    "micro_price_deviation",
    "ofi",
    "mlofi_pc1",
    "realized_vol",
    "spread",
]


def newey_west_tstat(ic_series: np.ndarray, max_lags: int = None) -> dict:
    """
    Compute Newey-West HAC corrected t-statistic for mean IC.

    Parameters
    ----------
    ic_series : array of per-fold IC values
    max_lags  : number of lags for HAC correction
                default: floor(4 * (T/100)^(2/9)) per Andrews (1991)

    Returns
    -------
    dict with naive and corrected t-stats and standard errors
    """
    T = len(ic_series)
    if T < 3:
        return {"t_naive": np.nan, "t_nw": np.nan,
                "se_naive": np.nan, "se_nw": np.nan}

    if max_lags is None:
        max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
        max_lags = max(1, max_lags)

    mean_ic = np.mean(ic_series)

    # Naive standard error (assumes independence)
    se_naive = np.std(ic_series, ddof=1) / np.sqrt(T)
    t_naive  = mean_ic / se_naive if se_naive > 0 else np.nan

    # Newey-West HAC standard error
    # Regress IC on a constant and extract HAC standard error
    X  = np.ones((T, 1))
    y  = ic_series.reshape(-1, 1)
    try:
        ols    = sm.OLS(y, X).fit()
        hac    = ols.get_robustcov_results(cov_type='HAC', maxlags=max_lags)
        se_nw  = float(np.sqrt(hac.cov_params()[0, 0]))
        t_nw   = mean_ic / se_nw if se_nw > 0 else np.nan
    except Exception:
        se_nw = np.nan
        t_nw  = np.nan

    return {
        "mean_ic" : mean_ic,
        "se_naive": se_naive,
        "t_naive" : t_naive,
        "se_nw"   : se_nw,
        "t_nw"    : t_nw,
        "max_lags": max_lags,
    }


def compute_ic_decay(df: pd.DataFrame,
                     feature_cols: list[str],
                     label_cols: list[str],
                     horizons_labels: list[str],
                     n_splits: int = 5) -> pd.DataFrame:
    """
    Compute walk-forward IC at multiple horizons to build decay curve.

    Parameters
    ----------
    df              : feature DataFrame sorted by time
    feature_cols    : list of feature column names
    label_cols      : list of label column names (one per horizon)
    horizons_labels : human-readable horizon labels (e.g. ['5s', '10s', '20s'])
    n_splits        : number of CV folds

    Returns
    -------
    DataFrame with columns: horizon, ic_mean, ic_std, t_naive, t_nw
    """
    from mslab.research.dataset import run_walk_forward
    import copy

    results = []

    for label_col, horizon_label in zip(label_cols, horizons_labels):
        print(f"\nHorizon {horizon_label} (label={label_col}):")

        wf = run_walk_forward(
            df           = df,
            feature_cols = feature_cols,
            label_col    = label_col,
            model        = Ridge(alpha=1.0),
            scaler_class = StandardScaler,
            n_splits     = n_splits,
            label_horizon= 5,
        )

        # Newey-West corrected t-stat on the fold IC series
        ic_series = wf["fold_results"]["ic"].values
        nw        = newey_west_tstat(ic_series)

        results.append({
            "horizon"   : horizon_label,
            "ic_mean"   : wf["ic_mean"],
            "ic_std"    : wf["ic_std"],
            "rank_ic"   : wf["rank_ic_mean"],
            "t_naive"   : wf["ic_t_stat"],
            "t_nw"      : nw["t_nw"],
            "se_nw"     : nw["se_nw"],
        })

        print(f"  Newey-West t-stat: {nw['t_nw']:.2f} "
              f"(naive: {wf['ic_t_stat']:.2f}, "
              f"max_lags={nw['max_lags']})")

    return pd.DataFrame(results)


def plot_ic_decay(decay_df: pd.DataFrame) -> None:
    """Plot IC decay curve with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = range(len(decay_df))

    # ── IC decay with error bars ──────────────────────────────────────────────
    ax1.bar(x, decay_df["ic_mean"], color="steelblue", alpha=0.7,
            yerr=decay_df["ic_std"], capsize=5, label="IC ± std")
    ax1.plot(x, decay_df["rank_ic"], "o--", color="darkorange",
             label="Rank-IC", linewidth=1.5)
    ax1.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(decay_df["horizon"])
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("IC")
    ax1.set_title("IC Decay Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── t-stat comparison: naive vs Newey-West ────────────────────────────────
    width = 0.35
    x_arr = np.array(list(x))
    ax2.bar(x_arr - width/2, decay_df["t_naive"], width,
            color="steelblue", alpha=0.7, label="Naive t-stat")
    ax2.bar(x_arr + width/2, decay_df["t_nw"],    width,
            color="darkorange", alpha=0.7, label="Newey-West t-stat")
    ax2.axhline(2.0, color="red", linewidth=0.8, linestyle="--",
                label="t=2.0 significance threshold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(decay_df["horizon"])
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("t-statistic")
    ax2.set_title("Naive vs Newey-West t-statistic")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("IC Decay and Statistical Significance", fontsize=13)
    plt.tight_layout()
    out = FIGURES / "ic_decay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")

def plot_calibration(y_test: np.ndarray,
                     y_prob: np.ndarray,
                     n_bins: int = 10) -> dict:
    """
    Plot calibration curve and compute Brier score.

    Parameters
    ----------
    y_test : true binary labels (0 or 1)
    y_prob : predicted probabilities for class 1
    n_bins : number of probability buckets

    Returns
    -------
    dict with brier_score and brier_skill_score
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)

    # Brier score
    brier = brier_score_loss(y_test, y_prob)

    # Brier skill score vs naive baseline (always predict base rate)
    base_rate    = y_test.mean()
    brier_base   = brier_score_loss(y_test,
                                    np.full_like(y_prob, base_rate))
    brier_skill  = 1 - brier / brier_base

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Calibration curve ─────────────────────────────────────────────────────
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, "o-", color="steelblue",
             linewidth=2, markersize=6, label="Model")
    ax1.fill_between(prob_pred, prob_true, prob_pred,
                     alpha=0.15, color="steelblue")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"Calibration Curve\nBrier={brier:.4f}, "
                  f"Skill={brier_skill:.4f}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ── Predicted probability distribution ────────────────────────────────────
    ax2.hist(y_prob[y_test == 0], bins=30, alpha=0.6,
             color="salmon", label="Actual down", density=True)
    ax2.hist(y_prob[y_test == 1], bins=30, alpha=0.6,
             color="steelblue", label="Actual up", density=True)
    ax2.axvline(0.5, color="black", linewidth=1, linestyle="--")
    ax2.set_xlabel("Predicted probability (up)")
    ax2.set_ylabel("Density")
    ax2.set_title("Predicted Probability Distribution by Outcome")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")

    print(f"Brier score      : {brier:.4f}")
    print(f"Brier skill score: {brier_skill:.4f} "
          f"(>0 means beats naive baseline)")

    return {
        "brier"      : brier,
        "brier_skill": brier_skill,
        "prob_true"  : prob_true,
        "prob_pred"  : prob_pred,
    }


def compute_deflated_sharpe(y_test: np.ndarray,
                             y_prob: np.ndarray,
                             n_strategies_tried: int = 1) -> dict:
    """
    Compute Sharpe ratio and Deflated Sharpe Ratio (DSR).

    Constructs a simple PnL series: go long when model predicts up (prob > 0.5),
    short when it predicts down. PnL = predicted_direction * actual_move.

    DSR corrects for multiple testing — how many strategies did you try
    before finding this one?

    Parameters
    ----------
    y_test              : actual binary outcomes (0 or 1)
    y_prob              : predicted probabilities
    n_strategies_tried  : number of strategy configurations tested
                          (used to compute expected max Sharpe under null)

    Returns
    -------
    dict with sharpe, dsr, and supporting statistics
    """
    from scipy.stats import norm

    # Simple PnL: +1 if correct direction, -1 if wrong
    predicted_direction = (y_prob > 0.5).astype(float) * 2 - 1  # +1 or -1
    actual_direction    = (y_test > 0.5).astype(float) * 2 - 1   # +1 or -1
    pnl                 = predicted_direction * actual_direction   # +1 or -1

    T       = len(pnl)
    mean_r  = pnl.mean()
    std_r   = pnl.std(ddof=1)
    sharpe  = mean_r / std_r * np.sqrt(T) if std_r > 0 else np.nan

    skew    = pd.Series(pnl).skew()
    kurt    = pd.Series(pnl).kurt() + 3  # scipy returns excess kurtosis

    # Expected maximum Sharpe under null (Bailey & López de Prado 2014)
    # E[max SR] ≈ (1 - euler_gamma) * Z(1 - 1/N) + euler_gamma * Z(1 - 1/(N*e))
    # Simplified approximation for small N:
    euler_gamma = 0.5772
    if n_strategies_tried > 1:
        sr_star = (euler_gamma * norm.ppf(1 - 1 / n_strategies_tried) +
                   (1 - euler_gamma) * norm.ppf(1 - 1 / (n_strategies_tried * np.e)))
    else:
        sr_star = 0.0  # no multiple testing adjustment needed

    # DSR formula
    if std_r > 0 and not np.isnan(sharpe):
        sr_annualized = mean_r / std_r  # per-observation Sharpe
        numerator     = (sr_annualized - sr_star) * np.sqrt(T - 1)
        denominator   = np.sqrt(1 - skew * sr_annualized +
                                ((kurt - 1) / 4) * sr_annualized ** 2)
        if denominator > 0:
            dsr = norm.cdf(numerator / denominator)
        else:
            dsr = np.nan
    else:
        dsr = np.nan

    print(f"\nSharpe & Deflated Sharpe:")
    print(f"  Observations        : {T}")
    print(f"  Mean PnL per trade  : {mean_r:.4f}")
    print(f"  Std PnL             : {std_r:.4f}")
    print(f"  Sharpe (per obs)    : {sharpe:.4f}")
    print(f"  Skewness            : {skew:.4f}")
    print(f"  Kurtosis            : {kurt:.4f}")
    print(f"  SR* (expected max)  : {sr_star:.4f}")
    print(f"  Deflated Sharpe     : {dsr:.4f}")
    print(f"  Interpretation      : {dsr*100:.1f}% probability of genuine edge")

    return {
        "sharpe" : sharpe,
        "dsr"    : dsr,
        "sr_star": sr_star,
        "mean_r" : mean_r,
        "std_r"  : std_r,
        "skew"   : skew,
        "kurt"   : kurt,
        "T"      : T,
    }

def run(symbol: str = "BTCUSDT") -> None:
    print(f"Validation for {symbol}")
    print("=" * 50)

    df = pd.read_parquet(DATA_DIR / f"{symbol}_features.parquet")

    # Drop rows with NaN in any feature or label
    all_cols = FEATURE_COLS + [
        "future_mid_move_5",
        "future_mid_move_10",
        "future_mid_move_20",
    ]
    df = df.dropna(subset=all_cols).reset_index(drop=True)
    print(f"Loaded {len(df)} clean rows")

    # ── IC decay curve ────────────────────────────────────────────────────────
    print("\nComputing IC decay curve across horizons...")
    decay_df = compute_ic_decay(
        df              = df,
        feature_cols    = FEATURE_COLS,
        label_cols      = [
            "future_mid_move_5",
            "future_mid_move_10",
            "future_mid_move_20",
        ],
        horizons_labels = ["5s", "10s", "20s"],
        n_splits        = 5,
    )

    print("\n" + "=" * 50)
    print("IC DECAY SUMMARY")
    print("=" * 50)
    print(decay_df.to_string(index=False, float_format="{:.4f}".format))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_ic_decay(decay_df)

# ── Save IC decay results ─────────────────────────────────────────────────
    out = OUTPUT_DIR / f"{symbol}_ic_decay.csv"
    decay_df.to_csv(out, index=False)
    print(f"\nIC decay results saved to {out}")

    # ── Calibration and Brier score ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("CALIBRATION & BRIER SCORE")
    print("=" * 50)

    # Load saved logistic results
    import pickle
    results_path = OUTPUT_DIR / f"{symbol}_baseline_results.pkl"
    with open(results_path, "rb") as f:
        baseline = pickle.load(f)

    y_test = baseline["logistic"]["y_test"]
    y_prob = baseline["logistic"]["y_prob_test"]

    cal_results = plot_calibration(y_test, y_prob)

    # ── Deflated Sharpe ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("DEFLATED SHARPE RATIO")
    print("=" * 50)
    dsr_results = compute_deflated_sharpe(
        y_test             = y_test,
        y_prob             = y_prob,
        n_strategies_tried = 1,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FULL VALIDATION SUMMARY")
    print("=" * 50)
    print(f"IC at 5s horizon    : {decay_df.iloc[0]['ic_mean']:.4f} "
          f"(NW t={decay_df.iloc[0]['t_nw']:.2f})")
    print(f"IC at 10s horizon   : {decay_df.iloc[1]['ic_mean']:.4f} "
          f"(NW t={decay_df.iloc[1]['t_nw']:.2f})")
    print(f"IC at 20s horizon   : {decay_df.iloc[2]['ic_mean']:.4f} "
          f"(NW t={decay_df.iloc[2]['t_nw']:.2f})")
    print(f"Brier skill score   : {cal_results['brier_skill']:.4f}")
    print(f"Sharpe              : {dsr_results['sharpe']:.4f}")
    print(f"Deflated Sharpe     : {dsr_results['dsr']:.4f}")


if __name__ == "__main__":
    run()