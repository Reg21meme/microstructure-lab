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

    # ── Save results ──────────────────────────────────────────────────────────
    out = OUTPUT_DIR / f"{symbol}_ic_decay.csv"
    decay_df.to_csv(out, index=False)
    print(f"\nIC decay results saved to {out}")


if __name__ == "__main__":
    run()