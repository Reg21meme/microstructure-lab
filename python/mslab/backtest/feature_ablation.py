"""
feature_ablation.py
Feature ablation study — drop one feature at a time and measure
the impact on out-of-sample IC.

Two experiments:
  1. Leave-one-out: drop each feature, record IC degradation
  2. Standalone: each feature alone, record IC

The features that cause the largest IC drop when removed are
the ones actually carrying the signal.

Usage:
    python3 -m mslab.backtest.feature_ablation
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

from mslab.models.train_baseline import (
    load_clean_data, time_split, train_ridge, FEATURE_COLS
)

SYMBOL = "BTCUSDT"


def ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson IC between predictions and realized returns."""
    if len(y_true) < 2:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_with_features(train: pd.DataFrame,
                      test: pd.DataFrame,
                      features: list[str]) -> float:
    """Train ridge on subset of features, return test IC."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    target = "future_mid_move_5"
    train_clean = train[features + [target]].dropna()
    test_clean  = test[features + [target]].dropna()

    if len(train_clean) < 10 or len(test_clean) < 10:
        return np.nan

    X_train = train_clean[features].values
    y_train = train_clean[target].values
    X_test  = test_clean[features].values
    y_test  = test_clean[target].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return ic(y_test, y_pred)


def run_ablation(symbol: str = SYMBOL) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_df     = load_clean_data(symbol)
    train, test = time_split(feat_df, train_frac=0.7)

    # ── Baseline: all features ────────────────────────────────────────────────
    baseline_ic = run_with_features(train, test, FEATURE_COLS)
    print(f"Baseline IC (all {len(FEATURE_COLS)} features): {baseline_ic:.4f}")
    print()

    # ── Leave-one-out ─────────────────────────────────────────────────────────
    loo_rows = []
    print("Leave-one-out ablation:")
    print(f"  {'Dropped feature':<30} {'IC':>8} {'IC drop':>10}")
    print(f"  {'─'*50}")

    for feat in FEATURE_COLS:
        remaining = [f for f in FEATURE_COLS if f != feat]
        loo_ic    = run_with_features(train, test, remaining)
        drop      = baseline_ic - loo_ic
        print(f"  {'drop ' + feat:<30} {loo_ic:>8.4f} {drop:>+10.4f}")
        loo_rows.append({
            "dropped"    : feat,
            "ic"         : loo_ic,
            "ic_drop"    : drop,
            "ic_drop_pct": drop / baseline_ic * 100,
        })

    loo_df = pd.DataFrame(loo_rows).sort_values("ic_drop", ascending=False)

    # ── Standalone: each feature alone ───────────────────────────────────────
    solo_rows = []
    print(f"\nStandalone IC (each feature alone):")
    print(f"  {'Feature':<30} {'IC':>8}")
    print(f"  {'─'*40}")

    for feat in FEATURE_COLS:
        solo_ic = run_with_features(train, test, [feat])
        print(f"  {feat:<30} {solo_ic:>8.4f}")
        solo_rows.append({
            "feature": feat,
            "solo_ic": solo_ic,
        })

    solo_df = pd.DataFrame(solo_rows).sort_values("solo_ic", ascending=False)

    return loo_df, solo_df


def plot_ablation(loo_df: pd.DataFrame,
                  solo_df: pd.DataFrame,
                  baseline_ic: float) -> None:
    BLUE  = "#2563EB"
    RED   = "#DC2626"
    GREEN = "#16A34A"
    GRAY  = "#6B7280"
    BG    = "#F9FAFB"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # ── Left: IC drop when feature removed ───────────────────────────────────
    colors = [RED if d > 0 else GREEN for d in loo_df["ic_drop"]]
    bars = ax1.barh(loo_df["dropped"], loo_df["ic_drop"],
                    color=colors, alpha=0.8, zorder=3)
    ax1.axvline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)

    for bar, val in zip(bars, loo_df["ic_drop"]):
        ax1.text(val + (0.001 if val >= 0 else -0.001),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.4f}",
                 va="center",
                 ha="left" if val >= 0 else "right",
                 fontsize=8, color="#1F2937")

    ax1.set_xlabel("IC drop when feature removed\n(positive = feature helped)", fontsize=10)
    ax1.set_title("Leave-One-Out Feature Importance\n"
                  f"Baseline IC = {baseline_ic:.4f}",
                  fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.25, linewidth=0.6, axis="x", zorder=0)
    ax1.tick_params(labelsize=9)

    # ── Right: standalone IC ─────────────────────────────────────────────────
    colors2 = [BLUE if ic > 0 else RED for ic in solo_df["solo_ic"]]
    bars2 = ax2.barh(solo_df["feature"], solo_df["solo_ic"],
                     color=colors2, alpha=0.8, zorder=3)
    ax2.axvline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axvline(baseline_ic, color=GRAY, linewidth=1.0,
                linestyle=":", alpha=0.6, label=f"Baseline IC={baseline_ic:.3f}")

    for bar, val in zip(bars2, solo_df["solo_ic"]):
        ax2.text(max(val, 0) + 0.002,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}",
                 va="center", ha="left",
                 fontsize=8, color="#1F2937")

    ax2.set_xlabel("Standalone IC (feature alone)", fontsize=10)
    ax2.set_title("Standalone Feature IC", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, linewidth=0.6, axis="x", zorder=0)
    ax2.tick_params(labelsize=9)

    plt.suptitle("MicrostructureLab — Feature Ablation Study",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = FIGURES / "feature_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nFeature ablation chart saved to {out}")


def run(symbol: str = SYMBOL) -> None:
    print("=" * 60)
    print(f"FEATURE ABLATION — {symbol}")
    print("=" * 60)

    feat_df     = load_clean_data(symbol)
    train, test = time_split(feat_df, train_frac=0.7)
    baseline_ic = run_with_features(train, test, FEATURE_COLS)

    loo_df, solo_df = run_ablation(symbol)

    print(f"\n{'='*60}")
    print("SUMMARY — features ranked by importance (IC drop)")
    print(f"{'='*60}")
    print(loo_df[["dropped", "ic", "ic_drop", "ic_drop_pct"]].to_string(index=False))

    print(f"\nBaseline IC: {baseline_ic:.4f}")
    top = loo_df.iloc[0]
    print(f"Most important feature: {top['dropped']} "
          f"(IC drops by {top['ic_drop']:+.4f} = "
          f"{top['ic_drop_pct']:.1f}% when removed)")

    plot_ablation(loo_df, solo_df, baseline_ic)


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else SYMBOL
    run(symbol)