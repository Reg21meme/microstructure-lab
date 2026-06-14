"""
threshold_sweep.py
Signal confidence threshold sweep.

Varies SIGNAL_THRESH from $0.05 to $1.00 and records:
  - n_fills, fill_rate
  - net_pnl, gross_pnl, fee_drag
  - mean markout at 1s (adverse selection proxy)

Higher threshold = fewer but higher-conviction trades.
Question: does selectivity improve net PnL or just reduce turnover?

Usage:
    python3 -m mslab.backtest.threshold_sweep
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp" / "build"))

from mslab.configs.fees import load_fees
from mslab.backtest.queue_model import load_queue_config
from mslab.backtest.run_cpp_sim import run_simulation, LATENCY_MS

FIGURES    = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]


def run_sweep() -> pd.DataFrame:
    fee_cfg   = load_fees("binance_vip0")
    queue_cfg = load_queue_config("base")
    rows      = []

    # Monkey-patch SIGNAL_THRESH in run_cpp_sim for each iteration
    import mslab.backtest.run_cpp_sim as sim_module

    for thresh in THRESHOLDS:
        print(f"\n{'─'*50}")
        print(f"Signal threshold: ${thresh:.2f}")

        # Patch the module-level constant
        sim_module.SIGNAL_THRESH = thresh

        result = run_simulation(
            latency_ms = LATENCY_MS,
            fee_cfg    = fee_cfg,
            queue_cfg  = queue_cfg,
            label      = f"thresh=${thresh:.2f}",
        )

        # Extract mean 1s markout if available
        mo_summary = result.get("markout_summary", pd.DataFrame())
        if not mo_summary.empty and "1s" in mo_summary["horizon"].values:
            mo_1s = float(mo_summary[mo_summary["horizon"] == "1s"]["mean_$"].iloc[0])
        else:
            mo_1s = np.nan

        rows.append({
            "threshold"  : thresh,
            "net_pnl"    : result["net_pnl"],
            "gross_pnl"  : result["realized_pnl"],
            "fee_drag"   : result["fee_drag"],
            "n_fills"    : result["n_fills"],
            "n_orders"   : result["n_orders"],
            "n_signals"  : result["n_signals"],
            "fill_rate"  : result["n_fills"] / result["n_orders"]
                           if result["n_orders"] > 0 else 0.0,
            "markout_1s" : mo_1s,
        })

    # Restore default
    sim_module.SIGNAL_THRESH = 0.10
    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame) -> None:
    BLUE = "#2563EB"
    RED  = "#DC2626"
    GREEN = "#16A34A"
    GRAY = "#6B7280"
    BG   = "#F9FAFB"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(BG)

    ax1, ax2, ax3, ax4 = axes.flat
    x = df["threshold"].values

    # ── Net PnL vs threshold ──────────────────────────────────────────────────
    ax1.plot(x, df["net_pnl"], color=RED, linewidth=2.0,
             marker="o", markersize=7, zorder=3)
    ax1.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.set_xlabel("Signal threshold ($)", fontsize=10)
    ax1.set_ylabel("Net PnL ($)", fontsize=10)
    ax1.set_title("Net PnL vs Threshold", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.25, linewidth=0.6)

    # ── Number of fills vs threshold ──────────────────────────────────────────
    ax2.plot(x, df["n_fills"], color=BLUE, linewidth=2.0,
             marker="o", markersize=7, zorder=3)
    ax2.set_xlabel("Signal threshold ($)", fontsize=10)
    ax2.set_ylabel("Number of fills", fontsize=10)
    ax2.set_title("Fill Count vs Threshold", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.25, linewidth=0.6)

    # ── Fee drag vs threshold ─────────────────────────────────────────────────
    ax3.plot(x, df["fee_drag"], color=RED, linewidth=2.0,
             marker="o", markersize=7, zorder=3, label="Fee drag")
    ax3.plot(x, df["gross_pnl"], color=GRAY, linewidth=1.5,
             marker="s", markersize=5, linestyle="--",
             zorder=3, label="Gross PnL")
    ax3.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.4)
    ax3.set_xlabel("Signal threshold ($)", fontsize=10)
    ax3.set_ylabel("$ Amount", fontsize=10)
    ax3.set_title("Fee Drag vs Gross PnL", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.25, linewidth=0.6)

    # ── Mean 1s markout vs threshold ──────────────────────────────────────────
    ax4.plot(x, df["markout_1s"], color=GREEN, linewidth=2.0,
             marker="o", markersize=7, zorder=3)
    ax4.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax4.set_xlabel("Signal threshold ($)", fontsize=10)
    ax4.set_ylabel("Mean markout at 1s ($)", fontsize=10)
    ax4.set_title("Adverse Selection vs Threshold", fontsize=11,
                  fontweight="bold")
    ax4.grid(alpha=0.25, linewidth=0.6)

    plt.suptitle("MicrostructureLab — Confidence Threshold Sweep",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = FIGURES / "threshold_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nThreshold sweep chart saved to {out}")


def run() -> None:
    print("=" * 60)
    print("CONFIDENCE THRESHOLD SWEEP")
    print("=" * 60)

    df = run_sweep()

    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 60)
    print(df[["threshold", "net_pnl", "gross_pnl", "fee_drag",
              "n_fills", "markout_1s"]].to_string(index=False))

    # Find threshold that maximizes net PnL
    best_idx   = df["net_pnl"].idxmax()
    best_row   = df.iloc[best_idx]
    print(f"\nBest threshold: ${best_row['threshold']:.2f} "
          f"→ net PnL = ${best_row['net_pnl']:.2f} "
          f"({best_row['n_fills']} fills)")

    plot_sweep(df)


if __name__ == "__main__":
    run()