"""
latency_sweep.py
Latency stress test — run the sim at multiple latency levels and
show how net PnL, fill rate, and fee drag degrade.

Usage:
    python3 -m mslab.backtest.latency_sweep
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp" / "build"))

from mslab.configs.fees import load_fees
from mslab.configs.latency import load_latency, list_scenarios
from mslab.backtest.queue_model import load_queue_config
from mslab.backtest.run_cpp_sim import run_simulation

SCENARIOS    = ["zero", "fast", "medium", "slow", "jittery"]
FEE_SCENARIO = "binance_vip0"
QUEUE_SCENARIO = "base"
FIGURES      = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def run_sweep() -> pd.DataFrame:
    """Run sim at each latency scenario and collect results."""
    fee_cfg   = load_fees(FEE_SCENARIO)
    queue_cfg = load_queue_config(QUEUE_SCENARIO)
    rows      = []

    for scenario in SCENARIOS:
        lat_cfg = load_latency(scenario)
        print(f"\n{'─'*50}")
        print(f"Latency scenario: {scenario} "
              f"({lat_cfg.base_ms}ms base, {lat_cfg.jitter_ms}ms jitter)")

        result = run_simulation(
            latency_ms = lat_cfg.base_ms,
            fee_cfg    = fee_cfg,
            queue_cfg  = queue_cfg,
            label      = scenario,
        )

        rows.append({
            "scenario"   : scenario,
            "base_ms"    : lat_cfg.base_ms,
            "jitter_ms"  : lat_cfg.jitter_ms,
            "description": lat_cfg.description,
            "net_pnl"    : result["net_pnl"],
            "gross_pnl"  : result["realized_pnl"],
            "fee_drag"   : result["fee_drag"],
            "n_fills"    : result["n_fills"],
            "n_orders"   : result["n_orders"],
            "fill_rate"  : result["n_fills"] / result["n_orders"]
                           if result["n_orders"] > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame) -> None:
    """Plot net PnL and fill rate vs latency."""
    BLUE = "#2563EB"
    RED  = "#DC2626"
    GRAY = "#6B7280"
    BG   = "#F9FAFB"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    x     = df["base_ms"].values
    x_lab = df["scenario"].values

    # ── Left: Net PnL vs latency ──────────────────────────────────────────────
    ax1.plot(x, df["net_pnl"].values, color=RED,
             linewidth=2.0, marker="o", markersize=7, zorder=3)
    ax1.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)

    for i, (xi, yi, lab) in enumerate(zip(x, df["net_pnl"].values, x_lab)):
        ax1.annotate(f"{lab}\n${yi:.1f}",
                     xy=(xi, yi),
                     xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=8, color=GRAY)

    ax1.set_xlabel("Base latency (ms)", fontsize=11)
    ax1.set_ylabel("Net PnL ($)", fontsize=11)
    ax1.set_title("Net PnL vs Latency", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.tick_params(labelsize=10)

    # ── Right: Fill rate vs latency ───────────────────────────────────────────
    ax2.plot(x, df["fill_rate"].values * 100, color=BLUE,
             linewidth=2.0, marker="o", markersize=7, zorder=3)

    for i, (xi, yi, lab) in enumerate(zip(x, df["fill_rate"].values, x_lab)):
        ax2.annotate(f"{lab}\n{yi*100:.1f}%",
                     xy=(xi, yi * 100),
                     xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=8, color=GRAY)

    ax2.set_xlabel("Base latency (ms)", fontsize=11)
    ax2.set_ylabel("Fill rate (%)", fontsize=11)
    ax2.set_title("Fill Rate vs Latency", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.tick_params(labelsize=10)

    plt.suptitle("MicrostructureLab — Latency Stress Test",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = FIGURES / "latency_stress_test.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nLatency stress chart saved to {out}")


def run() -> None:
    print("=" * 60)
    print("LATENCY STRESS TEST")
    print("=" * 60)

    df = run_sweep()

    print("\n" + "=" * 60)
    print("LATENCY SWEEP RESULTS")
    print("=" * 60)
    print(df[["scenario", "base_ms", "net_pnl", "fee_drag",
              "n_fills", "fill_rate"]].to_string(index=False))

    # Breakeven latency
    if df["net_pnl"].iloc[0] > 0 and df["net_pnl"].iloc[-2] < 0:
        print(f"\nStrategy is profitable at {df['scenario'].iloc[0]} "
              f"({df['base_ms'].iloc[0]}ms) and unprofitable beyond that.")
    else:
        print(f"\nStrategy is unprofitable across all latency scenarios "
              f"(fee-dominated on this 30-min sample).")

    plot_sweep(df)
    

if __name__ == "__main__":
    run()