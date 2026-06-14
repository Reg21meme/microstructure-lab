"""
fee_sweep.py
Fee sensitivity sweep — vary taker fee from 0 to 8bps and plot
net PnL vs fee level. Finds the breakeven fee rate.

Usage:
    python3 -m mslab.backtest.fee_sweep
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp" / "build"))

from mslab.configs.fees import FeeConfig
from mslab.backtest.queue_model import load_queue_config
from mslab.backtest.run_cpp_sim import run_simulation

FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Taker fee levels to sweep (in fraction, not bps)
# 0.0 = 0bps, 0.0008 = 8bps
TAKER_FEES = [0.0, 0.0001, 0.0002, 0.0003, 0.0004,
              0.0005, 0.0006, 0.0007, 0.0008]

# Maker rebate = half the taker fee (typical exchange structure)
def maker_from_taker(taker: float) -> float:
    return -taker / 4.0


def run_sweep() -> pd.DataFrame:
    queue_cfg = load_queue_config("base")
    rows      = []

    for taker in TAKER_FEES:
        maker   = maker_from_taker(taker)
        bps     = taker * 10_000
        fee_cfg = FeeConfig(
            scenario    = f"{bps:.1f}bps",
            maker_fee   = maker,
            taker_fee   = taker,
            description = f"Taker={bps:.1f}bps, Maker={maker*10000:.1f}bps",
        )
        print(f"\n{'─'*50}")
        print(f"Fee level: taker={bps:.1f}bps  maker={maker*10000:.1f}bps")

        result = run_simulation(
            latency_ms = 10.0,
            fee_cfg    = fee_cfg,
            queue_cfg  = queue_cfg,
            label      = f"{bps:.1f}bps",
        )

        rows.append({
            "taker_bps" : bps,
            "maker_bps" : maker * 10_000,
            "taker_fee" : taker,
            "maker_fee" : maker,
            "net_pnl"   : result["net_pnl"],
            "gross_pnl" : result["realized_pnl"],
            "fee_drag"  : result["fee_drag"],
            "n_fills"   : result["n_fills"],
        })

    return pd.DataFrame(rows)


def find_breakeven(df: pd.DataFrame) -> float | None:
    """Linear interpolation to find breakeven taker fee in bps."""
    for i in range(len(df) - 1):
        y0, y1 = df["net_pnl"].iloc[i], df["net_pnl"].iloc[i + 1]
        x0, x1 = df["taker_bps"].iloc[i], df["taker_bps"].iloc[i + 1]
        if y0 >= 0 >= y1:
            # Linear interpolation
            return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    return None


def plot_sweep(df: pd.DataFrame, breakeven_bps: float | None) -> None:
    BLUE = "#2563EB"
    RED  = "#DC2626"
    GRAY = "#6B7280"
    BG   = "#F9FAFB"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # ── Left: Net PnL vs taker fee ────────────────────────────────────────────
    ax1.plot(df["taker_bps"], df["net_pnl"],
             color=RED, linewidth=2.0, marker="o", markersize=7, zorder=3)
    ax1.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)

    if breakeven_bps is not None:
        ax1.axvline(breakeven_bps, color=BLUE, linewidth=1.2,
                    linestyle="--", alpha=0.8)
        ax1.annotate(
            f"Breakeven\n≈ {breakeven_bps:.2f} bps",
            xy=(breakeven_bps, 0),
            xytext=(breakeven_bps + 0.3, df["net_pnl"].max() * 0.4),
            fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=BLUE, alpha=0.8),
        )

    ax1.set_xlabel("Taker fee (bps)", fontsize=11)
    ax1.set_ylabel("Net PnL ($)", fontsize=11)
    ax1.set_title("Net PnL vs Taker Fee", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.tick_params(labelsize=10)

    # ── Right: Fee drag vs taker fee ──────────────────────────────────────────
    ax2.plot(df["taker_bps"], df["fee_drag"],
             color=BLUE, linewidth=2.0, marker="o", markersize=7, zorder=3)
    ax2.plot(df["taker_bps"], df["gross_pnl"],
             color=GRAY, linewidth=1.5, marker="s", markersize=5,
             linestyle="--", zorder=3, label="Gross PnL")

    if breakeven_bps is not None:
        ax2.axvline(breakeven_bps, color=RED, linewidth=1.0,
                    linestyle="--", alpha=0.6)

    ax2.set_xlabel("Taker fee (bps)", fontsize=11)
    ax2.set_ylabel("$ Amount", fontsize=11)
    ax2.set_title("Fee Drag vs Gross PnL", fontsize=12, fontweight="bold")
    ax2.legend(["Fee drag", "Gross PnL"], fontsize=9)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.tick_params(labelsize=10)

    plt.suptitle("MicrostructureLab — Fee Sensitivity Sweep",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = FIGURES / "fee_sensitivity_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nFee sweep chart saved to {out}")


def run() -> None:
    print("=" * 60)
    print("FEE SENSITIVITY SWEEP")
    print("=" * 60)

    df = run_sweep()

    print("\n" + "=" * 60)
    print("FEE SWEEP RESULTS")
    print("=" * 60)
    print(df[["taker_bps", "maker_bps", "net_pnl",
              "fee_drag", "n_fills"]].to_string(index=False))

    breakeven_bps = find_breakeven(df)
    if breakeven_bps is not None:
        print(f"\nBreakeven taker fee: {breakeven_bps:.2f} bps")
        print(f"(Binance VIP0 is 4.0 bps — strategy needs "
              f"{'lower' if breakeven_bps < 4.0 else 'higher'} fees to be profitable)")
    else:
        print("\nNo breakeven found in swept range.")

    plot_sweep(df, breakeven_bps)


if __name__ == "__main__":
    run()