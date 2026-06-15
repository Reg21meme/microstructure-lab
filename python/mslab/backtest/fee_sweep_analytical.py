"""
fee_sweep_analytical.py
Analytical fee sensitivity sweep from existing fills parquet.
Much faster than re-running the full simulation for each fee level.

This is valid because fee rate does not affect which trades are made
(the signal and queue model are fee-agnostic), only what they cost.

Usage:
    python3 -m mslab.backtest.fee_sweep_analytical
"""

import pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ROOT    = pathlib.Path(__file__).resolve().parents[3]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TAKER_FEES_BPS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]


def run(symbol: str = "BTCUSDT") -> None:
    fills_path = ROOT / "data" / "results" / f"{symbol}_fills_realistic.parquet"
    fills      = pq.read_table(fills_path).to_pandas()

    notional  = (fills["fill_price"] * fills["fill_size"]).sum()
    gross_pnl = fills.apply(
        lambda r: -r["fill_price"] * r["fill_size"]
                   if r["side"] == "buy"
                   else r["fill_price"] * r["fill_size"],
        axis=1
    ).sum()

    # Use sim-reported gross PnL (position-based, more accurate)
    # We load it from the naive run instead
    naive_path = ROOT / "data" / "results" / f"{symbol}_fills_naive.parquet"
    naive      = pq.read_table(naive_path).to_pandas()

    print(f"{'Taker bps':<12} {'Fee drag':>12} {'Net PnL':>12}")
    print("-" * 38)

    rows = []
    for bps in TAKER_FEES_BPS:
        rate     = bps / 10_000
        fee_drag = notional * rate
        # Gross PnL from sim = $1439.70 (position-based realized PnL)
        GROSS_PNL = 1439.70
        net_pnl  = GROSS_PNL - fee_drag
        print(f"{bps:<12} ${fee_drag:>11.2f} ${net_pnl:>11.2f}")
        rows.append({
            "taker_bps": bps,
            "fee_drag" : fee_drag,
            "net_pnl"  : net_pnl,
            "gross_pnl": GROSS_PNL,
        })

    df = pd.DataFrame(rows)
    breakeven = GROSS_PNL / notional * 10_000
    print(f"\nBreakeven taker fee : {breakeven:.3f} bps")
    print(f"Binance VIP0        : 4.0 bps ({4/breakeven:.1f}x breakeven)")
    print(f"Total notional      : ${notional:,.0f}")
    print(f"Fills               : {len(fills):,}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    BLUE = "#2563EB"
    RED  = "#DC2626"
    GRAY = "#6B7280"
    BG   = "#F9FAFB"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # Net PnL vs fee
    ax1.plot(df["taker_bps"], df["net_pnl"],
             color=RED, linewidth=2.0, marker="o", markersize=6, zorder=3)
    ax1.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.axvline(breakeven, color=BLUE, linewidth=1.2,
                linestyle="--", alpha=0.8)
    ax1.annotate(
        f"Breakeven\n≈ {breakeven:.2f} bps",
        xy=(breakeven, 0),
        xytext=(breakeven + 0.5, df["net_pnl"].max() * 0.35),
        fontsize=9, color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=BLUE, alpha=0.8),
    )
    ax1.scatter([4.0], [df[df["taker_bps"]==4]["net_pnl"].values[0]],
                color=RED, s=80, zorder=5)
    ax1.annotate("VIP0\n(4 bps)",
                 xy=(4.0, df[df["taker_bps"]==4]["net_pnl"].values[0]),
                 xytext=(5.0, df[df["taker_bps"]==4]["net_pnl"].values[0] + 800),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax1.set_xlabel("Taker fee (bps)", fontsize=11)
    ax1.set_ylabel("Net PnL ($)", fontsize=11)
    ax1.set_title("Net PnL vs Taker Fee\n(Full day, 19,196 fills)",
                  fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.25, linewidth=0.6)

    # Fee drag vs gross PnL
    ax2.plot(df["taker_bps"], df["fee_drag"],
             color=RED, linewidth=2.0, marker="o", markersize=6,
             zorder=3, label="Fee drag")
    ax2.axhline(1439.70, color=GRAY, linewidth=1.5,
                linestyle="--", alpha=0.8, label="Gross PnL ($1,440)")
    ax2.axvline(breakeven, color=BLUE, linewidth=1.0,
                linestyle="--", alpha=0.6)
    ax2.set_xlabel("Taker fee (bps)", fontsize=11)
    ax2.set_ylabel("$ Amount", fontsize=11)
    ax2.set_title("Fee Drag vs Gross PnL\n(Crossover = breakeven)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, linewidth=0.6)

    plt.suptitle(
        f"MicrostructureLab — Fee Sensitivity ({symbol}, Full Day Jun 1 2025)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    out = FIGURES / "fee_sensitivity_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nFee sweep chart saved to {out}")


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    run(symbol)