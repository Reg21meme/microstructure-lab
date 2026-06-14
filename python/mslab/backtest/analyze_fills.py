"""
analyze_fills.py
Produces the headline chart: naive vs realistic PnL over time.
Also computes PnL decomposition and summary statistics.

Usage:
    python3 -m mslab.backtest.analyze_fills
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT    = pathlib.Path(__file__).resolve().parents[3]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_fills(symbol: str = "BTCUSDT") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load naive and realistic fill Parquets."""
    results_dir = ROOT / "data" / "results"
    naive_path  = results_dir / f"{symbol}_fills_naive.parquet"
    real_path   = results_dir / f"{symbol}_fills_realistic.parquet"

    if not naive_path.exists() or not real_path.exists():
        print("Fill files not found — run run_cpp_sim.py first")
        return pd.DataFrame(), pd.DataFrame()

    naive = pd.read_parquet(naive_path)
    real  = pd.read_parquet(real_path)
    return naive, real


def compute_pnl_series(fills: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative PnL from fills using position-based method.

    Tracks position and computes realized PnL on each position reduction.
    This correctly handles partial fills and multiple round trips.
    """
    if fills.empty:
        return pd.DataFrame(columns=["fill_ts_ns", "cum_pnl", "cum_fees"])

    df = fills.copy().sort_values("fill_ts_ns").reset_index(drop=True)

    position    = 0.0
    avg_entry   = 0.0
    realized    = 0.0
    cum_fees    = 0.0
    pnl_records = []

    for _, row in df.iterrows():
        size  = row["fill_size"]
        price = row["fill_price"]
        fee   = row["fee"]
        is_buy = row["side"] == "buy"
        signed_size = size if is_buy else -size

        # Update realized PnL on position reduction
        if position > 0 and not is_buy:
            # Closing long
            closed = min(size, position)
            realized += closed * (price - avg_entry)
        elif position < 0 and is_buy:
            # Closing short
            closed = min(size, abs(position))
            realized += closed * (avg_entry - price)

        # Update average entry on position increase
        new_position = position + signed_size
        if abs(new_position) > abs(position) or \
           (position >= 0 and is_buy) or \
           (position <= 0 and not is_buy):
            if abs(new_position) > 1e-10:
                old_value = abs(position) * avg_entry
                new_value = size * price
                avg_entry = (old_value + new_value) / abs(new_position) \
                            if abs(new_position) > 1e-10 else price

        position  = new_position
        cum_fees += fee
        net_pnl   = realized - cum_fees

        pnl_records.append({
            "fill_ts_ns": row["fill_ts_ns"],
            "cum_pnl"   : net_pnl,
            "cum_fees"  : cum_fees,
            "realized"  : realized,
        })

    return pd.DataFrame(pnl_records)

def compute_summary(fills: pd.DataFrame,
                    label: str) -> dict:
    """Compute summary statistics from fills."""
    if fills.empty:
        return {"label": label}

    pnl_df     = compute_pnl_series(fills)
    total_pnl  = pnl_df["cum_pnl"].iloc[-1]
    total_fees = pnl_df["cum_fees"].iloc[-1]
    n_fills    = len(fills)
    n_buys     = (fills["side"] == "buy").sum()
    n_sells    = (fills["side"] == "sell").sum()
    n_maker    = fills["is_maker"].sum()
    n_taker    = (~fills["is_maker"]).sum()

    # Per-fill PnL for Sharpe
    per_fill_pnl = pnl_df["cum_pnl"].diff().dropna().values
    sharpe = (per_fill_pnl.mean() / per_fill_pnl.std() * np.sqrt(len(per_fill_pnl))
              if per_fill_pnl.std() > 0 else np.nan)

    # Max drawdown
    cum = pnl_df["cum_pnl"].values
    running_max = np.maximum.accumulate(cum)
    drawdowns   = cum - running_max
    max_dd      = drawdowns.min()

    return {
        "label"      : label,
        "total_pnl"  : total_pnl,
        "total_fees" : total_fees,
        "gross_pnl"  : total_pnl + total_fees,
        "n_fills"    : n_fills,
        "n_buys"     : n_buys,
        "n_sells"    : n_sells,
        "n_maker"    : n_maker,
        "n_taker"    : n_taker,
        "sharpe"     : sharpe,
        "max_dd"     : max_dd,
    }


def plot_headline_chart(naive_fills: pd.DataFrame,
                        real_fills:  pd.DataFrame) -> None:
    """
    The headline chart: naive vs realistic cumulative PnL over time.
    This is the README hero image.
    """
    naive_pnl = compute_pnl_series(naive_fills)
    real_pnl  = compute_pnl_series(real_fills)
    naive_sum = compute_summary(naive_fills, "Naive")
    real_sum  = compute_summary(real_fills,  "Realistic")

    # ── Style ─────────────────────────────────────────────────────────────────
    BLUE   = "#2563EB"
    RED    = "#DC2626"
    GRAY   = "#6B7280"
    BG     = "#F9FAFB"

    fig, (ax_main, ax_decomp) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.patch.set_facecolor(BG)
    for ax in (ax_main, ax_decomp):
        ax.set_facecolor(BG)

    # ── Top panel: cumulative PnL ─────────────────────────────────────────────
    if not naive_pnl.empty:
        t_naive = (naive_pnl["fill_ts_ns"] -
                   naive_pnl["fill_ts_ns"].iloc[0]) / 1e9
        ax_main.plot(t_naive, naive_pnl["cum_pnl"],
                     color=BLUE, linewidth=2.0,
                     label="Naive  (no fees · no latency · no queue)")

    if not real_pnl.empty:
        t_real = (real_pnl["fill_ts_ns"] -
                  real_pnl["fill_ts_ns"].iloc[0]) / 1e9
        ax_main.plot(t_real, real_pnl["cum_pnl"],
                     color=RED, linewidth=2.0,
                     label="Realistic  (10ms latency · VIP0 fees · queue model)")

    ax_main.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate the gap at the end of the shorter series
    if not naive_pnl.empty and not real_pnl.empty:
        # Find where realistic ends
        x_ann  = float(t_real.iloc[-1])
        y_real = float(real_pnl["cum_pnl"].iloc[-1])
        y_naive_at_end = float(naive_pnl["cum_pnl"].iloc[
            min(len(naive_pnl) - 1,
                int(x_ann / (t_naive.iloc[-1] / len(t_naive))))
        ])
        gap = naive_sum["total_pnl"] - real_sum["total_pnl"]

        ax_main.annotate(
            f"Gap = ${gap:.2f}\n(fees + latency\n+ queue rejection)",
            xy=(x_ann * 0.6, (y_real + y_naive_at_end) / 2),
            xytext=(x_ann * 0.72, 15),
            fontsize=9,
            color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRAY, alpha=0.8),
        )

        # Shade the gap region
        min_len = min(len(naive_pnl), len(real_pnl))
        t_shared     = t_naive.iloc[:min_len].values
        naive_shared = naive_pnl["cum_pnl"].iloc[:min_len].values
        real_shared  = real_pnl["cum_pnl"].iloc[:min_len].values
        ax_main.fill_between(t_shared, real_shared, naive_shared,
                             alpha=0.08, color=GRAY, label="Gap (shaded)")

    ax_main.set_xlabel("Time into test period (seconds)", fontsize=11)
    ax_main.set_ylabel("Cumulative PnL ($)", fontsize=11)
    ax_main.set_title(
        "Naive vs Realistic Cumulative PnL",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax_main.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax_main.grid(alpha=0.25, linewidth=0.6)
    ax_main.tick_params(labelsize=10)

    # ── Bottom panel: PnL decomposition bar chart ─────────────────────────────
    categories  = ["Gross PnL", "Fee Drag", "Net PnL"]
    naive_vals  = [
        naive_sum.get("gross_pnl", 0),
        -naive_sum.get("total_fees", 0),
        naive_sum.get("total_pnl", 0),
    ]
    real_vals = [
        real_sum.get("gross_pnl", 0),
        -real_sum.get("total_fees", 0),
        real_sum.get("total_pnl", 0),
    ]

    x     = np.arange(len(categories))
    width = 0.32

    bars_n = ax_decomp.bar(x - width / 2, naive_vals, width,
                            color=BLUE, alpha=0.75, label="Naive",
                            zorder=3)
    bars_r = ax_decomp.bar(x + width / 2, real_vals, width,
                            color=RED, alpha=0.75, label="Realistic",
                            zorder=3)

    # Value labels on bars
    for bar in list(bars_n) + list(bars_r):
        h = bar.get_height()
        if abs(h) > 0.5:
            ax_decomp.text(
                bar.get_x() + bar.get_width() / 2,
                h + (1.5 if h >= 0 else -4.5),
                f"${h:.1f}",
                ha="center",
                va="top" if h < 0 else "bottom",
                fontsize=8, color="#1F2937",
            )

    ax_decomp.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_decomp.set_xticks(x)
    ax_decomp.set_xticklabels(categories, fontsize=11)
    ax_decomp.set_ylabel("$ Amount", fontsize=11)
    ax_decomp.set_title("PnL Decomposition", fontsize=12,
                         fontweight="bold", pad=8)
    ax_decomp.legend(fontsize=9, framealpha=0.9)
    ax_decomp.grid(alpha=0.25, linewidth=0.6, axis="y", zorder=0)
    ax_decomp.tick_params(labelsize=10)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        f"BTCUSDT · 30-min sample · "
        f"Naive: +${naive_sum['total_pnl']:.2f}  "
        f"Realistic: ${real_sum['total_pnl']:.2f}  "
        f"Gap: ${naive_sum['total_pnl'] - real_sum['total_pnl']:.2f}  "
        f"· Fills: {naive_sum['n_fills']} naive / {real_sum['n_fills']} realistic",
        ha="center", fontsize=8, color=GRAY,
    )

    plt.suptitle(
        "MicrostructureLab — Execution Simulation",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = FIGURES / "headline_pnl_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Headline chart saved to {out}")

def run(symbol: str = "BTCUSDT") -> None:
    print("=" * 60)
    print("FILL ANALYSIS & HEADLINE CHART")
    print("=" * 60)

    naive_fills, real_fills = load_fills(symbol)

    if naive_fills.empty and real_fills.empty:
        print("No fill data found. Run run_cpp_sim.py first.")
        return

    # Summary statistics
    for fills, label in [(naive_fills,  "Naive"),
                         (real_fills,   "Realistic")]:
        s = compute_summary(fills, label)
        print(f"\n{label}:")
        print(f"  Total PnL   : ${s.get('total_pnl',  0):.2f}")
        print(f"  Gross PnL   : ${s.get('gross_pnl',  0):.2f}")
        print(f"  Fee drag    : ${s.get('total_fees',  0):.2f}")
        print(f"  Fills       : {s.get('n_fills',      0)}")
        print(f"  Buys/Sells  : {s.get('n_buys', 0)}/{s.get('n_sells', 0)}")
        print(f"  Maker/Taker : {s.get('n_maker', 0)}/{s.get('n_taker', 0)}")
        print(f"  Sharpe      : {s.get('sharpe', float('nan')):.4f}")
        print(f"  Max DD      : ${s.get('max_dd', 0):.2f}")

    # Headline chart
    plot_headline_chart(naive_fills, real_fills)


if __name__ == "__main__":
    run()