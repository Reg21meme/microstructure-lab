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

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Main chart: cumulative PnL comparison ─────────────────────────────────
    ax_main = fig.add_subplot(gs[0, :])

    if not naive_pnl.empty:
        t_naive = (naive_pnl["fill_ts_ns"] -
                   naive_pnl["fill_ts_ns"].iloc[0]) / 1e9
        ax_main.plot(t_naive, naive_pnl["cum_pnl"],
                     color="steelblue", linewidth=1.5,
                     label="Naive (no fees, no latency)")

    if not real_pnl.empty:
        t_real = (real_pnl["fill_ts_ns"] -
                  real_pnl["fill_ts_ns"].iloc[0]) / 1e9
        ax_main.plot(t_real, real_pnl["cum_pnl"],
                     color="crimson", linewidth=1.5,
                     label="Realistic (10ms latency + fees)")

    ax_main.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_main.set_xlabel("Time (seconds into test period)")
    ax_main.set_ylabel("Cumulative PnL ($)")
    ax_main.set_title("Naive vs Realistic Cumulative PnL\n"
                      "Gap = latency cost + fee drag + missed fills",
                      fontsize=12)
    ax_main.legend(fontsize=10)
    ax_main.grid(alpha=0.3)

    # ── Fee drag over time ────────────────────────────────────────────────────
    ax_fees = fig.add_subplot(gs[1, 0])
    if not real_pnl.empty:
        ax_fees.plot(t_real, real_pnl["cum_fees"],
                     color="darkorange", linewidth=1.2)
        ax_fees.set_title("Cumulative Fee Drag (Realistic)")
        ax_fees.set_xlabel("Time (seconds)")
        ax_fees.set_ylabel("Fees paid ($)")
        ax_fees.grid(alpha=0.3)

    # ── Fill price distribution ───────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 1])
    if not naive_fills.empty:
        ax_dist.hist(naive_fills["fill_price"], bins=30,
                     alpha=0.6, color="steelblue",
                     label="Naive", density=True)
    if not real_fills.empty:
        ax_dist.hist(real_fills["fill_price"], bins=30,
                     alpha=0.6, color="crimson",
                     label="Realistic", density=True)
    ax_dist.set_title("Fill Price Distribution")
    ax_dist.set_xlabel("Fill price ($)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(fontsize=9)
    ax_dist.grid(alpha=0.3)

    # ── PnL decomposition bar chart ───────────────────────────────────────────
    ax_decomp = fig.add_subplot(gs[2, 0])
    naive_sum = compute_summary(naive_fills,  "Naive")
    real_sum  = compute_summary(real_fills,   "Realistic")

    categories = ["Gross PnL", "Fee Drag", "Net PnL"]
    naive_vals = [
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
    width = 0.35
    ax_decomp.bar(x - width/2, naive_vals, width,
                  color="steelblue", alpha=0.7, label="Naive")
    ax_decomp.bar(x + width/2, real_vals,  width,
                  color="crimson",   alpha=0.7, label="Realistic")
    ax_decomp.axhline(0, color="black", linewidth=0.8)
    ax_decomp.set_xticks(x)
    ax_decomp.set_xticklabels(categories)
    ax_decomp.set_ylabel("$ Amount")
    ax_decomp.set_title("PnL Decomposition")
    ax_decomp.legend(fontsize=9)
    ax_decomp.grid(alpha=0.3, axis="y")

    # ── Summary text ──────────────────────────────────────────────────────────
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis("off")

    gap = naive_sum.get("total_pnl", 0) - real_sum.get("total_pnl", 0)
    summary_text = (
        f"SUMMARY\n"
        f"{'─'*35}\n"
        f"{'':20s} {'Naive':>8s} {'Real':>8s}\n"
        f"{'Gross PnL':20s} "
        f"${naive_sum.get('gross_pnl', 0):>7.2f} "
        f"${real_sum.get('gross_pnl', 0):>7.2f}\n"
        f"{'Fee drag':20s} "
        f"${-naive_sum.get('total_fees', 0):>7.2f} "
        f"${-real_sum.get('total_fees', 0):>7.2f}\n"
        f"{'Net PnL':20s} "
        f"${naive_sum.get('total_pnl', 0):>7.2f} "
        f"${real_sum.get('total_pnl', 0):>7.2f}\n"
        f"{'Fills':20s} "
        f"{naive_sum.get('n_fills', 0):>8d} "
        f"{real_sum.get('n_fills', 0):>8d}\n"
        f"{'Max drawdown':20s} "
        f"${naive_sum.get('max_dd', 0):>7.2f} "
        f"${real_sum.get('max_dd', 0):>7.2f}\n"
        f"{'─'*35}\n"
        f"PnL gap: ${gap:.2f}\n"
        f"(naive overstates PnL by "
        f"{abs(gap)/max(abs(naive_sum.get('total_pnl',1)),0.01)*100:.0f}%)"
    )
    ax_text.text(0.05, 0.95, summary_text,
                 transform=ax_text.transAxes,
                 fontsize=9, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("MicrostructureLab — Execution Simulation Results",
                 fontsize=13, fontweight="bold")

    out = FIGURES / "headline_pnl_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
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