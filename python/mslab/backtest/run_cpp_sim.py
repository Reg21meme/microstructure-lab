"""
run_cpp_sim.py
Connects the Python signal to the C++ ExecutionSim.

Workflow:
  1. Load feature Parquet and normalized L2 updates
  2. Train ridge model on first 70% of data (training period)
  3. Replay L2 updates through C++ sim on last 30% (test period)
  4. At each 1-second snapshot, generate a signal from the model
  5. Submit orders to the sim based on signal direction and confidence
  6. Collect fills, compute PnL, compare naive vs realistic

Usage:
    python3 -m mslab.backtest.run_cpp_sim
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp" / "build"))

import mslab_bindings as mb
from mslab.models.train_baseline import (
    load_clean_data, time_split, train_ridge, FEATURE_COLS
)

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOL        = "BTCUSDT"
LATENCY_MS    = 10.0       # simulated order latency
TAKER_FEE     = 0.0004     # Binance taker fee
MAKER_FEE     = -0.0001    # Binance maker rebate
MAX_POSITION  = 1.0        # max 1 BTC position
MAX_DRAWDOWN  = 500.0      # kill switch at $500 loss
SIGNAL_THRESH = 0.10       # minimum predicted move to trigger order (dollars)
ORDER_SIZE    = 0.01       # order size in BTC per trade


def make_sim(latency_ms: float = LATENCY_MS,
             taker_fee: float  = TAKER_FEE,
             maker_fee: float  = MAKER_FEE) -> mb.ExecutionSim:
    """Create a configured ExecutionSim instance."""
    latency        = mb.LatencyModel(base_ms=latency_ms, jitter_ms=0.0)
    fees           = mb.FeeModel()
    fees.taker_fee = taker_fee
    fees.maker_fee = maker_fee
    limits         = mb.RiskLimits()
    limits.max_position = MAX_POSITION
    limits.max_drawdown = MAX_DRAWDOWN
    return mb.ExecutionSim(SYMBOL, latency, fees, limits)


def load_updates() -> pd.DataFrame:
    """Load normalized L2 updates sorted by sequence number."""
    path = ROOT / "data" / "normalized" / f"{SYMBOL}_updates.parquet"
    df   = pq.read_table(path).to_pandas()
    df   = df.sort_values("seq").reset_index(drop=True)
    return df


def load_snapshot_rows() -> pd.DataFrame:
    """Load snapshot rows."""
    path = ROOT / "data" / "normalized" / f"{SYMBOL}_snapshot.parquet"
    return pq.read_table(path).to_pandas()


def initialize_book(sim: mb.ExecutionSim,
                    snap_rows: pd.DataFrame) -> None:
    """Load snapshot into sim's internal book."""
    snapshot_seq = None
    for _, row in snap_rows.iterrows():
        is_bid = row["side"] == "bid"
        sim_book_update_snapshot(sim, row["price"], row["size"],
                                 is_bid, row["seq"])
        snapshot_seq = row["seq"]


def sim_book_update_snapshot(sim: mb.ExecutionSim,
                              price: float, size: float,
                              is_bid: bool, seq: int) -> None:
    """Apply a snapshot row as a book update at ts=0."""
    sim.on_book_update(price, size, is_bid, 0, seq, seq)


def generate_signal(model,
                    scaler,
                    features: pd.Series) -> float:
    """
    Generate a trading signal from the ridge model.

    Returns predicted future mid-price move in dollars.
    Positive = predict up, negative = predict down.
    """
    X = scaler.transform(features[FEATURE_COLS].values.reshape(1, -1))
    return float(model.predict(X)[0])


def run_simulation(latency_ms: float = LATENCY_MS,
                   taker_fee:  float = TAKER_FEE,
                   maker_fee:  float = MAKER_FEE,
                   label:      str   = "realistic") -> dict:
    """
    Run the full backtest simulation.

    Parameters
    ----------
    latency_ms : order latency in milliseconds
    taker_fee  : taker fee rate
    maker_fee  : maker rebate rate (negative = receive)
    label      : label for this run (e.g. 'naive', 'realistic')

    Returns
    -------
    dict with fills, pnl_series, and summary statistics
    """
    print(f"\nRunning simulation: {label}")
    print(f"  Latency={latency_ms}ms, taker_fee={taker_fee}, "
          f"maker_fee={maker_fee}")

    # ── Load and split data ───────────────────────────────────────────────────
    feat_df     = load_clean_data(SYMBOL)
    train, test = time_split(feat_df, train_frac=0.7)

    # ── Train model on training set only ─────────────────────────────────────
    ridge_results = train_ridge(train, test)
    model         = ridge_results["model"]
    scaler        = ridge_results["scaler"]

    print(f"  Model trained: test IC={ridge_results['ic_test']:.4f}")

    # ── Load L2 updates ───────────────────────────────────────────────────────
    upd_rows  = load_updates()
    snap_rows = load_snapshot_rows()

    # Find the timestamp cutoff for test period
    # test starts at 70% of feature rows
    test_start_ts = test["ts_local"].iloc[0]
    print(f"  Test period starts at ts={test_start_ts}")

    # ── Create and initialize sim ─────────────────────────────────────────────
    sim = make_sim(latency_ms, taker_fee, maker_fee)

    # Initialize book from snapshot
    snapshot_seq = int(snap_rows["seq"].iloc[0])
    for _, row in snap_rows.iterrows():
        is_bid = row["side"] == "bid"
        sim.on_book_update(row["price"], row["size"], is_bid,
                           0, snapshot_seq, snapshot_seq)

    # ── Replay updates and generate signals ───────────────────────────────────
    last_signal_ts  = -1
    signal_interval = 1_000  # generate signal every 1000ms = 1s
    n_signals       = 0
    n_orders        = 0
    in_test_period  = False

    # Build feature lookup by timestamp for signal generation
    feat_by_ts = feat_df.set_index("ts_local")

    for _, row in upd_rows.iterrows():
        ts_ms  = int(row["ts_local"])
        ts_ns  = ts_ms * 1_000_000  # convert ms to ns

        # Check if we've entered the test period
        if ts_ms >= test_start_ts:
            in_test_period = True

        # Apply book update to sim
        sim.on_book_update(
            float(row["price"]),
            float(row["size"]),
            row["side"] == "bid",
            ts_ns,
            int(row["seq_start"]),
            int(row["seq"])
        )

        if sim.is_killed():
            print(f"  Risk kill switch triggered at ts={ts_ms}")
            break

        # Generate signal only in test period, at 1s intervals
        if not in_test_period:
            continue
        if ts_ms - last_signal_ts < signal_interval:
            continue

        last_signal_ts = ts_ms
        n_signals     += 1

        # Find nearest feature row for this timestamp
        # Use the most recent feature snapshot at or before this ts
        available = feat_by_ts[feat_by_ts.index <= ts_ms]
        if available.empty:
            continue

        feat_row = available.iloc[-1]

        # Check for NaN in features
        if feat_row[FEATURE_COLS].isna().any():
            continue

        # Generate signal
        predicted_move = generate_signal(model, scaler, feat_row)

        # Submit order if signal exceeds threshold
        if abs(predicted_move) < SIGNAL_THRESH:
            continue

        n_orders += 1
        if predicted_move > 0:
            # Predict up — buy
            best_ask = None
            # Use current mid as limit price approximation
            mid = feat_row["mid_price"]
            sim.submit_order(mb.Side.BUY, mb.OrderType.LIMIT,
                             mid + 0.01, ORDER_SIZE, ts_ns)
        else:
            # Predict down — sell
            mid = feat_row["mid_price"]
            sim.submit_order(mb.Side.SELL, mb.OrderType.LIMIT,
                             mid - 0.01, ORDER_SIZE, ts_ns)

    # ── Collect results ───────────────────────────────────────────────────────
    fills    = sim.fills()
    position = sim.position()

    print(f"  Signals generated : {n_signals}")
    print(f"  Orders submitted  : {n_orders}")
    print(f"  Fills received    : {len(fills)}")
    print(f"  Final position    : {position.position:.4f}")
    print(f"  Realized PnL      : ${position.realized_pnl:.2f}")
    print(f"  Fee drag          : ${position.fee_drag:.2f}")
    print(f"  Net PnL           : ${position.realized_pnl - position.fee_drag:.2f}")

    # Convert fills to DataFrame
    if fills:
        fills_df = pd.DataFrame([{
            "order_id"  : f.order_id,
            "side"      : "buy" if f.side == mb.Side.BUY else "sell",
            "fill_price": f.fill_price,
            "fill_size" : f.fill_size,
            "fee"       : f.fee,
            "fill_ts_ns": f.fill_ts_ns,
            "is_maker"  : f.is_maker,
        } for f in fills])
    else:
        fills_df = pd.DataFrame()

    return {
        "label"        : label,
        "fills_df"     : fills_df,
        "realized_pnl" : position.realized_pnl,
        "fee_drag"     : position.fee_drag,
        "net_pnl"      : position.realized_pnl - position.fee_drag,
        "n_signals"    : n_signals,
        "n_orders"     : n_orders,
        "n_fills"      : len(fills),
        "ic_test"      : ridge_results["ic_test"],
    }


def run(symbol: str = SYMBOL) -> None:
    print("=" * 60)
    print("EXECUTION SIMULATION")
    print("=" * 60)

    # ── Naive run: zero latency, zero fees, large position limit ─────────────
    naive = run_simulation(
        latency_ms = 0.0,
        taker_fee  = 0.0,
        maker_fee  = 0.0,
        label      = "naive (no fees, no latency)",
    )

    # ── Realistic run: 10ms latency, real fees ────────────────────────────────
    realistic = run_simulation(
        latency_ms = LATENCY_MS,
        taker_fee  = TAKER_FEE,
        maker_fee  = MAKER_FEE,
        label      = "realistic (10ms latency + fees)",
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NAIVE vs REALISTIC COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Naive':>12} {'Realistic':>12}")
    print("-" * 55)
    print(f"{'Signals generated':<30} "
          f"{naive['n_signals']:>12} {realistic['n_signals']:>12}")
    print(f"{'Orders submitted':<30} "
          f"{naive['n_orders']:>12} {realistic['n_orders']:>12}")
    print(f"{'Fills received':<30} "
          f"{naive['n_fills']:>12} {realistic['n_fills']:>12}")
    print(f"{'Realized PnL ($)':<30} "
          f"{naive['realized_pnl']:>12.2f} {realistic['realized_pnl']:>12.2f}")
    print(f"{'Fee drag ($)':<30} "
          f"{naive['fee_drag']:>12.2f} {realistic['fee_drag']:>12.2f}")
    print(f"{'Net PnL ($)':<30} "
          f"{naive['net_pnl']:>12.2f} {realistic['net_pnl']:>12.2f}")

    pnl_gap = naive["net_pnl"] - realistic["net_pnl"]
    print(f"\nPnL gap (naive - realistic): ${pnl_gap:.2f}")
    print("This gap = latency cost + fee drag + missed fills")

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not naive["fills_df"].empty:
        naive["fills_df"].to_parquet(
            output_dir / f"{symbol}_fills_naive.parquet", index=False)
    if not realistic["fills_df"].empty:
        realistic["fills_df"].to_parquet(
            output_dir / f"{symbol}_fills_realistic.parquet", index=False)

    print(f"\nFills saved to data/results/")


if __name__ == "__main__":
    run()