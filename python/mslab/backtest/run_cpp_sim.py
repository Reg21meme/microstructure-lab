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

from mslab.configs.fees import load_fees, FeeConfig
from mslab.backtest.queue_model import load_queue_config, QueueModel, QueueConfig
from mslab.backtest.markout import compute_markouts, markout_summary 

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp" / "build"))

import mslab_bindings as mb
from mslab.models.train_baseline import (
    load_clean_data, time_split, train_ridge, FEATURE_COLS
)

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOL         = "BTCUSDT"
LATENCY_MS     = 10.0
FEE_SCENARIO   = "binance_vip0"
QUEUE_SCENARIO = "base"
MAX_POSITION   = 1.0
MAX_DRAWDOWN   = 5000.0
SIGNAL_THRESH  = 0.10
ORDER_SIZE     = 0.01


def make_sim(latency_ms: float = LATENCY_MS,
             fee_cfg: FeeConfig | None = None) -> mb.ExecutionSim:
    """Create a configured ExecutionSim instance."""
    if fee_cfg is None:
        fee_cfg = load_fees(FEE_SCENARIO)
    latency             = mb.LatencyModel(base_ms=latency_ms, jitter_ms=0.0)
    fees                = mb.FeeModel()
    fees.taker_fee      = fee_cfg.taker_fee
    fees.maker_fee      = fee_cfg.maker_fee
    limits              = mb.RiskLimits()
    limits.max_position = MAX_POSITION
    limits.max_drawdown = MAX_DRAWDOWN
    return mb.ExecutionSim(SYMBOL, latency, fees, limits)


def load_updates() -> pd.DataFrame:
    path = ROOT / "data" / "normalized" / f"{SYMBOL}_updates.parquet"
    df   = pq.read_table(path).to_pandas()
    return df.sort_values("seq").reset_index(drop=True)


def load_snapshot_rows() -> pd.DataFrame:
    path = ROOT / "data" / "normalized" / f"{SYMBOL}_snapshot.parquet"
    return pq.read_table(path).to_pandas()


def generate_signal(model, scaler, features: pd.Series) -> float:
    X = scaler.transform(features[FEATURE_COLS].values.reshape(1, -1))
    return float(model.predict(X)[0])


def run_simulation(latency_ms: float = LATENCY_MS,
                   fee_cfg: FeeConfig | None = None,
                   queue_cfg: QueueConfig | None = None,
                   label: str = "realistic") -> dict:
    print(f"\nRunning simulation: {label}")
    print(f"  Latency={latency_ms}ms")

    # ── Load and split data ───────────────────────────────────────────────────
    feat_df     = load_clean_data(SYMBOL)
    train, test = time_split(feat_df, train_frac=0.7)

    # ── Train model ───────────────────────────────────────────────────────────
    ridge_results = train_ridge(train, test)
    model         = ridge_results["model"]
    scaler        = ridge_results["scaler"]
    print(f"  Model trained: test IC={ridge_results['ic_test']:.4f}")

    # ── Load L2 updates ───────────────────────────────────────────────────────
    upd_rows      = load_updates()
    snap_rows     = load_snapshot_rows()
    test_start_ts = test["ts_local"].iloc[0]
    print(f"  Test period starts at ts={test_start_ts}")

    # ── Create sim ────────────────────────────────────────────────────────────
    if fee_cfg is None:
        fee_cfg = load_fees(FEE_SCENARIO)
    sim = make_sim(latency_ms, fee_cfg)
    print(f"  Fee scenario     : {fee_cfg.scenario} "
          f"(maker={fee_cfg.maker_fee}, taker={fee_cfg.taker_fee})")

    if queue_cfg is None:
        queue_cfg = load_queue_config(QUEUE_SCENARIO)
    queue_model = QueueModel(queue_cfg)
    print(f"  Queue scenario   : {queue_cfg.scenario} "
          f"(use_model={queue_cfg.use_queue_model}, "
          f"min_prob={queue_cfg.min_fill_prob})")

    # ── Initialize book from snapshot ─────────────────────────────────────────
    snapshot_seq = int(snap_rows["seq"].iloc[0])
    for _, row in snap_rows.iterrows():
        sim.on_book_update(row["price"], row["size"],
                           row["side"] == "bid",
                           0, snapshot_seq, snapshot_seq)

    # ── Replay updates and generate signals ───────────────────────────────────
    feat_by_ts     = feat_df.set_index("ts_local")
    last_signal_ts = -1
    signal_interval = 1_000
    n_signals      = 0
    n_orders       = 0
    in_test_period = False

    for _, row in upd_rows.iterrows():
        ts_ms = int(row["ts_local"])
        ts_ns = ts_ms * 1_000_000

        if ts_ms >= test_start_ts:
            in_test_period = True

        sim.on_book_update(
            float(row["price"]),
            float(row["size"]),
            row["side"] == "bid",
            ts_ns,
            int(row["seq_start"]),
            int(row["seq"]),
        )

        if sim.is_killed():
            print(f"  Risk kill switch triggered at ts={ts_ms}")
            break

        if not in_test_period:
            continue
        if ts_ms - last_signal_ts < signal_interval:
            continue

        last_signal_ts = ts_ms
        n_signals     += 1

        available = feat_by_ts[feat_by_ts.index <= ts_ms]
        if available.empty:
            continue

        feat_row = available.iloc[-1]
        if feat_row[FEATURE_COLS].isna().any():
            continue

        predicted_move = generate_signal(model, scaler, feat_row)
        mid            = feat_row["mid_price"]
        position       = sim.position().position

        # ── Queue-position fill gate ──────────────────────────────────────────
        def maybe_submit(side_is_buy: bool, order_type, price: float,
                         size: float) -> None:
            filled, prob = queue_model.should_fill(
                side_is_buy, float(feat_row["depth_imbalance_5"])
            )
            if filled:
                nonlocal n_orders
                n_orders += 1
                sim.submit_order(
                    mb.Side.BUY if side_is_buy else mb.Side.SELL,
                    order_type, price, size, ts_ns,
                )

        # ── Entry and exit logic ──────────────────────────────────────────────
        if predicted_move > SIGNAL_THRESH:
            if position < MAX_POSITION:
                maybe_submit(True, mb.OrderType.LIMIT, mid + 0.01, ORDER_SIZE)
            elif position < 0:
                maybe_submit(True, mb.OrderType.LIMIT, mid + 0.01,
                             min(ORDER_SIZE, abs(position)))

        elif predicted_move < -SIGNAL_THRESH:
            if position > -MAX_POSITION:
                maybe_submit(False, mb.OrderType.LIMIT, mid - 0.01, ORDER_SIZE)
            elif position > 0:
                maybe_submit(False, mb.OrderType.LIMIT, mid - 0.01,
                             min(ORDER_SIZE, position))

        else:
            if abs(position) > ORDER_SIZE / 2:
                if position > 0:
                    maybe_submit(False, mb.OrderType.IOC, mid - 0.01,
                                 min(ORDER_SIZE, position))
                else:
                    maybe_submit(True, mb.OrderType.IOC, mid + 0.01,
                                 min(ORDER_SIZE, abs(position)))

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

    fills_df = pd.DataFrame([{
        "order_id"  : f.order_id,
        "side"      : "buy" if f.side == mb.Side.BUY else "sell",
        "fill_price": f.fill_price,
        "fill_size" : f.fill_size,
        "fee"       : f.fee,
        "fill_ts_ns": f.fill_ts_ns,
        "is_maker"  : f.is_maker,
    } for f in fills]) if fills else pd.DataFrame()

    # ── Markout calculation ───────────────────────────────────────────────────
    if not fills_df.empty:
        fills_df   = compute_markouts(fills_df, feat_df)
        mo_summary = markout_summary(fills_df)
        print(f"\n  Markout summary (adverse selection proxy):")
        print(mo_summary.to_string(index=False))
    else:
        mo_summary = pd.DataFrame()
        
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
        "markout_summary": mo_summary,
    }

def run(symbol: str = SYMBOL) -> None:
    print("=" * 60)
    print("EXECUTION SIMULATION")
    print("=" * 60)

    naive = run_simulation(
        latency_ms = 0.0,
        fee_cfg    = load_fees("naive"),
        queue_cfg  = load_queue_config("naive"),
        label      = "naive (no fees, no latency, no queue)",
    )

    realistic = run_simulation(
        latency_ms = LATENCY_MS,
        fee_cfg    = load_fees("binance_vip0"),
        queue_cfg  = load_queue_config("base"),
        label      = "realistic (10ms latency + fees + queue)",
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
    print("This gap = latency cost + fee drag + missed fills + queue rejection")

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