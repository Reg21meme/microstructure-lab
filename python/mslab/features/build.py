"""
build.py
Feature pipeline: loads normalized Parquet data, replays through C++ book,
captures snapshots at regular intervals, computes microstructure features,
writes output Parquet to data/features/.

Usage:
    python3 -m mslab.features.build
"""

import sys
import pathlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Add cpp/build to path for mslab_bindings
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cpp" / "build"))

import mslab_bindings
from mslab.features.microstructure import (
    compute_features,
    compute_ofi_single,
    compute_mlofi,
)


def load_parquet(path: pathlib.Path) -> list[dict]:
    """Load a Parquet file and return as list of row dicts."""
    table = pq.read_table(path)
    return table.to_pandas().to_dict(orient="records")


def get_book_snapshot(book: mslab_bindings.OrderBook,
                      ts: int,
                      depth: int = 20) -> dict:
    """
    Extract current book state as a snapshot dict.
    Returns bids and asks as lists of (price, size) tuples.
    depth controls how many levels to extract per side.
    """
    if book.bid_levels == 0 or book.ask_levels == 0:
        return None

    bids = book.get_bids(depth)  # list of (price, size), highest first
    asks = book.get_asks(depth)  # list of (price, size), lowest first

    if not bids or not asks:
        return None

    return {
        "ts_local": ts,
        "bids"    : bids,
        "asks"    : asks,
    }


def run_pipeline(symbol: str = "BTCUSDT",
                 snapshot_every: int = 10) -> None:
    """
    Replay book updates and compute features every N updates.

    Parameters:
    symbol         : trading pair to process
    snapshot_every : capture a feature snapshot every N update events
    """
    data_dir    = ROOT / "data" / "normalized"
    output_dir  = ROOT / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading snapshot data for {symbol}...")
    snap_rows = load_parquet(data_dir / f"{symbol}_snapshot.parquet")

    print(f"Loading update data for {symbol}...")
    upd_rows = load_parquet(data_dir / f"{symbol}_updates.parquet")
    upd_rows.sort(key=lambda r: r["seq"])

    print(f"Loaded {len(snap_rows)} snapshot rows, "
          f"{len(upd_rows)} update rows")

    # ── Build C++ book from snapshot ─────────────────────────────────────────
    book = mslab_bindings.OrderBook(symbol)
    book.clear()

    snapshot_seq = None
    for row in snap_rows:
        is_bid = row["side"] == "bid"
        book.apply_snapshot(row["price"], row["size"], is_bid)
        snapshot_seq = row["seq"]

    if snapshot_seq is not None:
        book.set_snapshot_seq(snapshot_seq)

    print(f"Snapshot loaded: {book.bid_levels} bid levels, "
          f"{book.ask_levels} ask levels, seq={book.last_seq}")

    # ── Replay updates and capture feature snapshots ─────────────────────────
    features     = []
    update_count = 0
    prev_snap    = None  # previous book snapshot for OFI computation

    for row in upd_rows:
        is_bid = row["side"] == "bid"
        book.apply_update(
            row["price"], row["size"], is_bid,
            row["seq_start"], row["seq"]
        )
        update_count += 1

        # Capture snapshot every N updates
        if update_count % snapshot_every == 0:
            snap = get_book_snapshot(book, row["ts_local"])
            if snap is None:
                continue

            feat = compute_features(snap)
            if feat is None:
                continue

            # OFI requires a previous snapshot — skip the first one
            if prev_snap is not None:
                feat["ofi"]   = compute_ofi_single(prev_snap, snap)
                mlofi         = compute_mlofi(prev_snap, snap, levels=10)
                for i, val in enumerate(mlofi):
                    feat[f"mlofi_{i+1}"] = val
            else:
                feat["ofi"] = np.nan
                for i in range(10):
                    feat[f"mlofi_{i+1}"] = np.nan

            prev_snap = snap
            features.append(feat)

    print(f"Replayed {update_count} updates, "
          f"captured {len(features)} feature snapshots")

    if not features:
        print("ERROR: no features generated — check data")
        return

    # ── Write output Parquet ──────────────────────────────────────────────────
    output_path = output_dir / f"{symbol}_features.parquet"
    df = pd.DataFrame(features)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

    print(f"Features written to {output_path}")
    print(f"Schema: {table.schema}")
    print(f"\nSample (first 5 rows):")
    print(df.head())
    print(f"\nBasic stats:")
    print(df.describe())


if __name__ == "__main__":
    run_pipeline()