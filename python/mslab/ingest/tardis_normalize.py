"""
tardis_normalize.py
Normalize Tardis incremental_book_L2 CSV into the same Parquet schema
used by the live Binance collector, so the existing feature pipeline
works unchanged.

Tardis schema:
    exchange, symbol, timestamp, local_timestamp, is_snapshot,
    side, price, amount

Output schema (matches collector.py):
    ts_local    : int64  (milliseconds)
    seq         : int64  (row index — Tardis has no seq number)
    seq_start   : int64  (same as seq for Tardis data)
    side        : str    ('bid' or 'ask')
    price       : float64
    size        : float64
    update_type : str    ('snapshot' or 'update')
    symbol      : str

Usage:
    python3 -m mslab.ingest.tardis_normalize \
        data/raw/tardis/BTCUSDT_2025-06-01_incremental_book_L2.csv.gz \
        BTCUSDT
"""

import sys
import pathlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT     = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "normalized"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_tardis(csv_path: str | pathlib.Path,
                     symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read Tardis incremental_book_L2 CSV and split into
    snapshot rows and update rows matching collector.py schema.

    Returns
    -------
    (snapshot_df, updates_df)
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, compression="gzip")
    print(f"  Raw rows: {len(df):,}")

    # Tardis timestamps are microseconds — convert to milliseconds
    df["ts_local"] = (df["local_timestamp"] // 1000).astype("int64")

    # Use row index as seq (Tardis has no native seq number)
    df["seq"]       = df.index.astype("int64")
    df["seq_start"] = df["seq"]

    # Rename columns to match collector schema
    df["size"]   = df["amount"].astype("float64")
    df["symbol"] = symbol

    # update_type: 'snapshot' if is_snapshot==True, else 'update'
    df["update_type"] = df["is_snapshot"].apply(
        lambda x: "snapshot" if str(x).lower() == "true" else "update"
    )

    # Keep only rows from first snapshot onward
    # (Tardis may have buffered updates before the first snapshot)
    first_snap_idx = df[df["update_type"] == "snapshot"].index[0]
    df = df.loc[first_snap_idx:].reset_index(drop=True)
    df["seq"]       = df.index.astype("int64")
    df["seq_start"] = df["seq"]

    print(f"  Rows after sync filter: {len(df):,}")
    print(f"  Snapshot rows: {(df['update_type']=='snapshot').sum():,}")
    print(f"  Update rows  : {(df['update_type']=='update').sum():,}")
    print(f"  Price range  : {df['price'].min():.2f} – {df['price'].max():.2f}")
    print(f"  Time range   : {df['ts_local'].min()} – {df['ts_local'].max()} ms")

    # Split into snapshot and updates
    snap_df    = df[df["update_type"] == "snapshot"].copy()
    updates_df = df[df["update_type"] == "update"].copy()

    # Select and order columns to match collector schema
    cols = ["ts_local", "seq", "seq_start", "side", "price", "size",
            "update_type", "symbol"]

    return snap_df[cols], updates_df[cols]


def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    print(f"  Saved: {path} ({len(df):,} rows)")


def run(csv_path: str, symbol: str) -> None:
    print("=" * 60)
    print(f"TARDIS NORMALIZER — {symbol}")
    print("=" * 60)

    snap_df, updates_df = normalize_tardis(csv_path, symbol)

    snap_path    = DATA_DIR / f"{symbol}_snapshot.parquet"
    updates_path = DATA_DIR / f"{symbol}_updates.parquet"

    print(f"\nSaving to {DATA_DIR}...")
    save_parquet(snap_df,    snap_path)
    save_parquet(updates_df, updates_path)

    # Sanity check
    print(f"\nSanity check:")
    print(f"  First snapshot seq : {snap_df['seq'].iloc[0]}")
    print(f"  First update seq   : {updates_df['seq'].iloc[0] if not updates_df.empty else 'N/A'}")
    print(f"  Snapshot levels    : {len(snap_df)} rows "
          f"({(snap_df['side']=='bid').sum()} bid, "
          f"{(snap_df['side']=='ask').sum()} ask)")
    print(f"\nDone. Run feature pipeline next:")
    print(f"  python3 -m mslab.features.build {symbol}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 -m mslab.ingest.tardis_normalize "
              "<path/to/file.csv.gz> <SYMBOL>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])