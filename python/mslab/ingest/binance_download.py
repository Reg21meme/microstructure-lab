"""
Downloads a Binance L2 order book snapshot and depth updates,  normalizes them, and saves to Parquet.
"""

import time
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Where we save the data
DATA_DIR = Path("data/normalized")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_snapshot(symbol: str, depth: int = 100) -> dict:
    """
    Fetch a full order book snapshot from Binance and returns raw JSON with bids and asks.
    """
    url = "https://data-api.binance.vision/api/v3/depth"

    params = {"symbol": symbol, "limit": depth}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def normalize_snapshot(snapshot: dict, symbol: str) -> pa.Table:
    """
    Convert raw Binance snapshot into a table
    """
    ts_local = int(time.time() * 1000)  # milliseconds
    seq = snapshot["lastUpdateId"]

    rows = []

    for price, size in snapshot["bids"]:
        rows.append({
            "ts_local": ts_local,
            "seq": seq,
            "side": "bid",
            "price": float(price),
            "size": float(size),
            "update_type": "snapshot",
            "symbol": symbol,
        })

    for price, size in snapshot["asks"]:
        rows.append({
            "ts_local": ts_local,
            "seq": seq,
            "side": "ask",
            "price": float(price),
            "size": float(size),
            "update_type": "snapshot",
            "symbol": symbol,
        })

    return pa.table({
        "ts_local":   pa.array([r["ts_local"]   for r in rows], type=pa.int64()),
        "seq":        pa.array([r["seq"]         for r in rows], type=pa.int64()),
        "side":       pa.array([r["side"]        for r in rows], type=pa.string()),
        "price":      pa.array([r["price"]       for r in rows], type=pa.float64()),
        "size":       pa.array([r["size"]        for r in rows], type=pa.float64()),
        "update_type":pa.array([r["update_type"] for r in rows], type=pa.string()),
        "symbol":     pa.array([r["symbol"]      for r in rows], type=pa.string()),
    })


def download_and_save(symbol: str = "BTCUSDT"):
    print(f"Fetching {symbol} order book snapshot...")
    snapshot = fetch_snapshot(symbol)

    print(f"  Last update ID (seq): {snapshot['lastUpdateId']}")
    print(f"  Bids: {len(snapshot['bids'])} levels")
    print(f"  Asks: {len(snapshot['asks'])} levels")

    table = normalize_snapshot(snapshot, symbol)

    out_path = DATA_DIR / f"{symbol}_snapshot.parquet"
    pq.write_table(table, out_path)
    print(f"  Saved to {out_path}")
    print(f"  Rows: {table.num_rows}")
    print(table.to_pandas().head(10))


if __name__ == "__main__":
    download_and_save("BTCUSDT")