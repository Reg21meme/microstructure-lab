"""
normalize.py
Collects Binance depth update stream for a short window,
normalizes updates to the same schema as the snapshot,
and saves to Parquet.
"""

import json
import time
import threading
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import websocket

DATA_DIR = Path("data/normalized")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_depth_update(msg: dict, symbol: str) -> list[dict]:
    """
    Convert a single Binance depth update message into a list of rows.
    Each price level change becomes one row.
    """
    ts_local = int(time.time() * 1000)
    seq_start = msg["U"]  # first update ID in this message
    seq_end = msg["u"]    # last update ID in this message

    rows = []

    for price, size in msg["b"]:  # bids
        rows.append({
            "ts_local":    ts_local,
            "seq":         seq_end,
            "seq_start":   seq_start,
            "side":        "bid",
            "price":       float(price),
            "size":        float(size),
            "update_type": "delete" if float(size) == 0.0 else "update",
            "symbol":      symbol,
        })

    for price, size in msg["a"]:  # asks
        rows.append({
            "ts_local":    ts_local,
            "seq":         seq_end,
            "seq_start":   seq_start,
            "side":        "ask",
            "price":       float(price),
            "size":        float(size),
            "update_type": "delete" if float(size) == 0.0 else "update",
            "symbol":      symbol,
        })

    return rows


def collect_updates(symbol: str = "BTCUSDT", duration_seconds: int = 5):
    """
    Connect to Binance WebSocket stream and collect depth updates
    for a fixed duration, then save to Parquet.
    """
    stream = symbol.lower() + "@depth@100ms"
    url = f"wss://data-stream.binance.vision/ws/{stream}"

    all_rows = []
    stop_event = threading.Event()

    def on_message(ws, message):
        msg = json.loads(message)
        rows = normalize_depth_update(msg, symbol)
        all_rows.extend(rows)

    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    def on_open(ws):
        print(f"Connected to {url}")
        # Stop after duration_seconds
        def stop():
            time.sleep(duration_seconds)
            ws.close()
        threading.Thread(target=stop).start()

    ws = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_open=on_open,
    )
    ws.run_forever()

    if not all_rows:
        print("No updates received.")
        return

    # Convert to PyArrow table
    table = pa.table({
        "ts_local":    pa.array([r["ts_local"]    for r in all_rows], type=pa.int64()),
        "seq":         pa.array([r["seq"]         for r in all_rows], type=pa.int64()),
        "seq_start":   pa.array([r["seq_start"]   for r in all_rows], type=pa.int64()),
        "side":        pa.array([r["side"]        for r in all_rows], type=pa.string()),
        "price":       pa.array([r["price"]       for r in all_rows], type=pa.float64()),
        "size":        pa.array([r["size"]        for r in all_rows], type=pa.float64()),
        "update_type": pa.array([r["update_type"] for r in all_rows], type=pa.string()),
        "symbol":      pa.array([r["symbol"]      for r in all_rows], type=pa.string()),
    })

    out_path = DATA_DIR / f"{symbol}_updates.parquet"
    pq.write_table(table, out_path)
    print(f"Saved {table.num_rows} update rows to {out_path}")
    print(table.to_pandas().head(10))


if __name__ == "__main__":
    collect_updates("BTCUSDT", duration_seconds=5)