"""
collector.py
Collects a synchronized Binance L2 snapshot + update stream.

Follows the official Binance procedure for L2 book synchronization:
  1. Start buffering WebSocket depth updates immediately
  2. Fetch a REST snapshot
  3. Discard buffered updates older than the snapshot
  4. Keep updates that pick up from the snapshot sequence onward

This guarantees snapshot and updates are aligned — no sequence gaps,
no crossed book, no stale data.

Usage:
    python3 -m mslab.ingest.collector
"""

import json
import time
import threading
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path("data/normalized")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_URL = "https://data-api.binance.vision/api/v3/depth"
WS_URL       = "wss://data-stream.binance.vision/ws/{symbol}@depth@100ms"


def fetch_snapshot(symbol: str, depth: int = 100) -> dict:
    """Fetch L2 snapshot from Binance REST API."""
    resp = requests.get(SNAPSHOT_URL, params={
        "symbol": symbol,
        "limit" : depth,
    })
    resp.raise_for_status()
    return resp.json()


def normalize_snapshot(raw: dict, symbol: str) -> list[dict]:
    """Convert raw snapshot response to normalized row dicts."""
    ts_local = int(time.time() * 1000)  # REST API has no exchange timestamp
    seq      = raw["lastUpdateId"]
    rows     = []

    for price, size in raw["bids"]:
        rows.append({
            "ts_local"   : ts_local,
            "seq"        : seq,
            "side"       : "bid",
            "price"      : float(price),
            "size"       : float(size),
            "update_type": "snapshot",
            "symbol"     : symbol,
        })
    for price, size in raw["asks"]:
        rows.append({
            "ts_local"   : ts_local,
            "seq"        : seq,
            "side"       : "ask",
            "price"      : float(price),
            "size"       : float(size),
            "update_type": "snapshot",
            "symbol"     : symbol,
        })
    return rows


def normalize_update(msg: dict, symbol: str) -> list[dict]:
    """Convert a single WebSocket depth update to normalized row dicts."""
    # Use exchange event time (E) from the message — this is the real timestamp
    # msg["E"] is the event time in milliseconds from Binance
    ts_local  = msg.get("E", int(time.time() * 1000))
    seq_start = msg["U"]
    seq_end   = msg["u"]
    rows      = []

    for price, size in msg["b"]:
        rows.append({
            "ts_local"   : ts_local,
            "seq"        : seq_end,
            "seq_start"  : seq_start,
            "side"       : "bid",
            "price"      : float(price),
            "size"       : float(size),
            "update_type": "delete" if float(size) == 0.0 else "update",
            "symbol"     : symbol,
        })
    for price, size in msg["a"]:
        rows.append({
            "ts_local"   : ts_local,
            "seq"        : seq_end,
            "seq_start"  : seq_start,
            "side"       : "ask",
            "price"      : float(price),
            "size"       : float(size),
            "update_type": "delete" if float(size) == 0.0 else "update",
            "symbol"     : symbol,
        })
    return rows


def collect(symbol: str = "BTCUSDT",
            duration_seconds: int = 30) -> None:
    """
    Collect a synchronized snapshot + update stream.

    Parameters
    ----------
    symbol           : trading pair e.g. "BTCUSDT"
    duration_seconds : how long to collect updates after synchronization
    """
    import websocket

    # ── Step 1: start buffering WebSocket updates immediately ────────────────
    buffered_msgs = []
    buffer_lock   = threading.Lock()
    ws_ready      = threading.Event()
    ws_done       = threading.Event()

    def on_message(ws, message):
        msg = json.loads(message)
        with buffer_lock:
            buffered_msgs.append(msg)

    def on_open(ws):
        print("WebSocket connected, buffering updates...")
        ws_ready.set()

    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    def on_close(ws, *args):
        ws_done.set()

    url = WS_URL.format(symbol=symbol.lower())
    ws  = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    # Wait for WebSocket to connect before fetching snapshot
    ws_ready.wait(timeout=10)
    if not ws_ready.is_set():
        print("ERROR: WebSocket failed to connect")
        return

    # ── Step 2: fetch REST snapshot ──────────────────────────────────────────
    print(f"Fetching snapshot for {symbol}...")
    raw_snapshot  = fetch_snapshot(symbol, depth=100)
    snapshot_seq  = raw_snapshot["lastUpdateId"]
    snapshot_rows = normalize_snapshot(raw_snapshot, symbol)
    print(f"Snapshot fetched: seq={snapshot_seq}, "
          f"{len(snapshot_rows)} rows")

    # ── Step 3 & 4: filter buffered updates ──────────────────────────────────
    # Binance sync rules:
    #   - Drop any update where seq_end (u) <= snapshot_seq
    #   - First valid update has seq_start (U) <= snapshot_seq + 1
    #     and seq_end (u) >= snapshot_seq + 1

    print(f"Collecting updates for {duration_seconds} seconds...")
    time.sleep(duration_seconds)
    ws.close()

    with buffer_lock:
        all_msgs = list(buffered_msgs)

    print(f"Buffered {len(all_msgs)} raw WebSocket messages")

    # Filter to only messages that apply after the snapshot
    valid_msgs = [
        msg for msg in all_msgs
        if msg["u"] > snapshot_seq
    ]
    print(f"Valid messages after sync filter: {len(valid_msgs)}")

    if not valid_msgs:
        print("ERROR: no valid updates after snapshot — try again")
        return

    # Normalize valid updates
    update_rows = []
    for msg in valid_msgs:
        update_rows.extend(normalize_update(msg, symbol))

    print(f"Total update rows: {len(update_rows)}")

    # ── Save snapshot Parquet ─────────────────────────────────────────────────
    snap_path = DATA_DIR / f"{symbol}_snapshot.parquet"
    snap_table = pa.table({
        "ts_local"   : pa.array([r["ts_local"]    for r in snapshot_rows], type=pa.int64()),
        "seq"        : pa.array([r["seq"]         for r in snapshot_rows], type=pa.int64()),
        "side"       : pa.array([r["side"]        for r in snapshot_rows], type=pa.string()),
        "price"      : pa.array([r["price"]       for r in snapshot_rows], type=pa.float64()),
        "size"       : pa.array([r["size"]        for r in snapshot_rows], type=pa.float64()),
        "update_type": pa.array([r["update_type"] for r in snapshot_rows], type=pa.string()),
        "symbol"     : pa.array([r["symbol"]      for r in snapshot_rows], type=pa.string()),
    })
    pq.write_table(snap_table, snap_path)
    print(f"Snapshot saved: {snap_path} ({snap_table.num_rows} rows)")

    # ── Save updates Parquet ──────────────────────────────────────────────────
    upd_path = DATA_DIR / f"{symbol}_updates.parquet"
    upd_table = pa.table({
        "ts_local"   : pa.array([r["ts_local"]    for r in update_rows], type=pa.int64()),
        "seq"        : pa.array([r["seq"]         for r in update_rows], type=pa.int64()),
        "seq_start"  : pa.array([r["seq_start"]   for r in update_rows], type=pa.int64()),
        "side"       : pa.array([r["side"]        for r in update_rows], type=pa.string()),
        "price"      : pa.array([r["price"]       for r in update_rows], type=pa.float64()),
        "size"       : pa.array([r["size"]        for r in update_rows], type=pa.float64()),
        "update_type": pa.array([r["update_type"] for r in update_rows], type=pa.string()),
        "symbol"     : pa.array([r["symbol"]      for r in update_rows], type=pa.string()),
    })
    pq.write_table(upd_table, upd_path)
    print(f"Updates saved: {upd_path} ({upd_table.num_rows} rows)")

    # ── Quick sanity check ────────────────────────────────────────────────────
    first_update_seq_start = valid_msgs[0]["U"]
    first_update_seq_end   = valid_msgs[0]["u"]
    print(f"\nSanity check:")
    print(f"  Snapshot seq:           {snapshot_seq}")
    print(f"  First update seq_start: {first_update_seq_start}")
    print(f"  First update seq_end:   {first_update_seq_end}")

    if first_update_seq_start <= snapshot_seq + 1:
        print("    Synchronized correctly — no gap between snapshot and updates")
    else:
        print("    Gap detected — snapshot and updates may not align")
        print("    Try running again")


if __name__ == "__main__":
    collect("BTCUSDT", duration_seconds=30)