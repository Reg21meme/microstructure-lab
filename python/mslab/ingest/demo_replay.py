"""
demo_replay.py
Replays the order book update by update, printing best bid/ask + spread
at each step. Run via: make demo-replay
"""

import time
import pyarrow.parquet as pq
from pathlib import Path
from mslab.ingest.book import OrderBook


def load_parquet_as_dicts(path: Path) -> list[dict]:
    """Load a Parquet file and return as a list of row dicts."""
    table = pq.read_table(path)
    return table.to_pandas().to_dict(orient="records")


def demo_replay(
    symbol: str = "BTCUSDT",
    print_every: int = 50,
    slow_mode: bool = True,
):
    """
    Replay the order book update by update.

    Args:
        symbol:      which symbol to replay
        print_every: print book state every N updates (keeps output readable)
        slow_mode:   if True, pause between prints so you can watch it live
    """
    data_dir = Path("data/normalized")

    snapshot_path = data_dir / f"{symbol}_snapshot.parquet"
    updates_path  = data_dir / f"{symbol}_updates.parquet"

    print(f"\n{'='*60}")
    print(f"  MicrostructureLab — L2 Order Book Replay")
    print(f"  Symbol: {symbol}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading snapshot...")
    snapshot_rows = load_parquet_as_dicts(snapshot_path)

    print("Loading updates...")
    update_rows = load_parquet_as_dicts(updates_path)
    update_rows.sort(key=lambda r: r["seq"])

    # Initialize book
    book = OrderBook(symbol)
    book.apply_snapshot(snapshot_rows)

    print(f"\nInitial book state (from snapshot):")
    print_book_state(book, update_num=0)

    print(f"\nReplaying {len(update_rows)} updates "
          f"(printing every {print_every})...\n")

    # Replay updates one by one
    for i, row in enumerate(update_rows, start=1):
        book.apply_update(row)

        if i % print_every == 0 or i == len(update_rows):
            print_book_state(book, update_num=i)
            if slow_mode:
                time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"  Replay complete.")
    print(f"  Total updates applied: {len(update_rows)}")
    print(f"  Sequence gaps detected: {book.sequence_gaps}")
    print(f"{'='*60}\n")


def print_book_state(book: OrderBook, update_num: int):
    """Print a single line summary of current book state."""
    bid = book.best_bid()
    ask = book.best_ask()
    spread = book.spread()
    mid = book.mid_price()

    bid_str = f"${bid[0]:,.2f} x {bid[1]:.4f}" if bid else "empty"
    ask_str = f"${ask[0]:,.2f} x {ask[1]:.4f}" if ask else "empty"
    spread_str = f"${spread:.4f}" if spread else "N/A"
    mid_str = f"${mid:,.4f}" if mid else "N/A"

    print(f"  Update {update_num:>5} | "
          f"Bid: {bid_str:<28} | "
          f"Ask: {ask_str:<28} | "
          f"Spread: {spread_str:<12} | "
          f"Mid: {mid_str}")


if __name__ == "__main__":
    demo_replay("BTCUSDT", print_every=50, slow_mode=False)