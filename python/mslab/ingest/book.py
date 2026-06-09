"""
book.py
Pure Python L2 order book reconstruction from snapshot + incremental updates.
Validates sequence numbers to detect missed messages.
"""

import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict


class OrderBook:
    """
    Reconstructs an L2 order book from a Binance snapshot + depth updates.

    Maintains two sorted dictionaries:
      bids: price -> size  (highest price = best bid)
      asks: price -> size  (lowest price = best ask)
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_seq: int = -1
        self.sequence_gaps: int = 0

    def apply_snapshot(self, rows: list[dict]) -> None:
        """
        Initialize the book from snapshot rows.
        Clears any existing state first.
        """
        self.bids.clear()
        self.asks.clear()

        for row in rows:
            price = row["price"]
            size = row["size"]
            side = row["side"]

            if side == "bid":
                self.bids[price] = size
            elif side == "ask":
                self.asks[price] = size

        # Set last_seq from snapshot
        if rows:
            self.last_seq = rows[0]["seq"]

        print(f"Snapshot loaded: {len(self.bids)} bid levels, "
              f"{len(self.asks)} ask levels, seq={self.last_seq}")

    def apply_update(self, row: dict) -> None:
        """
        Apply a single depth update row to the book.
        Validates sequence continuity.
        """
        seq_start = row["seq_start"]
        seq_end = row["seq"]
        side = row["side"]
        price = row["price"]
        size = row["size"]
        update_type = row["update_type"]

        # Sequence validation — the critical part
        # First update after snapshot: seq_start must be <= last_seq + 1
        if self.last_seq >= 0 and seq_start > self.last_seq + 1:
            self.sequence_gaps += 1
            print(f"WARNING: sequence gap detected! "
                  f"last_seq={self.last_seq}, this seq_start={seq_start}")

        # Apply the update
        book_side = self.bids if side == "bid" else self.asks

        if update_type == "delete" or size == 0.0:
            book_side.pop(price, None)  # remove level, ignore if not present
        else:
            book_side[price] = size

        self.last_seq = seq_end

    def best_bid(self) -> tuple[float, float] | None:
        """Returns (price, size) of highest bid, or None if empty."""
        if not self.bids:
            return None
        best_price = max(self.bids)
        return best_price, self.bids[best_price]

    def best_ask(self) -> tuple[float, float] | None:
        """Returns (price, size) of lowest ask, or None if empty."""
        if not self.asks:
            return None
        best_price = min(self.asks)
        return best_price, self.asks[best_price]

    def spread(self) -> float | None:
        """Returns the bid-ask spread, or None if book is empty."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def mid_price(self) -> float | None:
        """Returns the mid price (average of best bid and ask)."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid[0] + ask[0]) / 2.0

    def summary(self) -> str:
        """Human readable summary of current book state."""
        bid = self.best_bid()
        ask = self.best_ask()
        spread = self.spread()
        mid = self.mid_price()
        return (
            f"Symbol: {self.symbol} | "
            f"Best Bid: {bid} | "
            f"Best Ask: {ask} | "
            f"Spread: {spread:.4f} | "
            f"Mid: {mid:.4f} | "
            f"Seq gaps: {self.sequence_gaps}"
        )


def load_parquet_as_dicts(path: Path) -> list[dict]:
    """Load a Parquet file and return as a list of row dictionaries."""
    table = pq.read_table(path)
    df = table.to_pandas()
    return df.to_dict(orient="records")


def replay_book(symbol: str = "BTCUSDT") -> OrderBook:
    """
    Full replay: load snapshot, apply all updates, return final book state.
    """
    data_dir = Path("data/normalized")

    snapshot_path = data_dir / f"{symbol}_snapshot.parquet"
    updates_path = data_dir / f"{symbol}_updates.parquet"

    print(f"Loading snapshot from {snapshot_path}")
    snapshot_rows = load_parquet_as_dicts(snapshot_path)

    print(f"Loading updates from {updates_path}")
    update_rows = load_parquet_as_dicts(updates_path)

    # Sort updates by seq to ensure correct order
    update_rows.sort(key=lambda r: r["seq"])

    book = OrderBook(symbol)
    book.apply_snapshot(snapshot_rows)

    print(f"Applying {len(update_rows)} update rows...")
    for row in update_rows:
        book.apply_update(row)

    print(f"\nFinal book state:")
    print(book.summary())

    return book


if __name__ == "__main__":
    replay_book("BTCUSDT")