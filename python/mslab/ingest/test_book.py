"""
test_book.py
Unit tests for OrderBook — verifies snapshot + update replay is correct.
Run with: python3 -m pytest python/mslab/ingest/test_book.py -v
"""

import pytest
import sys
from pathlib import Path

# Make sure Python can find the mslab package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mslab.ingest.book import OrderBook


def make_snapshot_rows(bids: list[tuple], asks: list[tuple], seq: int = 100) -> list[dict]:
    """Helper: turn simple (price, size) tuples into snapshot row dicts."""
    rows = []
    for price, size in bids:
        rows.append({
            "ts_local": 1000, "seq": seq, "seq_start": seq,
            "side": "bid", "price": price, "size": size,
            "update_type": "snapshot", "symbol": "TEST"
        })
    for price, size in asks:
        rows.append({
            "ts_local": 1000, "seq": seq, "seq_start": seq,
            "side": "ask", "price": price, "size": size,
            "update_type": "snapshot", "symbol": "TEST"
        })
    return rows


def make_update_row(side: str, price: float, size: float,
                    seq_start: int, seq_end: int) -> dict:
    """Helper: build a single update row dict."""
    return {
        "ts_local": 2000,
        "seq": seq_end,
        "seq_start": seq_start,
        "side": side,
        "price": price,
        "size": size,
        "update_type": "delete" if size == 0.0 else "update",
        "symbol": "TEST"
    }


# ── Test 1 ────────────────────────────────────────────────────────────────────
def test_snapshot_loading():
    """Basic snapshot loads correctly — best bid, best ask, spread, mid."""
    book = OrderBook("TEST")
    rows = make_snapshot_rows(
        bids=[(100.0, 1.0), (99.0, 2.0)],
        asks=[(101.0, 1.0), (102.0, 2.0)],
        seq=100
    )
    book.apply_snapshot(rows)

    assert book.best_bid() == (100.0, 1.0), "Best bid should be highest bid price"
    assert book.best_ask() == (101.0, 1.0), "Best ask should be lowest ask price"
    assert book.spread() == pytest.approx(1.0), "Spread should be ask - bid = 1.0"
    assert book.mid_price() == pytest.approx(100.5), "Mid should be (100+101)/2"


# ── Test 2 ────────────────────────────────────────────────────────────────────
def test_update_changes_size():
    """Applying an update changes the size at that price level."""
    book = OrderBook("TEST")
    book.apply_snapshot(make_snapshot_rows(
        bids=[(100.0, 1.0)], asks=[(101.0, 1.0)], seq=100
    ))

    update = make_update_row("bid", 100.0, 5.0, seq_start=101, seq_end=101)
    book.apply_update(update)

    assert book.best_bid() == (100.0, 5.0), "Size should update to 5.0"


# ── Test 3 ────────────────────────────────────────────────────────────────────
def test_delete_removes_level():
    """Deleting best bid (size=0) removes it; next level becomes best bid."""
    book = OrderBook("TEST")
    book.apply_snapshot(make_snapshot_rows(
        bids=[(100.0, 1.0), (99.0, 2.0)],
        asks=[(101.0, 1.0)],
        seq=100
    ))

    delete = make_update_row("bid", 100.0, 0.0, seq_start=101, seq_end=101)
    book.apply_update(delete)

    assert book.best_bid() == (99.0, 2.0), "After deleting 100.0, best bid should be 99.0"
    assert 100.0 not in book.bids, "Price level 100.0 should be gone"


# ── Test 4 ────────────────────────────────────────────────────────────────────
def test_sequence_gap_detection():
    """A gap in sequence numbers increments sequence_gaps counter."""
    book = OrderBook("TEST")
    book.apply_snapshot(make_snapshot_rows(
        bids=[(100.0, 1.0)], asks=[(101.0, 1.0)], seq=100
    ))

    assert book.sequence_gaps == 0

    # seq_start=200 when last_seq=100 — gap of 99
    gapped_update = make_update_row("bid", 100.0, 2.0, seq_start=200, seq_end=200)
    book.apply_update(gapped_update)

    assert book.sequence_gaps == 1, "Should detect the sequence gap"


# ── Test 5 ────────────────────────────────────────────────────────────────────
def test_new_price_level_added():
    """An update for a price not in the snapshot adds a new level."""
    book = OrderBook("TEST")
    book.apply_snapshot(make_snapshot_rows(
        bids=[(100.0, 1.0)], asks=[(101.0, 1.0)], seq=100
    ))

    new_level = make_update_row("bid", 100.5, 3.0, seq_start=101, seq_end=101)
    book.apply_update(new_level)

    assert book.best_bid() == (100.5, 3.0), "New higher bid should become best bid"
    assert 100.5 in book.bids, "New price level should exist in bids"


# ── Test 6 ────────────────────────────────────────────────────────────────────
def test_book_never_crossed():
    """After multiple updates, best bid should always be less than best ask."""
    book = OrderBook("TEST")
    book.apply_snapshot(make_snapshot_rows(
        bids=[(100.0, 1.0), (99.0, 2.0)],
        asks=[(101.0, 1.0), (102.0, 2.0)],
        seq=100
    ))

    updates = [
        make_update_row("bid", 100.0, 3.0, seq_start=101, seq_end=101),
        make_update_row("ask", 101.0, 0.5, seq_start=102, seq_end=102),
        make_update_row("bid", 99.0, 0.0, seq_start=103, seq_end=103),
        make_update_row("ask", 102.0, 1.5, seq_start=104, seq_end=104),
    ]

    for update in updates:
        book.apply_update(update)
        bid = book.best_bid()
        ask = book.best_ask()
        if bid and ask:
            assert bid[0] < ask[0], f"Book crossed! bid={bid[0]} >= ask={ask[0]}"