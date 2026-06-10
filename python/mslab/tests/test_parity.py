import pytest
import mslab_bindings
from mslab.ingest.book import OrderBook as PyOrderBook


def make_py_book():
    """Build a clean two-sided Python book: bid@99, ask@101, seq=100."""
    book = PyOrderBook("BTCUSDT")
    book.apply_snapshot([
        {"side": "bid", "price": 99.0,  "size": 5.0, "seq": 100},
        {"side": "ask", "price": 101.0, "size": 3.0, "seq": 100},
    ])
    return book


def make_cpp_book():
    """Build a clean two-sided C++ book: bid@99, ask@101, seq=100."""
    book = mslab_bindings.OrderBook("BTCUSDT")
    book.clear()
    book.apply_snapshot(99.0,  5.0, True)
    book.apply_snapshot(101.0, 3.0, False)
    book.set_snapshot_seq(100)
    return book


# ── Test 1 ───────────────────────────────────────────────────────────────────
def test_best_bid_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    assert py.best_bid()  is not None
    assert cpp.best_bid() is not None

    # Python returns (price, size) tuple; C++ returns PriceLevel object
    assert py.best_bid()[0] == pytest.approx(cpp.best_bid().price)
    assert py.best_bid()[1] == pytest.approx(cpp.best_bid().size)


# ── Test 2 ───────────────────────────────────────────────────────────────────
def test_best_ask_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    assert py.best_ask()  is not None
    assert cpp.best_ask() is not None

    assert py.best_ask()[0] == pytest.approx(cpp.best_ask().price)
    assert py.best_ask()[1] == pytest.approx(cpp.best_ask().size)


# ── Test 3 ───────────────────────────────────────────────────────────────────
def test_spread_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    assert py.spread()  is not None
    assert cpp.spread() is not None

    assert py.spread() == pytest.approx(cpp.spread())


# ── Test 4 ───────────────────────────────────────────────────────────────────
def test_mid_price_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    assert py.mid_price()  is not None
    assert cpp.mid_price() is not None

    assert py.mid_price() == pytest.approx(cpp.mid_price())


# ── Test 5 ───────────────────────────────────────────────────────────────────
def test_update_overwrite_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    # Overwrite bid at 99.0 with size 10.0
    py.apply_update({"side": "bid", "price": 99.0, "size": 10.0,
                     "seq_start": 101, "seq": 101, "update_type": "update"})
    cpp.apply_update(99.0, 10.0, True, 101, 101)

    assert py.best_bid()[0] == pytest.approx(cpp.best_bid().price)
    assert py.best_bid()[1] == pytest.approx(cpp.best_bid().size)


# ── Test 6 ───────────────────────────────────────────────────────────────────
def test_delete_level_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    # Delete the only bid level (size=0, update_type=delete)
    py.apply_update({"side": "bid", "price": 99.0, "size": 0.0,
                     "seq_start": 101, "seq": 101, "update_type": "delete"})
    cpp.apply_update(99.0, 0.0, True, 101, 101)

    assert py.best_bid()  is None
    assert cpp.best_bid() is None


# ── Test 7 ───────────────────────────────────────────────────────────────────
def test_sequence_gap_agrees():
    py  = make_py_book()
    cpp = make_cpp_book()

    # Consecutive update — no gap expected
    py.apply_update({"side": "bid", "price": 99.0, "size": 6.0,
                     "seq_start": 101, "seq": 101, "update_type": "update"})
    cpp.apply_update(99.0, 6.0, True, 101, 101)

    assert py.sequence_gaps == cpp.sequence_gaps

    # Gap: skips seq 102 — both should now show 1 gap
    py.apply_update({"side": "bid", "price": 99.0, "size": 7.0,
                     "seq_start": 103, "seq": 103, "update_type": "update"})
    cpp.apply_update(99.0, 7.0, True, 103, 103)

    assert py.sequence_gaps == cpp.sequence_gaps
