#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "mslab/book.hpp"

using namespace mslab;
using Catch::Approx;

// Helper: build a clean two-sided book with one bid and one ask
static OrderBook make_simple_book() {
    OrderBook book("BTCUSDT");
    book.clear();
    book.apply_snapshot(99.0, 5.0, true);   // bid at 99
    book.apply_snapshot(101.0, 3.0, false); // ask at 101
    book.set_snapshot_seq(100);
    return book;
}

// ── Test 1 ──────────────────────────────────────────────────────────────────
TEST_CASE("empty book returns nullopt", "[book]") {
    OrderBook book("BTCUSDT");

    REQUIRE_FALSE(book.best_bid().has_value());
    REQUIRE_FALSE(book.best_ask().has_value());
    REQUIRE_FALSE(book.spread().has_value());
    REQUIRE_FALSE(book.mid_price().has_value());
}

// ── Test 2 ──────────────────────────────────────────────────────────────────
TEST_CASE("snapshot populates both sides correctly", "[book]") {
    OrderBook book = make_simple_book();

    REQUIRE(book.best_bid().has_value());
    REQUIRE(book.best_ask().has_value());

    CHECK(book.best_bid()->price == Approx(99.0));
    CHECK(book.best_bid()->size  == Approx(5.0));
    CHECK(book.best_ask()->price == Approx(101.0));
    CHECK(book.best_ask()->size  == Approx(3.0));
}

// ── Test 3 ──────────────────────────────────────────────────────────────────
TEST_CASE("best_bid returns highest bid price", "[book]") {
    OrderBook book("BTCUSDT");
    book.clear();
    book.apply_snapshot(95.0, 1.0, true);
    book.apply_snapshot(97.0, 2.0, true);
    book.apply_snapshot(99.0, 3.0, true);
    book.set_snapshot_seq(100);

    REQUIRE(book.best_bid().has_value());
    CHECK(book.best_bid()->price == Approx(99.0));
    CHECK(book.bid_levels() == 3);
}

// ── Test 4 ──────────────────────────────────────────────────────────────────
TEST_CASE("best_ask returns lowest ask price", "[book]") {
    OrderBook book("BTCUSDT");
    book.clear();
    book.apply_snapshot(101.0, 1.0, false);
    book.apply_snapshot(103.0, 2.0, false);
    book.apply_snapshot(105.0, 3.0, false);
    book.set_snapshot_seq(100);

    REQUIRE(book.best_ask().has_value());
    CHECK(book.best_ask()->price == Approx(101.0));
    CHECK(book.ask_levels() == 3);
}

// ── Test 5 ──────────────────────────────────────────────────────────────────
TEST_CASE("spread and mid_price are correct after valid snapshot", "[book]") {
    OrderBook book = make_simple_book();
    // bid=99, ask=101 → spread=2, mid=100

    REQUIRE(book.spread().has_value());
    REQUIRE(book.mid_price().has_value());

    CHECK(*book.spread()    == Approx(2.0));  // ask - bid
    CHECK(*book.mid_price() == Approx(100.0));
    CHECK(*book.spread() > 0.0); // book is never crossed
}

// ── Test 6 ──────────────────────────────────────────────────────────────────
TEST_CASE("update overwrites size at existing price level", "[book]") {
    OrderBook book = make_simple_book();

    // Overwrite bid at 99.0 with new size 10.0
    book.apply_update(99.0, 10.0, true, 101, 101);

    REQUIRE(book.best_bid().has_value());
    CHECK(book.best_bid()->price == Approx(99.0));
    CHECK(book.best_bid()->size  == Approx(10.0));
    CHECK(book.bid_levels() == 1); // still one level, just updated
}

// ── Test 7 ──────────────────────────────────────────────────────────────────
TEST_CASE("update with size zero removes price level", "[book]") {
    OrderBook book = make_simple_book();
    CHECK(book.bid_levels() == 1);

    // Delete the only bid level
    book.apply_update(99.0, 0.0, true, 101, 101);

    CHECK(book.bid_levels() == 0);
    REQUIRE_FALSE(book.best_bid().has_value());
}

// ── Test 8 ──────────────────────────────────────────────────────────────────
TEST_CASE("update at new price inserts a new level", "[book]") {
    OrderBook book = make_simple_book();
    CHECK(book.bid_levels() == 1);

    // Add a new bid level at 98.0
    book.apply_update(98.0, 4.0, true, 101, 101);

    CHECK(book.bid_levels() == 2);
    // best bid should still be 99.0, not the new 98.0
    REQUIRE(book.best_bid().has_value());
    CHECK(book.best_bid()->price == Approx(99.0));
}

// ── Test 9 ──────────────────────────────────────────────────────────────────
TEST_CASE("sequence gap is detected", "[book]") {
    OrderBook book = make_simple_book();
    // last_seq_ is 100 after set_snapshot_seq(100)
    // consecutive update: seq 101→101, no gap
    book.apply_update(99.0, 6.0, true, 101, 101);
    CHECK(book.sequence_gaps() == 0);

    // gap: seq 103→103, skips 102
    book.apply_update(99.0, 7.0, true, 103, 103);
    CHECK(book.sequence_gaps() == 1);
}

// ── Test 10 ─────────────────────────────────────────────────────────────────
TEST_CASE("no gap detected for consecutive sequence numbers", "[book]") {
    OrderBook book = make_simple_book();
    // Feed a chain of consecutive updates, no gaps should be counted
    book.apply_update(99.0, 6.0, true, 101, 101);
    book.apply_update(99.0, 7.0, true, 102, 102);
    book.apply_update(99.0, 8.0, true, 103, 103);

    CHECK(book.sequence_gaps() == 0);
}