#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "mslab/execution_sim.hpp"

using namespace mslab;
using Catch::Approx;

static ExecutionSim make_sim() {
    LatencyModel latency(0.0, 0.0);
    FeeModel     fees{0.0, 0.0};
    RiskLimits   limits{100.0, 1'000'000.0};
    return ExecutionSim("BTCUSDT", latency, fees, limits);
}

static void load_book(ExecutionSim& sim) {
    sim.on_book_update(99.0,  5.0, true,  1'000'000, 1, 1);
    sim.on_book_update(101.0, 3.0, false, 1'000'000, 1, 1);
}

TEST_CASE("buy limit order fills when ask price <= order price", "[sim]") {
    auto sim = make_sim();
    load_book(sim);
    sim.submit_order(Side::BUY, OrderType::LIMIT, 101.0, 1.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);
    REQUIRE(sim.fills().size() == 1);
    CHECK(sim.fills()[0].fill_price == Approx(101.0));
    CHECK(sim.fills()[0].fill_size  == Approx(1.0));
    CHECK(sim.fills()[0].side       == Side::BUY);
}

TEST_CASE("sell limit order fills when bid price >= order price", "[sim]") {
    auto sim = make_sim();
    load_book(sim);
    sim.submit_order(Side::SELL, OrderType::LIMIT, 99.0, 1.0, 100);
    sim.on_book_update(99.0, 5.0, true, 2'000'000, 2, 2);
    REQUIRE(sim.fills().size() == 1);
    CHECK(sim.fills()[0].fill_price == Approx(99.0));
    CHECK(sim.fills()[0].fill_size  == Approx(1.0));
    CHECK(sim.fills()[0].side       == Side::SELL);
}

TEST_CASE("order does not fill when price does not cross", "[sim]") {
    auto sim = make_sim();
    load_book(sim);
    sim.submit_order(Side::BUY, OrderType::LIMIT, 100.0, 1.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);
    CHECK(sim.fills().empty());
}

TEST_CASE("latency delays order arrival", "[sim]") {
    LatencyModel latency(10.0, 0.0); // 10ms
    FeeModel     fees{0.0, 0.0};
    RiskLimits   limits{100.0, 1'000'000.0};
    ExecutionSim sim("BTCUSDT", latency, fees, limits);
    load_book(sim);

    sim.submit_order(Side::BUY, OrderType::LIMIT, 101.0, 1.0, 0);

    // 5ms later — order not arrived yet (arrives at 10ms)
    sim.on_book_update(101.0, 3.0, false, 5'000'000, 2, 2);
    CHECK(sim.fills().empty());

    // 15ms later — order has arrived
    sim.on_book_update(101.0, 3.0, false, 15'000'000, 3, 3);
    REQUIRE(sim.fills().size() == 1);
}

TEST_CASE("position updates correctly after buy fill", "[sim]") {
    auto sim = make_sim();
    load_book(sim);
    sim.submit_order(Side::BUY, OrderType::LIMIT, 101.0, 2.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);
    CHECK(sim.position().position  == Approx(2.0));
    CHECK(sim.position().avg_entry == Approx(101.0));
}

TEST_CASE("realized PnL computed correctly on round trip", "[sim]") {
    auto sim = make_sim();
    load_book(sim);

    sim.submit_order(Side::BUY, OrderType::LIMIT, 101.0, 1.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);

    sim.on_book_update(103.0, 5.0, true, 3'000'000, 3, 3);

    sim.submit_order(Side::SELL, OrderType::LIMIT, 103.0, 1.0, 200);
    sim.on_book_update(103.0, 5.0, true, 4'000'000, 4, 4);

    CHECK(sim.fills().size()          == 2);
    CHECK(sim.position().position     == Approx(0.0).margin(1e-9));
    CHECK(sim.position().realized_pnl == Approx(2.0));
}

TEST_CASE("fees are applied correctly", "[sim]") {
    LatencyModel latency(0.0, 0.0);
    FeeModel     fees{0.0, 0.0004};
    RiskLimits   limits{100.0, 1'000'000.0};
    ExecutionSim sim("BTCUSDT", latency, fees, limits);
    load_book(sim);

    sim.submit_order(Side::BUY, OrderType::LIMIT, 101.0, 1.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);

    REQUIRE(sim.fills().size() == 1);
    CHECK(sim.fills()[0].fee      == Approx(0.0404));
    CHECK(sim.position().fee_drag == Approx(0.0404));
}

TEST_CASE("IOC order cancels unfilled remainder", "[sim]") {
    auto sim = make_sim();
    load_book(sim);
    sim.submit_order(Side::BUY, OrderType::IOC, 101.0, 10.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);
    REQUIRE(sim.fills().size() == 1);
    CHECK(sim.fills()[0].fill_size == Approx(3.0));
}

TEST_CASE("risk kill switch triggers on max drawdown", "[sim]") {
    LatencyModel latency(0.0, 0.0);
    FeeModel     fees{0.0, 0.0};
    RiskLimits   limits{100.0, 1.0};
    ExecutionSim sim("BTCUSDT", latency, fees, limits);

    sim.on_book_update(101.0, 5.0, false, 1'000'000, 1, 1);
    sim.submit_order(Side::BUY, OrderType::LIMIT, 101.0, 1.0, 100);
    sim.on_book_update(101.0, 5.0, false, 2'000'000, 2, 2);

    sim.on_book_update(99.0, 5.0, true, 3'000'000, 3, 3);
    sim.submit_order(Side::SELL, OrderType::LIMIT, 99.0, 1.0, 200);
    sim.on_book_update(99.0, 5.0, true, 4'000'000, 4, 4);

    CHECK(sim.is_killed() == true);
}

TEST_CASE("cancel order removes it from active orders", "[sim]") {
    auto sim = make_sim();
    load_book(sim);

    uint64_t id = sim.submit_order(Side::BUY, OrderType::LIMIT,
                                    100.0, 1.0, 100);
    sim.on_book_update(101.0, 3.0, false, 2'000'000, 2, 2);
    CHECK(sim.fills().empty());

    bool cancelled = sim.cancel_order(id);
    CHECK(cancelled == true);

    sim.on_book_update(100.0, 3.0, false, 3'000'000, 3, 3);
    CHECK(sim.fills().empty());
}