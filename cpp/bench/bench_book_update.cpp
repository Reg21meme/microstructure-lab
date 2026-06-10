#include <benchmark/benchmark.h>
#include "mslab/book.hpp"
#include <vector>
#include <random>

// ── Shared test data ─────────────────────────────────────────────────────────
// Generate N price/size/side tuples once; reused across all benchmarks.
// Prices cluster around 100.0 in 0.01 increments across 20 levels per side,
// matching realistic L2 book depth without requiring real data.

struct UpdateRow {
    double  price;
    double  size;
    bool    is_bid;
    int64_t seq;
};

static std::vector<UpdateRow> make_updates(int n) {
    std::mt19937 rng(42); // fixed seed — deterministic
    std::uniform_int_distribution<int>  level_dist(0, 19);  // 20 levels per side
    std::uniform_real_distribution<>    size_dist(0.1, 10.0);
    std::bernoulli_distribution         side_dist(0.5);

    std::vector<UpdateRow> rows;
    rows.reserve(n);

    for (int i = 0; i < n; ++i) {
        bool is_bid = side_dist(rng);
        int  level  = level_dist(rng);

        // Bids cluster below 100, asks cluster above 100
        double base  = is_bid ? 99.0 : 101.0;
        double price = base - (is_bid ? 1 : -1) * level * 0.01;

        rows.push_back({price, size_dist(rng), is_bid, int64_t(i)});
    }
    return rows;
}

// Pre-generate 1M updates at static init time — shared across all benchmarks
static const std::vector<UpdateRow> kUpdates = make_updates(1'000'000);

// ── Benchmark 1: single apply_update call ────────────────────────────────────
// Measures the per-update hot-path latency.
// Reports: time/iteration (ns), which gives you p50/p95/p99 via --benchmark_repetitions.

static void BM_BookUpdate(benchmark::State& state) {
    mslab::OrderBook book("BTCUSDT");

    // Load a minimal snapshot so the book is in a valid state
    for (int i = 0; i < 20; ++i) {
        book.apply_snapshot(99.0 - i * 0.01, 1.0, true);
        book.apply_snapshot(101.0 + i * 0.01, 1.0, false);
    }
    book.set_snapshot_seq(0);

    int64_t idx = 0;
    int64_t n   = static_cast<int64_t>(kUpdates.size());

    for (auto _ : state) {
        const auto& u = kUpdates[idx % n];
        book.apply_update(u.price, u.size, u.is_bid, idx, idx);
        benchmark::DoNotOptimize(book);

        ++idx;
    }

    state.SetItemsProcessed(state.iterations());
}

// ── Benchmark 2: replay throughput ───────────────────────────────────────────
// Measures how many updates/sec the book can process in a batch.
// Reports: items/sec = replay throughput.

static void BM_BookReplay(benchmark::State& state) {
    const int64_t batch = state.range(0); // parameterised batch size

    mslab::OrderBook book("BTCUSDT");

    for (int i = 0; i < 20; ++i) {
        book.apply_snapshot(99.0 - i * 0.01, 1.0, true);
        book.apply_snapshot(101.0 + i * 0.01, 1.0, false);
    }
    book.set_snapshot_seq(0);

    int64_t n = static_cast<int64_t>(kUpdates.size());

    for (auto _ : state) {
        for (int64_t i = 0; i < batch; ++i) {
            const auto& u = kUpdates[i % n];
            book.apply_update(u.price, u.size, u.is_bid, i, i);
            benchmark::DoNotOptimize(book);
        }
    }

    state.SetItemsProcessed(state.iterations() * batch);
}

BENCHMARK(BM_BookUpdate)
    ->Repetitions(5)          // run 5 times to get stable statistics
    ->ReportAggregatesOnly(false); // show all repetitions + mean/median/stddev

BENCHMARK(BM_BookReplay)
    ->Arg(100'000)            // 100K batch
    ->Arg(1'000'000)          // 1M batch
    ->Repetitions(3)
    ->ReportAggregatesOnly(true); // just show mean/median/stddev for replay

BENCHMARK_MAIN();