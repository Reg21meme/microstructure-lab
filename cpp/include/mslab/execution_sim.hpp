#pragma once

#include "mslab/book.hpp"
#include "mslab/event.hpp"
#include "mslab/latency_model.hpp"
#include <vector>
#include <map>
#include <optional>
#include <cstdint>
#include <string>

namespace mslab {

// ── Order ────────────────────────────────────────────────────────────────────

struct Order {
    uint64_t  order_id;
    Side      side;
    OrderType type;
    double    price;
    double    size;
    double    remaining;       // size not yet filled
    int64_t   submit_ts_ns;    // when signal fired
    int64_t   arrive_ts_ns;    // submit_ts + latency — when order reaches book
    bool      active = true;   // false if filled, cancelled, or expired
};

// ── Fill ─────────────────────────────────────────────────────────────────────

struct Fill {
    uint64_t order_id;
    Side     side;
    double   fill_price;
    double   fill_size;
    double   fee;              // positive = cost, negative = rebate
    int64_t  fill_ts_ns;
    bool     is_maker;         // true if resting order (limit), false if taker
};

// ── Position tracking ────────────────────────────────────────────────────────

struct PositionSnapshot {
    double  position      = 0.0;   // net position (positive = long)
    double  avg_entry     = 0.0;   // volume-weighted average entry price
    double  realized_pnl  = 0.0;   // PnL from closed trades
    double  fee_drag      = 0.0;   // cumulative fees paid
    int64_t ts_ns         = 0;
};

// ── Risk limits ───────────────────────────────────────────────────────────────

struct RiskLimits {
    double max_position    = 1.0;   // max absolute position size
    double max_drawdown    = 500.0; // kill switch: max loss in dollars
    bool   killed          = false; // true if kill switch triggered
};

// ── Fee model ─────────────────────────────────────────────────────────────────

struct FeeModel {
    double maker_fee = -0.0001;  // rebate for providing liquidity (negative = receive)
    double taker_fee =  0.0004;  // cost for taking liquidity
};

// ── ExecutionSim ──────────────────────────────────────────────────────────────

class ExecutionSim {
public:
    explicit ExecutionSim(const std::string& symbol,
                          LatencyModel       latency  = LatencyModel(),
                          FeeModel           fees     = FeeModel(),
                          RiskLimits         limits   = RiskLimits());

    // Submit an order — latency is applied automatically
    // Returns the order_id assigned
    uint64_t submit_order(Side      side,
                          OrderType type,
                          double    price,
                          double    size,
                          int64_t   signal_ts_ns);

    // Process a single L2 book update
    // Checks if any resting orders can be filled
    void on_book_update(double  price,
                        double  size,
                        bool    is_bid,
                        int64_t ts_ns,
                        int64_t seq_start,
                        int64_t seq_end);

    // Cancel a resting order by ID
    bool cancel_order(uint64_t order_id);

    // Accessors
    const std::vector<Fill>&            fills()    const { return fills_; }
    const PositionSnapshot&             position() const { return position_; }
    const RiskLimits&                   limits()   const { return limits_; }
    const std::vector<PositionSnapshot> snapshots()const { return snapshots_; }
    int64_t                             last_ts()  const { return last_ts_ns_; }
    bool                                is_killed()const { return limits_.killed; }

    // Reset sim state (keep config)
    void reset();

private:
    std::string  symbol_;
    LatencyModel latency_;
    FeeModel     fees_;
    RiskLimits   limits_;
    OrderBook    book_;

    uint64_t                     next_order_id_ = 1;
    std::map<uint64_t, Order>    orders_;        // active resting orders
    std::vector<Fill>            fills_;
    PositionSnapshot             position_;
    std::vector<PositionSnapshot> snapshots_;
    int64_t                      last_ts_ns_ = 0;

    // Internal helpers
    void     try_fill_orders(int64_t ts_ns);
    void     apply_fill(Order& order, double fill_price,
                        double fill_size, int64_t ts_ns, bool is_maker);
    void     update_position(const Fill& fill);
    void     check_risk(int64_t ts_ns);
    double   compute_fee(double fill_size, double fill_price, bool is_maker);
    void     record_snapshot(int64_t ts_ns);
};

} // namespace mslab