#include "mslab/execution_sim.hpp"
#include <cmath>
#include <stdexcept>

namespace mslab {

ExecutionSim::ExecutionSim(const std::string& symbol,
                           LatencyModel       latency,
                           FeeModel           fees,
                           RiskLimits         limits)
    : symbol_(symbol)
    , latency_(latency)
    , fees_(fees)
    , limits_(limits)
    , book_(symbol)
{}

void ExecutionSim::reset() {
    orders_.clear();
    fills_.clear();
    snapshots_.clear();
    position_      = PositionSnapshot{};
    next_order_id_ = 1;
    last_ts_ns_    = 0;
    limits_.killed = false;
    book_.clear();
}

uint64_t ExecutionSim::submit_order(Side      side,
                                     OrderType type,
                                     double    price,
                                     double    size,
                                     int64_t   signal_ts_ns) {
    if (limits_.killed) return 0;
    if (size <= 0.0)    return 0;

    uint64_t id = next_order_id_++;

    Order order;
    order.order_id     = id;
    order.side         = side;
    order.type         = type;
    order.price        = price;
    order.size         = size;
    order.remaining    = size;
    order.submit_ts_ns = signal_ts_ns;
    order.arrive_ts_ns = signal_ts_ns + latency_.delay_ns();
    order.active       = true;

    orders_[id] = order;
    return id;
}

void ExecutionSim::on_book_update(double  price,
                                   double  size,
                                   bool    is_bid,
                                   int64_t ts_ns,
                                   int64_t seq_start,
                                   int64_t seq_end) {
    last_ts_ns_ = ts_ns;
    book_.apply_update(price, size, is_bid, seq_start, seq_end);
    try_fill_orders(ts_ns);
    record_snapshot(ts_ns);
}

bool ExecutionSim::cancel_order(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return false;
    it->second.active = false;
    orders_.erase(it);
    return true;
}

void ExecutionSim::try_fill_orders(int64_t ts_ns) {
    if (limits_.killed) return;

    auto best_bid = book_.best_bid();
    auto best_ask = book_.best_ask();

    for (auto& [id, order] : orders_) {
        if (!order.active)           continue;
        if (order.arrive_ts_ns > ts_ns) continue; // not arrived yet

        bool filled = false;

        if (order.side == Side::BUY) {
            // Buy order fills when best ask <= order price
            if (!best_ask) continue;
            if (best_ask->price <= order.price) {
                double fill_price = best_ask->price;
                double fill_size  = std::min(order.remaining,
                                             best_ask->size);
                bool is_maker = (order.type == OrderType::LIMIT &&
                                 order.price < best_ask->price);
                apply_fill(order, fill_price, fill_size, ts_ns, is_maker);
                filled = true;
            }
        } else {
            // Sell order fills when best bid >= order price
            if (!best_bid) continue;
            if (best_bid->price >= order.price) {
                double fill_price = best_bid->price;
                double fill_size  = std::min(order.remaining,
                                             best_bid->size);
                bool is_maker = (order.type == OrderType::LIMIT &&
                                 order.price > best_bid->price);
                apply_fill(order, fill_price, fill_size, ts_ns, is_maker);
                filled = true;
            }
        }

        // IOC: cancel unfilled remainder immediately
        if (order.type == OrderType::IOC && !filled) {
            order.active = false;
        }
        // POST_ONLY: cancel if it would take liquidity
        if (order.type == OrderType::POST_ONLY && filled) {
            // Already filled as taker — this shouldn't happen for POST_ONLY
            // Cancel the order (in reality the exchange would reject it)
            order.active = false;
        }
    }

    // Remove inactive orders
    for (auto it = orders_.begin(); it != orders_.end(); ) {
        if (!it->second.active || it->second.remaining <= 0.0)
            it = orders_.erase(it);
        else
            ++it;
    }

    check_risk(ts_ns);
}

void ExecutionSim::apply_fill(Order&  order,
                               double  fill_price,
                               double  fill_size,
                               int64_t ts_ns,
                               bool    is_maker) {
    double fee = compute_fee(fill_size, fill_price, is_maker);

    Fill f;
    f.order_id   = order.order_id;
    f.side       = order.side;
    f.fill_price = fill_price;
    f.fill_size  = fill_size;
    f.fee        = fee;
    f.fill_ts_ns = ts_ns;
    f.is_maker   = is_maker;

    fills_.push_back(f);
    order.remaining -= fill_size;
    if (order.remaining <= 1e-10) order.active = false;

    update_position(f);
}

void ExecutionSim::update_position(const Fill& fill) {
    double signed_size = (fill.side == Side::BUY)
                         ? fill.fill_size : -fill.fill_size;

    double old_pos = position_.position;
    double new_pos = old_pos + signed_size;

    // Update average entry price
    if (std::abs(new_pos) > 1e-10) {
        if (std::abs(old_pos) < 1e-10 ||
            (old_pos > 0 && signed_size > 0) ||
            (old_pos < 0 && signed_size < 0)) {
            // Adding to position — update average entry
            double old_value  = std::abs(old_pos) * position_.avg_entry;
            double new_value  = fill.fill_size * fill.fill_price;
            position_.avg_entry = (old_value + new_value) /
                                   std::abs(new_pos);
        }
    }

    // Realized PnL on position reduction
    if ((old_pos > 0 && signed_size < 0) ||
        (old_pos < 0 && signed_size > 0)) {
        double closed = std::min(std::abs(old_pos), fill.fill_size);
        double pnl    = (fill.side == Side::SELL)
                        ? closed * (fill.fill_price - position_.avg_entry)
                        : closed * (position_.avg_entry - fill.fill_price);
        position_.realized_pnl += pnl;
    }

    position_.position  = new_pos;
    position_.fee_drag += fill.fee;
    position_.ts_ns     = fill.fill_ts_ns;
}

void ExecutionSim::check_risk(int64_t ts_ns) {
    // Max position check
    if (std::abs(position_.position) > limits_.max_position) {
        limits_.killed = true;
        return;
    }
    // Max drawdown check
    double total_pnl = position_.realized_pnl - position_.fee_drag;
    if (total_pnl < -limits_.max_drawdown) {
        limits_.killed = true;
    }
}

double ExecutionSim::compute_fee(double fill_size,
                                  double fill_price,
                                  bool   is_maker) {
    double rate   = is_maker ? fees_.maker_fee : fees_.taker_fee;
    return fill_size * fill_price * rate;
}

void ExecutionSim::record_snapshot(int64_t ts_ns) {
    // Record every 100th update to avoid massive memory usage
    if (snapshots_.empty() ||
        ts_ns - snapshots_.back().ts_ns > 100'000'000) { // 100ms in ns
        PositionSnapshot snap = position_;
        snap.ts_ns = ts_ns;
        snapshots_.push_back(snap);
    }
}

} // namespace mslab