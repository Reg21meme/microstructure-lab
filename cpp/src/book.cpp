#include "mslab/book.hpp"
#include <optional>

namespace mslab {

OrderBook::OrderBook(std::string symbol)
    : symbol_(std::move(symbol))
    , last_seq_(-1)
    , sequence_gaps_(0)
{}

void OrderBook::clear() {
    bids_.clear();
    asks_.clear();
    last_seq_      = -1;
    sequence_gaps_ = 0;
}

void OrderBook::apply_snapshot(double price, double size, bool is_bid) {
    if (is_bid) {
        bids_[price] = size;
    } else {
        asks_[price] = size;
    }
}

void OrderBook::set_snapshot_seq(int64_t seq) {
    last_seq_ = seq;
}

void OrderBook::apply_update(double price, double size, bool is_bid,
                              int64_t seq_start, int64_t seq_end) {
    if (last_seq_ >= 0 && seq_start > last_seq_ + 1) {
        ++sequence_gaps_;
    }
    if (is_bid) {
        if (size == 0.0) {
            bids_.erase(price);
        } else {
            bids_[price] = size;
        }
    } else {
        if (size == 0.0) {
            asks_.erase(price);
        } else {
            asks_[price] = size;
        }
    }
    last_seq_ = seq_end;
}

std::optional<PriceLevel> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    auto it = bids_.begin();
    return PriceLevel{it->first, it->second};
}

std::optional<PriceLevel> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    auto it = asks_.begin();
    return PriceLevel{it->first, it->second};
}

std::optional<double> OrderBook::spread() const {
    auto bid = best_bid();
    auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return ask->price - bid->price;
}

std::optional<double> OrderBook::mid_price() const {
    auto bid = best_bid();
    auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return (bid->price + ask->price) / 2.0;
}

} // namespace mslab