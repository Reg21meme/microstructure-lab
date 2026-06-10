#pragma once

#include <map>
#include <optional>
#include <string>
#include <cstdint>

namespace mslab {

// Represents one side of the order book (bids or asks)
// Uses std::map so prices are always sorted
// bids: highest price first (reverse order) | asks: lowest price first (natural order)

struct PriceLevel {
    double price;
    double size;
};

class OrderBook {
public:
    explicit OrderBook(std::string symbol);

    // Apply a full snapshot also clears existing state
    void apply_snapshot(double price, double size, bool is_bid);

    // Apply a single incremental update, size == 0.0 means delete that price level
    void apply_update(double price, double size, bool is_bid,
                      int64_t seq_start, int64_t seq_end);

    // Finalize snapshot loading — call after all snapshot rows applied
    void set_snapshot_seq(int64_t seq);

    // Clear the book
    void clear();

    // highest price buyers are willing to pay
    std::optional<PriceLevel> best_bid() const;

    // lowest price sellers are willing to accept
    std::optional<PriceLevel> best_ask() const;

    // Spread: best ask price - best bid price
    std::optional<double> spread() const;

    // halfway between best bid and ask
    std::optional<double> mid_price() const;

    // Getters for stats
    int64_t last_seq()       const { return last_seq_; }
    int     sequence_gaps()  const { return sequence_gaps_; }
    size_t  bid_levels()     const { return bids_.size(); }
    size_t  ask_levels()     const { return asks_.size(); }
    const std::string& symbol() const { return symbol_; }

private:
    std::string symbol_;

    // bids: sorted highest-first using reverse comparator
    std::map<double, double, std::greater<double>> bids_;

    // asks: sorted lowest-first (default map order)
    std::map<double, double> asks_;

    int64_t last_seq_      = -1;
    int     sequence_gaps_ =  0;
};

} // namespace mslab