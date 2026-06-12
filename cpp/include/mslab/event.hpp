#pragma once

#include <cstdint>
#include <string>

namespace mslab {

// Every occurrence in the simulator is typed as one of these
enum class EventType : uint8_t {
    BOOK_UPDATE   = 0,  // L2 book update — always processed before orders at same ts
    ORDER_SUBMIT  = 1,  // new order entering the sim
    ORDER_CANCEL  = 2,  // cancel a resting order
};

// Side of the market
enum class Side : uint8_t {
    BUY  = 0,
    SELL = 1,
};

// Order types supported by the matching engine
enum class OrderType : uint8_t {
    LIMIT      = 0,  // rests in book at specified price
    IOC        = 1,  // immediate-or-cancel: fill what you can, cancel rest
    POST_ONLY  = 2,  // limit order that cancels if it would take liquidity
};

// A single event in the replay timeline
struct Event {
    int64_t   ts_ns;        // timestamp in nanoseconds
    EventType type;         // what kind of event this is

    // For BOOK_UPDATE events
    double    price  = 0.0;
    double    size   = 0.0;
    bool      is_bid = false;
    int64_t   seq    = -1;

    // For ORDER_SUBMIT / ORDER_CANCEL events
    uint64_t  order_id   = 0;
    Side      side       = Side::BUY;
    OrderType order_type = OrderType::LIMIT;
    double    order_price = 0.0;
    double    order_size  = 0.0;

    // Deterministic tie-breaking: lower EventType value = processed first
    // BOOK_UPDATE (0) always before ORDER_SUBMIT (1) at same timestamp
    bool operator>(const Event& other) const {
        if (ts_ns != other.ts_ns) return ts_ns > other.ts_ns;
        return static_cast<uint8_t>(type) > static_cast<uint8_t>(other.type);
    }
};

} // namespace mslab