#pragma once

#include <cstdint>
#include <random>
#include <algorithm>

namespace mslab {

class LatencyModel {
public:
    // Parameters
    // base_ms   : fixed base latency in milliseconds (e.g. 10.0)
    // jitter_ms : standard deviation of random jitter (e.g. 2.0)
    // seed      : random seed for reproducibility
    explicit LatencyModel(double base_ms   = 10.0,
                          double jitter_ms = 0.0,
                          uint64_t seed    = 42)
        : base_ms_(base_ms)
        , jitter_ms_(jitter_ms)
        , rng_(seed)
        , dist_(0.0, jitter_ms > 0 ? jitter_ms : 1.0)
    {}

    // Returns total latency in milliseconds for one order submission
    // Always >= base_ms (jitter is clamped to zero minimum)
    double delay_ms() {
        if (jitter_ms_ <= 0.0) return base_ms_;
        double jitter = dist_(rng_);
        return base_ms_ + std::max(0.0, jitter);
    }

    // Returns total latency in nanoseconds
    int64_t delay_ns() {
        return static_cast<int64_t>(delay_ms() * 1'000'000);
    }

    double base_ms()   const { return base_ms_; }
    double jitter_ms() const { return jitter_ms_; }

    void set_base_ms(double ms)   { base_ms_   = ms; }
    void set_jitter_ms(double ms) { jitter_ms_ = ms; }

private:
    double base_ms_;
    double jitter_ms_;
    std::mt19937_64            rng_;
    std::normal_distribution<> dist_;
};

} // namespace mslab