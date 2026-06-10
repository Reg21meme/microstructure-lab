# microstructure-lab
C++/Python market-microstructure research and execution-simulation system on real L2 crypto data.

## What this demonstrates
- C++20 low-latency order-book replay + matching engine
- Python research pipeline for microstructure alpha (OFI, depth imbalance, micro-price)
- Event-driven execution simulator with fees, latency, and queue-aware fill modeling
- Purged/embargoed walk-forward validation with deflated Sharpe
- Honest PnL / risk / latency decomposition

## TL;DR results (fill in as project progresses)
- Engine: **~34M events/sec**, book-update latency median **~29ns** (stddev 0.14ns), measured on 1M updates
- Signal: _in progress_
- Strategy: _in progress_

## Systems benchmark (Week 2)

| Benchmark | Median | Stddev | CV |
|---|---|---|---|
| Book update latency | 28.8 ns | 0.14 ns | 0.49% |
| Replay throughput (100K batch) | 36.0 M updates/sec | — | 0.53% |
| Replay throughput (1M batch) | 34.4 M updates/sec | — | 0.74% |

> Benchmark note: Google Benchmark apt package is a debug build; timings are slightly pessimistic.
> Book code compiled with `-O3`. Hardware: 22-core 3072 MHz, 24 MB L3 cache.

## Architecture
Raw L2 → normalized Parquet → C++ book replay → features → models → execution sim → reports


## Limitations & honesty notes
- L2 cannot perfectly identify queue position; queue approximation used
- Crypto microstructure may not transfer directly to equities
- This is a research and simulation system, not a live trading claim