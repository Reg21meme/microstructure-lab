# MicrostructureLab

C++/Python market-microstructure research and execution-simulation system on real L2 crypto data.


## What this demonstrates

| Requirement | Evidence |
|---|---|
| C++ in trading/research infra | C++20 order-book replay engine, zero hot-path allocations |
| Large-dataset research | Tick/L2 stored as Parquet, queried with DuckDB/Polars |
| Backtesting & model testing | Event-driven backtest with fees, latency, fill logic (Week 5) |
| Probability, statistics, ML | OFI features, purged/embargoed CV, Newey-West errors, IC decay |
| Low-latency / high-performance | Benchmarked C++ hot path: p50/p95/p99, ~34M updates/sec |
| Production-quality code | CI, Catch2 + pytest, Docker, reproducible pipeline |
| Market microstructure | L2 reconstruction, queue modeling, adverse selection, maker/taker |

---

## TL;DR results

### Systems (C++ engine)

| Benchmark | Median | Stddev | CV |
|---|---|---|---|
| Book update latency | 28.8 ns | 0.14 ns | 0.49% |
| Replay throughput (1M batch) | ~34M updates/sec | — | 0.74% |

> Measured on 22-core 3072 MHz laptop. Google Benchmark apt package is debug build — timings slightly pessimistic. Book code compiled with `-O3`.

### Research (alpha models) — 30-minute BTCUSDT sample

| Metric | Value | Notes |
|---|---|---|
| Ridge test IC | 0.29 | Simple 70/30 time split |
| Walk-forward mean IC | 0.32 ± 0.04 | Purged/embargoed, 5 folds |
| Walk-forward Rank-IC | 0.41 ± 0.06 | Spearman |
| Newey-West t-stat (5s) | 39.7 | Inflated — see limitations |
| Logistic test AUC | 0.84 | Binary direction prediction |
| Brier skill score | 0.30 | 30% improvement over naive |
| IC decay | 0.32 → 0.29 → 0.26 | 5s → 10s → 20s horizon |

> **Honest caveat:** results reflect a 30-minute trending BTCUSDT period.
> IC and AUC are expected to be significantly lower on a full day with diverse regimes.
> Full-day Tardis historical data evaluation in progress.

### Execution simulation — in progress 
- Naive (no fees, no latency) vs honest (queue + latency + fees) PnL comparison coming
- Headline chart: PnL gap between vectorized backtest and realistic execution

---

## Architecture
Raw L2 (Binance WebSocket) → synchronized snapshot + updates (collector.py) → normalized Parquet (ts_local, seq, side, price, size) → C++ book replay (mslab_bindings — 34M updates/sec) → feature snapshots (microstructure.py) → Parquet feature store (25 columns) → baseline models (train_baseline.py) → purged walk-forward CV (dataset.py) → IC decay + Newey-West (validate.py) → execution simulator → PnL / risk / latency report

---

## Quickstart — reproduce in 5 commands

```bash
# 1. Build
git clone https://github.com/YOUR_USERNAME/microstructure-lab
cd microstructure-lab
cmake -S . -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build

# 2. Install Python deps
pip install -e ".[dev]"

# 3. Collect data (30 seconds of live L2)
python3 -m mslab.ingest.collector

# 4. Build features
python3 -m mslab.features.build

# 5. Train and validate
python3 -m mslab.models.train_baseline
python3 -m mslab.models.validate
```

---

## Methodology — read this before judging the numbers

**Time-based splits only.** No random shuffling. Ever.

**Purged + embargoed walk-forward CV** (López de Prado, 2018). Labels for row T use data from rows T+1 to T+horizon. Without purging, those rows contaminate training. Without embargo, autocorrelation between adjacent rows leaks across the split boundary. Both corrections applied.

**Newey-West HAC standard errors** for IC t-statistics. Raw t-stats assume fold ICs are independent — they're not, because market regimes persist. HAC correction accounts for autocorrelation in the IC series.

**Deflated Sharpe Ratio** (Bailey & López de Prado, 2014). Corrects for multiple testing. A Sharpe found after testing 100 configurations is not the same as a Sharpe found on the first try.

**IC decay curve.** IC computed at 5s, 10s, and 20s horizons. A real microstructure signal decays toward zero as horizon increases. A signal that stays high at long horizons is capturing trend, not microstructure.

---

## What breaks the strategy

1. **Market regime change.** Features trained on a trending period have lower IC on choppy or mean-reverting days. The 30-minute sample is a single regime.

2. **Latency budget.** At 50ms latency the signal edge degrades significantly. At 100ms it may disappear entirely. Ablation results in Week 5.

3. **Fee drag.** At $0.01 spread and Binance taker fees of ~0.04% per side, a strategy needs >$5 edge per trade to be net profitable after fees on a $63,000 BTC position. Most signals don't clear this bar.

4. **Queue position.** Limit orders are not guaranteed fills. Fill probability depends on queue depth ahead. A strategy that assumes 100% fill rate overstates PnL by an amount that depends on how deep in the queue you typically sit.

5. **Overfitting to BTCUSDT.** The pipeline is designed for one symbol. Transfer to other symbols or asset classes is untested.

---

## Limitations and honesty notes

- This is a **research and simulation system**, not a live trading system
- Results on 30-minute sample are not statistically meaningful for production evaluation
- L2 cannot perfectly identify queue position — approximation used
- Crypto microstructure may not transfer to equities
- IC of 0.32 on a trending 30-minute period ≠ IC of 0.32 in production
- Full evaluation pending Tardis historical data (full trading day)

---

## Validation

- C++ book: 10 Catch2 unit tests, all passing
- Python/C++ parity: 7 pybind11 parity tests, all passing
- CI: GitHub Actions — builds C++, runs all tests, runs benchmark smoke test on every push

---

## License

MIT