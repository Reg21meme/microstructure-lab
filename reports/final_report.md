# MicrostructureLab — Final Report

**Date:** June 2025  
**Data:** BTCUSDT, June 1 2025, Tardis historical L2 (19.4M updates)  
**System:** C++20 + Python 3.12, WSL Ubuntu 24.04  

---

## 1. Summary

MicrostructureLab is a C++/Python system for market microstructure
research and execution simulation on real L2 crypto order-book data.
It ingests Binance L2 updates, reconstructs the order book in a C++
engine, computes microstructure features, trains interpretable alpha
models, and evaluates them through a C++ event-driven simulator with
realistic latency, fees, and queue-position modeling.

The central finding: the signal has genuine predictive power
(IC = 0.26 at 5s horizon, Newey-West t = 54.5, deflated Sharpe = 1.0)
but is unprofitable at retail fee levels. Breakeven requires taker fees
below 0.73 bps; Binance VIP0 is 4 bps (5.5× above breakeven). The
naive backtest overstates PnL by $6,981 on a single trading day —
the gap between vectorized and realistic execution is the headline result.

---

## 2. System performance

The C++ order book engine processes 34M events/sec with 28.8ns median
update latency and zero hot-path allocations. This was measured with
Google Benchmark on a 22-core 3072 MHz laptop compiled with -O3.
The engine supports add/update/delete operations with price-time
priority, sequence gap detection, and IOC/LIMIT/POST_ONLY order types.

The execution simulator replays 19.4M L2 updates in approximately
12 minutes on the same hardware, applying a 10ms latency model,
Binance VIP0 fee schedule, and a queue-position fill probability
approximation derived from displayed depth imbalance.

---

## 3. Signal research

### Features
Six microstructure features computed at 1-second intervals:
depth imbalance (top 5 levels), micro-price deviation (Stoikov 2018),
OFI (Cont-Kukanov-Stoikov 2014), MLOFI PC1 (10-level PCA),
realized volatility (20-snapshot window), and spread.

### Model
Ridge regression (α=1.0) on standardized features, predicting
5-second future mid-price move. Trained on first 70% of data by time,
evaluated on last 30%.

### Validation results

| Metric | Value |
|---|---|
| Test IC (5s) | 0.258 |
| Newey-West t-stat (5s) | 54.5 |
| IC at 10s | 0.260 |
| IC at 20s | 0.244 |
| Brier skill score | 0.302 |
| Deflated Sharpe | 1.000 |

IC decays from 0.267 to 0.244 as horizon increases from 5s to 20s,
consistent with a short-horizon microstructure signal rather than
trend following.

### Feature ablation
On full-day data, depth_imbalance_5 carries 95.2% of the IC signal.
Removing it drops IC from 0.258 to 0.012. This contrasts with a
30-minute trending sample where micro_price_deviation was dominant
(10.9% drop when removed). Feature importance is regime-dependent:
depth imbalance captures mean-reverting microstructure while
micro-price deviation captures trending momentum.

OFI has near-zero marginal contribution (0.1% IC drop when removed),
despite being the theoretically motivated Cont-Kukanov-Stoikov
feature. On a full day with mixed regimes, book-state features
dominate flow-based features at 1-second frequency.

### Symbol generalization
IC generalizes to ETHUSDT (0.31 vs 0.258 on BTCUSDT). Naive PnL
on ETHUSDT is near zero, confirming the BTCUSDT naive result was
partly driven by directional market movement rather than pure signal.

---

## 4. Execution simulation

### Setup
- Signal: ridge model predictions, threshold $0.10
- Orders: LIMIT orders at mid ± $0.01 (aggressive, always taker)
- Latency: 10ms fixed
- Fees: Binance VIP0 (taker 4 bps, maker −1 bp)
- Queue model: fill probability from depth imbalance, min 10% floor
- Position limit: 10 BTC, drawdown limit: $50,000

### Results

| Metric | Naive | Realistic |
|---|---|---|
| Net PnL | +$527.78 | -$6,453.30 |
| Gross PnL | +$527.78 | +$1,439.70 |
| Fee drag | $0.00 | $7,893.00 |
| Fills | 5,377 | 19,196 |
| Orders submitted | 5,263 | 18,810 |
| PnL gap | — | $6,981.07 |

### Fee economics
Total notional traded: $19.7M. Average fee per fill: $0.41.
Average gross edge per fill: $0.075. Fee-to-edge ratio: 5.5×.
The strategy needs either institutional fee tiers (< 0.73 bps) or
a larger edge per trade to be net profitable.

### Adverse selection
Mean markout at 1s horizon: +$1.11/fill (t = 25.6, n = 19,194).
Positive markout confirms the signal is predictive and fills occur
before the anticipated price move. No adverse selection detected
on this sample.

### Queue model impact
The queue model rejected 25% of order submissions
(24,954 considered → 18,810 submitted), reducing fee drag
proportionally without significantly impacting gross PnL.
This confirms the queue model acts as a quality filter.

---

## 5. Robustness

### Fee sensitivity
Breakeven taker fee: 0.73 bps. Net PnL degrades linearly at
$1,973/bps above breakeven. At VIP0 (4 bps): -$6,453 net PnL.

### Latency sensitivity
Tested at 0, 10, 50, 100ms. Net PnL is insensitive to latency
at 1s signal frequency. The strategy is fee-dominated, not
latency-dominated. A latency race is not the right framing for
a 1s microstructure signal.

### Confidence threshold
Raising SIGNAL_THRESH from $0.05 to $1.00 reduces fills and fee
drag monotonically but does not produce positive net PnL at any
tested threshold on this sample. Gross PnL is insensitive to
threshold, suggesting the signal edge is uniformly distributed
across confidence levels — consistent with a linear ridge model.

---

## 6. Limitations

**Single trading day.** All execution results are on June 1 2025.
Market regime on that day (moderate directional movement, ~$1,000
intraday range) may not represent typical conditions. Multi-day
evaluation requires additional Tardis data.

**Fee dominance.** The signal is real but sub-threshold for retail
execution. This is an honest finding, not a failure — demonstrating
*why* a signal fails in execution is as valuable as finding one
that doesn't.

**Queue model approximation.** Fill probability from displayed
depth is a first-order approximation. Real queue position requires
L3 order-by-order data showing arrival time relative to other
participants.

**1-second resolution.** Markouts at 100ms and 500ms are aliased
to the 1s feature grid. Tick-level markouts require replaying the
book at each horizon separately.

**Crypto vs equities.** Crypto microstructure differs from
equity microstructure in tick size, maker-taker economics, and
participant composition. Transfer to equity L2 data requires
re-calibration.

---

## 7. What I would do next

1. **Multi-day evaluation.** Download 5–10 days of Tardis data,
   run the full pipeline on each, report mean and standard deviation
   of IC, net PnL, and fee drag across days.

2. **Maker order strategy.** Current orders are priced to cross
   the spread (always taker). Submitting passive limit orders at
   mid would earn the maker rebate (−1 bp at VIP0) and shift
   the breakeven fee from 0.73 bps to roughly 1.7 bps — a more
   achievable target.

3. **Higher-frequency signal.** At 1s frequency the signal earns
   $0.075/fill. At 100ms frequency with the same IC, fill count
   increases 10× while fees stay proportional — but if the signal
   decays faster than fees, this may not help.

4. **L3 queue model.** Replace the depth-imbalance fill probability
   with a model trained on actual L3 queue arrival data. This would
   make the fill rate estimate more accurate and reduce the gap
   between simulation and live execution.

5. **Cross-symbol alpha.** The signal generalizes to ETHUSDT
   in terms of IC. A portfolio approach trading both symbols
   with position sizing proportional to IC would diversify
   regime risk.

---

## 8. Reproduce

```bash
# Download Tardis data (free, no API key)
curl -L "https://datasets.tardis.dev/v1/binance/incremental_book_L2/2025/06/01/BTCUSDT.csv.gz" \
     -o data/raw/tardis/BTCUSDT_2025-06-01.csv.gz

# Normalize
python3 -m mslab.ingest.tardis_normalize \
     data/raw/tardis/BTCUSDT_2025-06-01.csv.gz BTCUSDT

# Features → model → sim → charts
python3 -m mslab.features.build BTCUSDT
python3 -m mslab.models.validate
python3 -m mslab.backtest.run_cpp_sim
python3 -m mslab.backtest.analyze_fills
python3 -m mslab.backtest.feature_ablation
python3 -m mslab.backtest.fee_sweep_analytical
```

All figures saved to `reports/figures/`. All fill data saved to
`data/results/`. Methodology in `docs/methodology.md`.