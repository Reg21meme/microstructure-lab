# MicrostructureLab — Methodology

This document describes the statistical and microstructure methodology
behind MicrostructureLab. It is written for a technical reader —
a quant researcher or engineer who wants to understand the choices
made and why.

---

## 1. Data

**Source:** Binance public WebSocket API (`{symbol}@depth@100ms`),
synchronized with a REST snapshot following the official Binance
L2 book synchronization procedure:

1. Start buffering WebSocket depth updates immediately
2. Fetch a REST snapshot
3. Discard buffered updates with `u < snapshot.lastUpdateId`
4. Keep updates where `U <= lastUpdateId + 1 <= u`

This guarantees no sequence gap between snapshot and live updates.
Sequence gaps in subsequent updates are detected and flagged.

**Symbols collected:** BTCUSDT (30 minutes, 393,394 updates),
ETHUSDT (35 minutes, 776,714 updates).

**Storage:** Normalized to `(ts_exchange, ts_local, seq, side, price,
size, update_type)` and stored as Parquet via PyArrow.

**Data quality filters:**
- Negative or zero spread rows removed (1.3% of BTCUSDT, 4.6% of ETHUSDT)
- Sequence gaps logged but not present in collected data
- NaN feature rows dropped before model training

---

## 2. Feature Construction

Features are computed at 1-second snapshot intervals by replaying
L2 updates through the C++ order book engine.

### Mid-price and spread
mid = (best_bid + best_ask) / 2

spread = best_ask - best_bid

relative_spread = spread / mid

### Depth imbalance
Measures relative liquidity imbalance across the top k levels:
depth_imbalance_k = (bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k)
where `bid_depth_k` is the total displayed size across the top k bid
levels. Computed at k=5 and k=10.

### Micro-price (Stoikov 2018)
A quantity-weighted mid-price that accounts for order book imbalance:
micro_price = best_ask * q_bid / (q_bid + q_ask)

+ best_bid * q_ask / (q_bid + q_ask)
micro_price_deviation = micro_price - mid
`micro_price_deviation` is the most predictive single feature in
ablation (standalone IC = 0.28, ~10% IC drop when removed from
the full model).

### Order Flow Imbalance (OFI)
Following Cont, Kukanov & Stoikov (2014):
OFI_t = ΔQ_bid_t - ΔQ_ask_t
where `ΔQ_bid_t` is the change in best-bid quantity and `ΔQ_ask_t`
is the change in best-ask quantity between consecutive snapshots.
Positive OFI = more buying pressure than selling pressure.

### Multi-Level OFI (MLOFI)
OFI extended to 10 price levels. The first principal component
(MLOFI PC1) summarizes the dominant direction of order flow across
all levels. PC1 explains ~22% of variance on BTCUSDT (short sample;
expected to increase with more data).

### Book pressure
book_pressure = sum(bid_size_i / bid_price_i)

- sum(ask_size_i / ask_price_i)
across top 5 levels. Measures liquidity-weighted directional pressure.

### Realized volatility
Rolling standard deviation of mid-price returns over a 20-snapshot
window (20 seconds at 1s intervals).

---

## 3. Labels

Target variable: `future_mid_move_5` — the mid-price change over
the next 5 snapshots (5 seconds):
future_mid_move_5[t] = mid[t+5] - mid[t]

**Look-ahead alignment:** Labels use strictly forward information.
The feature at time t uses book state at time t; the label uses
mid-price at time t+5. No future information enters the feature
computation.

---

## 4. Model

**Ridge regression** (L2 regularization, α=1.0) on 6 features:
`depth_imbalance_5`, `micro_price_deviation`, `ofi`, `mlofi_pc1`,
`realized_vol`, `spread`. Features are standardized (zero mean,
unit variance) before fitting.

Ridge was chosen over logistic regression for the regression
framing (predicting signed dollar move rather than binary direction),
and over gradient boosting to keep the model interpretable and
reduce overfitting risk on a short sample.

---

## 5. Validation

### Time-based splits only
No random shuffling. The training set is always strictly earlier
in time than the test set. This is required for financial time
series where random splits cause look-ahead leakage.

### Purged and embargoed walk-forward cross-validation
Following López de Prado (2018), cross-validation uses:

- **Purging:** Samples overlapping in time between train and
  validation folds are removed. Since labels look 5 steps forward,
  the last 5 rows of each training fold are dropped.

- **Embargo:** A gap of additional rows is excluded between train
  and validation folds to prevent information leakage from
  microstructure autocorrelation. Without the embargo, validation
  IC is inflated by autocorrelated features bleeding across the
  fold boundary.

5-fold purged/embargoed walk-forward CV on BTCUSDT gives
IC = 0.32 ± 0.04 at the 5-second horizon.

### Information Coefficient (IC)
Pearson correlation between predicted and realized
`future_mid_move_5`. IC = 0.29 on the held-out test set (last 30%
of data by time).

### Newey-West (HAC) standard errors
IC is computed on overlapping 5-second return windows, inducing
serial autocorrelation in the IC time series. Standard OLS standard
errors would understate uncertainty. Newey-West heteroskedasticity
and autocorrelation consistent (HAC) standard errors correct for
this, giving a t-statistic of 39.7 at the 5-second horizon.

The Newey-West lag truncation is set to `floor(4 * (T/100)^(2/9))`
following the standard data-driven rule.

### Deflated Sharpe Ratio
The deflated Sharpe ratio (Bailey & López de Prado 2014) adjusts
the observed Sharpe for:
- Multiple testing across parameter configurations
- Non-normality of returns (skewness and kurtosis)
- Estimation error from finite sample size

A strategy with deflated Sharpe > 0 has positive expected value
after accounting for these adjustments. We report deflated Sharpe
to avoid claiming a result that is an artifact of parameter search.

### Calibration
Logistic regression predictions are calibrated via Platt scaling
and evaluated with the Brier score. Brier skill score of 0.30
(vs. naive climatological baseline) indicates meaningful
probabilistic calibration.

---

## 6. Execution Simulation

### Architecture
The C++ execution simulator (`ExecutionSim`) replays L2 updates
in event-driven order and matches synthetic orders against the
reconstructed book. Orders submitted from the Python signal layer
are injected into the C++ event stream with a configurable latency
offset.

### Latency model
Fixed latency + optional Gaussian jitter:
arrive_time = signal_time + base_latency + N(0, jitter²)
Baseline: 10ms fixed, 0ms jitter. Stress tested at 0/10/50/100ms.
Result: net PnL is insensitive to latency in the 0–100ms range
at 1-second signal frequency, confirming that latency is not the
binding constraint for this strategy.

### Fee model
Maker-taker model loaded from `configs/fees.yaml`:
- Taker fee: 0.04% (4 bps) — paid when crossing the spread
- Maker rebate: 0.01% (1 bp) — received when resting in book

All 326 realistic fills are taker fills (orders priced to cross
the spread). Fee drag = $67.47 on a $5.19 gross PnL.
Breakeven taker fee: **0.31 bps** — 13× below Binance VIP0.

### Queue-position approximation
At order submission, fill probability is approximated from the
current depth imbalance:
For a BUY order:
depth_ahead_fraction = (1 - depth_imbalance_5) / 2

fill_prob = clip(1 - depth_ahead_fraction, min=0.10, max=1.0)
A uniform draw determines whether the order enters the sim.
The queue model rejects ~22% of orders (298 → 268 submitted)
and reduces fee drag proportionally.

**Limitation:** This approximates queue position from displayed
depth at submission time. It does not model cancellation rates,
hidden orders, or time-varying fill probability. Full queue
modeling requires L3 order-by-order data.

### Markout (adverse selection proxy)
Post-fill mid-price move at fixed horizons:
markout(T, h) = mid(T+h) - fill_price   [for buy fills]

markout(T, h) = fill_price - mid(T+h)   [for sell fills]
Positive markout = market moved in your favor. Mean markout at 1s
= $0.92/fill (t-stat = 5.99), indicating no adverse selection on
this trending sample.

**Limitation:** Markouts at 100ms and 500ms horizons are aliased
to the 1s snapshot grid. Tick-level markouts require replaying
the full book at each horizon.

---

## 7. Robustness

### Fee sensitivity
Net PnL degrades linearly with taker fee. Breakeven at 0.31 bps.
At Binance VIP0 (4 bps), the strategy loses $16.87 per basis point
of additional fee — directly derivable as
`n_fills × order_size × avg_price × fee_rate`.

### Confidence threshold sweep
Raising `SIGNAL_THRESH` from $0.05 to $1.00 monotonically
reduces fill count and fee drag but does not improve net PnL
to positive. Gross PnL is insensitive to threshold, suggesting
the signal edge is distributed across all confidence levels
rather than concentrated at high-conviction moments — consistent
with a linear ridge model.

### Symbol generalization
IC generalizes to ETHUSDT (0.31 vs 0.29 on BTCUSDT). Naive PnL
on ETHUSDT is near zero ($0.05), vs +$30.27 on BTCUSDT, confirming
the BTCUSDT result was inflated by a trending 30-minute sample.

### Feature ablation
| Feature | IC drop (removed) | Standalone IC |
|---|---|---|
| micro_price_deviation | -0.032 (10.9%) | 0.282 |
| depth_imbalance_5 | -0.029 (10.0%) | 0.259 |
| spread | -0.020 (6.8%) | -0.097 |
| realized_vol | -0.009 (2.9%) | -0.004 |
| mlofi_pc1 | -0.000 (0.2%) | 0.090 |
| ofi | +0.001 (-0.4%) | 0.019 |

`micro_price_deviation` and `depth_imbalance_5` carry ~90% of
the signal. OFI has near-zero marginal value on this short sample.
`spread` shows a suppression effect: negative standalone IC but
positive contribution in combination.

---

## 8. Limitations and Honest Assessment

1. **Short sample:** All results are on 30–35 minute samples.
   IC and PnL figures are expected to change materially on a
   full trading day. A trending sample inflates both IC and
   naive PnL.

2. **Fee dominance:** The strategy is unprofitable at any
   realistic retail fee level. It would require institutional
   fee tiers (< 0.31 bps) or a higher-frequency signal to
   overcome transaction costs.

3. **1-second signal frequency:** At 1s intervals, latency
   sensitivity is negligible. The strategy is not a latency
   race — it is a microstructure signal research project.

4. **Queue model approximation:** Fill probability from displayed
   depth is a first-order approximation. Real queue position
   depends on order arrival time relative to other participants,
   which requires L3 data.

5. **Crypto vs equities:** Crypto microstructure differs from
   equity microstructure in tick size, maker-taker structure,
   and participant composition. Results may not transfer to
   equity L2 data without re-calibration.

6. **No live trading:** This is a research and simulation system.
   No claims of live profitability are made.
