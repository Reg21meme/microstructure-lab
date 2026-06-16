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

**Symbols collected:** BTCUSDT (full trading day June 1 2025 via Tardis
historical datasets, 19,443,879 updates, 85,078 feature snapshots),
ETHUSDT (35 minutes live collection, 776,714 updates).

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
On full-day data, `micro_price_deviation` ranks third in feature
ablation (standalone IC = 0.025, 4.4% IC drop when removed).
`depth_imbalance_5` is the dominant feature (95.2% IC drop).

### Order Flow Imbalance (OFI)
Following Cont, Kukanov & Stoikov (2014):
OFI_t = ΔQ_bid_t - ΔQ_ask_t
where `ΔQ_bid_t` is the change in best-bid quantity and `ΔQ_ask_t`
is the change in best-ask quantity between consecutive snapshots.
Positive OFI = more buying pressure than selling pressure.

### Multi-Level OFI (MLOFI)
OFI extended to 10 price levels. The first principal component
(MLOFI PC1) summarizes the dominant direction of order flow across
all levels. PC1 explains 19.2% of variance on full-day BTCUSDT data.

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

Walk-forward CV on full-day BTCUSDT gives IC = 0.267 at the
5-second horizon (NW t-stat = 54.53, 85,078 feature snapshots).
IC decays to 0.260 at 10s and 0.244 at 20s — consistent with
a short-horizon microstructure signal.

### Information Coefficient (IC)
Pearson correlation between predicted and realized
`future_mid_move_5`. IC = 0.258 on the held-out test set (last 30% of data by time,
57,785 rows spanning ~8 hours of BTCUSDT data).

### Newey-West (HAC) standard errors
IC is computed on overlapping 5-second return windows, inducing
serial autocorrelation in the IC time series. Standard OLS standard
errors would understate uncertainty. Newey-West heteroskedasticity
and autocorrelation consistent (HAC) standard errors correct for
this, giving a t-statistic of 54.53 at the 5-second horizon
(85,078 observations, full trading day).


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

All 19,196 realistic fills are taker fills (orders priced to cross
the spread). Fee drag = $7,893 on a $1,440 gross PnL.
Total notional traded: $19.7M. Breakeven taker fee: **0.73 bps** —
5.5× below Binance VIP0. Fee per fill = $0.41 vs gross edge per
fill = $0.075 — the strategy is fee-dominated at retail rates.

### Queue-position approximation
At order submission, fill probability is approximated from the
current depth imbalance:
For a BUY order:
depth_ahead_fraction = (1 - depth_imbalance_5) / 2

fill_prob = clip(1 - depth_ahead_fraction, min=0.10, max=1.0)
A uniform draw determines whether the order enters the sim.
The queue model rejects ~25% of orders (24,954 → 18,810 submitted)
and reduces fee drag proportionally.

**Limitation:** This approximates queue position from displayed
depth at submission time. It does not model cancellation rates,
hidden orders, or time-varying fill probability. Full queue
modeling requires L3 order-by-order data.

### Markout (adverse selection proxy)
Post-fill mid-price move at fixed horizons:
markout(T, h) = mid(T+h) - fill_price   [for buy fills]

markout(T, h) = fill_price - mid(T+h)   [for sell fills]
Mean markout at 1s = $1.11/fill (t-stat = 25.6 on 19,194 fills),
indicating no adverse selection — the market moves in favor of
fills on average.

**Limitation:** Markouts at 100ms and 500ms horizons are aliased
to the 1s snapshot grid. Tick-level markouts require replaying
the full book at each horizon.

---

## 7. Robustness

### Fee sensitivity
Net PnL degrades linearly with taker fee. Breakeven at 0.73 bps
(full-day data, 19,196 fills, $19.7M notional). At Binance VIP0
(4 bps), fee drag = $7,893 vs gross PnL = $1,440 — a 5.5× gap.
Fee drag per fill ($0.41) exceeds gross edge per fill ($0.075),
making the strategy fee-dominated at any retail fee tier.

### Confidence threshold sweep
Full-day sweep across $0.05 to $1.00 shows non-monotonic behavior.
Best net PnL at $0.05 (-$474, 4,818 fills) — high-conviction moments
yield better gross PnL per fill ($1,506 on 4,818 fills vs $1,440 on
19,196 fills at $0.10). The strategy remains unprofitable at all
tested thresholds due to fee dominance.

### Symbol generalization
IC generalizes to ETHUSDT (0.31 vs 0.258 on BTCUSDT). ETHUSDT
was collected on a 35-minute live sample with near-zero naive PnL
($0.05), consistent with no strong directional trend in that window.

### Feature ablation
| Feature | IC drop (removed) | Standalone IC |
|---|---|---|
| depth_imbalance_5 | -0.246 (95.2%) | 0.012 |
| spread | -0.022 (8.5%) | 0.006 |
| micro_price_deviation | -0.011 (4.4%) | 0.025 |
| realized_vol | -0.001 (0.4%) | -0.029 |
| mlofi_pc1 | -0.001 (0.2%) | 0.030 |
| ofi | -0.000 (0.1%) | 0.004 |

`depth_imbalance_5` dominates on full-day data — removing it drops
IC by 95.2%. This contrasts with the 30-minute trending sample where
`micro_price_deviation` was most important (10.9% drop). Feature
importance is regime-dependent: depth imbalance captures mean-reverting
microstructure while micro-price deviation captures trending momentum.

---

## 8. Limitations and Honest Assessment

1. **Single day:** Primary results are on June 1 2025 BTCUSDT
   (full trading day, Tardis historical data). IC on the full
   day (0.258) is lower than the initial 30-minute trending
   sample (0.290), confirming the short sample was inflated.
   Multi-day generalization requires additional Tardis data.

2. **Fee dominance:** The strategy is unprofitable at any
   realistic retail fee level. It would require institutional
   fee tiers (< 0.73 bps) or a higher-frequency signal to
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
