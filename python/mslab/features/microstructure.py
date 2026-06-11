"""
microstructure.py
Computes microstructure features from a sequence of L2 order book snapshots.

Features computed per snapshot:
  - mid_price         : (best_bid + best_ask) / 2
  - spread            : best_ask - best_bid
  - relative_spread   : spread / mid_price
  - depth_imbalance_5 : (bid_vol - ask_vol) / (bid_vol + ask_vol) top 5 levels
  - depth_imbalance_10: same for top 10 levels
  - book_pressure     : depth imbalance weighted by distance from mid
"""

import numpy as np


def compute_features(book_snapshot: dict) -> dict:
    """
    Compute microstructure features from a single book snapshot.

    Parameters
    ----------
    book_snapshot : dict with keys:
        ts_local   : int   — timestamp in milliseconds
        bids       : list of (price, size) tuples, sorted highest first
        asks       : list of (price, size) tuples, sorted lowest first

    Returns
    -------
    dict of feature name -> value, plus ts_local
    """
    bids = book_snapshot["bids"]  # [(price, size), ...] highest first
    asks = book_snapshot["asks"]  # [(price, size), ...] lowest first
    ts   = book_snapshot["ts_local"]

    # Need at least one level on each side
    if not bids or not asks:
        return None

    best_bid = bids[0][0]
    best_ask = asks[0][0]

    # ── Basic features ───────────────────────────────────────────────────────
    mid_price       = (best_bid + best_ask) / 2.0
    spread          = best_ask - best_bid
    relative_spread = spread / mid_price if mid_price > 0 else np.nan

    # ── Depth imbalance (top k levels) ───────────────────────────────────────
    # Formula: (sum_bid_size - sum_ask_size) / (sum_bid_size + sum_ask_size)
    # Range: [-1, +1]. Positive = more bid pressure. Negative = more ask pressure.

    def depth_imbalance(k: int) -> float:
        bid_vol = sum(s for _, s in bids[:k])
        ask_vol = sum(s for _, s in asks[:k])
        total   = bid_vol + ask_vol
        if total == 0:
            return np.nan
        return (bid_vol - ask_vol) / total

    di_5  = depth_imbalance(5)
    di_10 = depth_imbalance(10)

    # ── Book pressure (distance-weighted depth imbalance) ────────────────────
    # Levels closer to mid-price get higher weight.
    # Weight for level i = 1 / (1 + distance_from_mid)
    # where distance_from_mid = abs(level_price - mid_price)

    def book_pressure() -> float:
        weighted_bid = 0.0
        weighted_ask = 0.0

        for price, size in bids:
            dist = abs(price - mid_price)
            weight = 1.0 / (1.0 + dist)
            weighted_bid += weight * size

        for price, size in asks:
            dist = abs(price - mid_price)
            weight = 1.0 / (1.0 + dist)
            weighted_ask += weight * size

        total = weighted_bid + weighted_ask
        if total == 0:
            return np.nan
        return (weighted_bid - weighted_ask) / total

    pressure = book_pressure()

    return {
        "ts_local"          : ts,
        "mid_price"         : mid_price,
        "spread"            : spread,
        "relative_spread"   : relative_spread,
        "depth_imbalance_5" : di_5,
        "depth_imbalance_10": di_10,
        "book_pressure"     : pressure,
    }