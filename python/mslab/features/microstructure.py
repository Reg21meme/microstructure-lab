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

# ── Micro-price (Stoikov) ─────────────────────────────────────────────────
    # Weights best bid and best ask by the opposite side's volume.
    # Reflects where price is more likely to move given current book imbalance.
    best_bid_size = bids[0][1]
    best_ask_size = asks[0][1]
    total_size    = best_bid_size + best_ask_size

    if total_size > 0:
        micro_price = (best_bid * (best_ask_size / total_size) +
                       best_ask * (best_bid_size / total_size))
        micro_price_deviation = micro_price - mid_price
    else:
        micro_price           = np.nan
        micro_price_deviation = np.nan

    return {
        "ts_local"             : ts,
        "mid_price"            : mid_price,
        "spread"               : spread,
        "relative_spread"      : relative_spread,
        "depth_imbalance_5"    : di_5,
        "depth_imbalance_10"   : di_10,
        "book_pressure"        : pressure,
        "micro_price"          : micro_price,
        "micro_price_deviation": micro_price_deviation,
    }

def compute_ofi_single(prev_snapshot: dict, curr_snapshot: dict) -> float:
    """
    Compute single-level Order Flow Imbalance (OFI) between two snapshots.

    Uses only the best bid and best ask level (top of book).

    Parameters
    ----------
    prev_snapshot : book snapshot at time T   (same format as compute_features input)
    curr_snapshot : book snapshot at time T+1

    Returns
    -------
    float — OFI value. Positive = net buying pressure. Negative = net selling pressure.
    """
    prev_bids = prev_snapshot["bids"]
    prev_asks = prev_snapshot["asks"]
    curr_bids = curr_snapshot["bids"]
    curr_asks = curr_snapshot["asks"]

    if not prev_bids or not prev_asks or not curr_bids or not curr_asks:
        return np.nan

    # ── Bid side contribution ────────────────────────────────────────────────
    old_bid_price, old_bid_size = prev_bids[0]
    new_bid_price, new_bid_size = curr_bids[0]

    if new_bid_price > old_bid_price:
        # Best bid price improved — aggressive new buyer entered
        ofi_bid = new_bid_size
    elif new_bid_price == old_bid_price:
        # Same price level — measure size change
        ofi_bid = new_bid_size - old_bid_size
    else:
        # Best bid price dropped — buyers retreated, old volume gone
        ofi_bid = -old_bid_size

    # ── Ask side contribution (mirror image) ─────────────────────────────────
    old_ask_price, old_ask_size = prev_asks[0]
    new_ask_price, new_ask_size = curr_asks[0]

    if new_ask_price < old_ask_price:
        # Best ask price improved (dropped) — aggressive new seller entered
        ofi_ask = -new_ask_size
    elif new_ask_price == old_ask_price:
        # Same price level — measure size change (increase = more supply = negative)
        ofi_ask = old_ask_size - new_ask_size
    else:
        # Best ask price rose — sellers retreated, old volume gone
        ofi_ask = old_ask_size

    return ofi_bid - ofi_ask


def compute_mlofi(prev_snapshot: dict,
                  curr_snapshot: dict,
                  levels: int = 10) -> list[float]:
    """
    Compute Multi-Level OFI across the top k price levels.

    Applies the same OFI formula independently at each level.
    Levels are matched by rank (level 1 = best, level 2 = second best, etc.)
    not by price — because prices shift between snapshots.

    Parameters
    ----------
    prev_snapshot : book snapshot at time T
    curr_snapshot : book snapshot at time T+1
    levels        : number of levels to compute (default 10)

    Returns
    -------
    list of float, length = levels. Element i is OFI at level i+1.
    """
    prev_bids = prev_snapshot["bids"][:levels]
    prev_asks = prev_snapshot["asks"][:levels]
    curr_bids = curr_snapshot["bids"][:levels]
    curr_asks = curr_snapshot["asks"][:levels]

    mlofi = []

    for i in range(levels):
        # If either snapshot doesn't have this level, record nan
        if (i >= len(prev_bids) or i >= len(curr_bids) or
                i >= len(prev_asks) or i >= len(curr_asks)):
            mlofi.append(np.nan)
            continue

        old_bid_price, old_bid_size = prev_bids[i]
        new_bid_price, new_bid_size = curr_bids[i]
        old_ask_price, old_ask_size = prev_asks[i]
        new_ask_price, new_ask_size = curr_asks[i]

        # Bid contribution at level i
        if new_bid_price > old_bid_price:
            ofi_bid = new_bid_size
        elif new_bid_price == old_bid_price:
            ofi_bid = new_bid_size - old_bid_size
        else:
            ofi_bid = -old_bid_size

        # Ask contribution at level i
        if new_ask_price < old_ask_price:
            ofi_ask = -new_ask_size
        elif new_ask_price == old_ask_price:
            ofi_ask = old_ask_size - new_ask_size
        else:
            ofi_ask = old_ask_size

        mlofi.append(ofi_bid - ofi_ask)

    return mlofi

def compute_realized_vol(mid_prices: list[float],
                         window: int = 20) -> list[float]:
    """
    Compute rolling realized volatility from a sequence of mid-prices.

    Parameters
    ----------
    mid_prices : list of mid-price values in time order
    window     : number of returns to include in each rolling std

    Returns
    -------
    list of float, same length as mid_prices.
    First (window) values are NaN — not enough history yet.
    """
    prices  = np.array(mid_prices)
    n       = len(prices)
    vol     = np.full(n, np.nan)

    if n < 2:
        return vol.tolist()

    # Compute log returns: log(p_t / p_{t-1})
    # Log returns are more statistically well-behaved than simple returns
    log_returns = np.log(prices[1:] / prices[:-1])

    for i in range(window, n):
        # Rolling window of log returns ending at position i
        window_returns = log_returns[i - window: i]
        vol[i]         = np.std(window_returns, ddof=1)

    return vol.tolist()


def compute_markouts(mid_prices: list[float],
                     horizons: list[int] = [5, 10, 20]) -> dict[str, list[float]]:
    """
    Compute forward mid-price moves at multiple horizons.

    These are LABELS / post-trade analysis only — never use as model inputs.
    Named with future_ prefix to make this explicit.

    Parameters
    ----------
    mid_prices : list of mid-price values in time order
    horizons   : list of forward snapshot counts to compute markouts for

    Returns
    -------
    dict mapping feature name -> list of float, same length as mid_prices.
    Last (horizon) values are NaN — no future data available.
    """
    prices  = np.array(mid_prices)
    n       = len(prices)
    result  = {}

    for h in horizons:
        markout = np.full(n, np.nan)
        # For each position, look h steps forward
        # Last h positions have no future data
        markout[:n - h] = prices[h:] - prices[:n - h]
        result[f"future_mid_move_{h}"] = markout.tolist()

    return result