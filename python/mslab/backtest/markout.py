"""
markout.py
Post-fill mid-price move at fixed horizons (adverse selection proxy).

Markout definition
------------------
For a BUY fill at time T with fill_price P:
    markout(h) = mid_price(T + h) - P

For a SELL fill at time T with fill_price P:
    markout(h) = P - mid_price(T + h)

Positive markout = market moved in your favor after the fill.
Negative markout = adverse selection (you bought before a drop, sold before a rally).

Horizons: 100ms, 500ms, 1s, 5s

Alignment note
--------------
mid_price is taken from the feature snapshot at or after T + h.
Feature snapshots are at 1s intervals, so 100ms/500ms horizons are
approximated by the next available 1s snapshot. This is noted as a
limitation in methodology.md — full tick-level markouts require
replaying the book at each horizon, which we defer to future work.

Look-ahead safety
-----------------
We always look FORWARD from the fill timestamp. We never use any
feature row with ts <= fill_ts. This is enforced by the strict
greater-than filter in _find_mid_after().
"""

import pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT     = pathlib.Path(__file__).resolve().parents[3]
HORIZONS = {
    "100ms": 100,
    "500ms": 500,
    "1s"   : 1_000,
    "5s"   : 5_000,
}


def _find_mid_after(feat_by_ts: pd.Series,
                    fill_ts_ms: float,
                    horizon_ms: int) -> float | None:
    """
    Find the mid-price at the first feature snapshot strictly after
    fill_ts_ms + horizon_ms.

    Parameters
    ----------
    feat_by_ts  : pd.Series with index=ts_local (ms), values=mid_price
    fill_ts_ms  : fill timestamp in milliseconds
    horizon_ms  : horizon in milliseconds

    Returns
    -------
    mid-price float, or None if no snapshot exists after that horizon
    """
    target_ts = fill_ts_ms + horizon_ms
    # Strict greater-than: never use a snapshot at or before the fill
    future = feat_by_ts[feat_by_ts.index > target_ts]
    if future.empty:
        return None
    return float(future.iloc[0])


def compute_markouts(fills_df: pd.DataFrame,
                     feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute markouts at all horizons for each fill.

    Parameters
    ----------
    fills_df : DataFrame with columns [side, fill_price, fill_ts_ns, ...]
    feat_df  : feature DataFrame with columns [ts_local, mid_price, ...]

    Returns
    -------
    fills_df with additional columns:
        markout_100ms, markout_500ms, markout_1s, markout_5s
    """
    if fills_df.empty:
        return fills_df

    # Build mid-price series indexed by ts_local (ms)
    feat_by_ts = feat_df.set_index("ts_local")["mid_price"].sort_index()

    result = fills_df.copy()

    for col, horizon_ms in HORIZONS.items():
        markouts = []
        for _, fill in fills_df.iterrows():
            fill_ts_ms = fill["fill_ts_ns"] / 1_000_000
            mid_after  = _find_mid_after(feat_by_ts, fill_ts_ms, horizon_ms)

            if mid_after is None:
                markouts.append(np.nan)
                continue

            if fill["side"] == "buy":
                markouts.append(mid_after - fill["fill_price"])
            else:
                markouts.append(fill["fill_price"] - mid_after)

        result[f"markout_{col}"] = markouts

    return result


def markout_summary(fills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize markouts across all fills.

    Returns a DataFrame with mean/std/t-stat for each horizon,
    split by maker/taker if available.
    """
    horizon_cols = [f"markout_{h}" for h in HORIZONS]
    missing = [c for c in horizon_cols if c not in fills_df.columns]
    if missing:
        raise ValueError(f"Missing markout columns: {missing}. "
                         f"Run compute_markouts() first.")

    rows = []
    for col in horizon_cols:
        vals = fills_df[col].dropna()
        if len(vals) < 2:
            continue
        mean   = vals.mean()
        std    = vals.std()
        tstat  = mean / (std / np.sqrt(len(vals)))
        rows.append({
            "horizon"  : col.replace("markout_", ""),
            "mean_$"   : round(mean, 6),
            "std_$"    : round(std, 6),
            "t_stat"   : round(tstat, 3),
            "n_fills"  : len(vals),
        })

    return pd.DataFrame(rows)


def load_and_compute(symbol: str = "BTCUSDT",
                     run_label: str = "realistic") -> pd.DataFrame:
    """
    Convenience loader: reads fills + features from disk, returns fills
    with markout columns attached.
    """
    fills_path = ROOT / "data" / "results" / f"{symbol}_fills_{run_label}.parquet"
    feat_path  = ROOT / "data" / "features" / f"{symbol}_features.parquet"

    fills_df = pq.read_table(fills_path).to_pandas()
    feat_df  = pq.read_table(feat_path).to_pandas()

    return compute_markouts(fills_df, feat_df)