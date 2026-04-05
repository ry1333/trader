"""Intra-bar features — extracts 1-min information for 5-min signal bars.

Instead of rebuilding the backtester for 1-min bars, we add 1-min derived
features to the 5-min model. This gives the AI the INFORMATION from 1-min
without the computational cost.

For each 5-min bar, looks at the 5 constituent 1-min bars and computes:
- Intra-bar trend consistency (all 5 same direction = strong)
- Volume profile (front-loaded vs back-loaded)
- Intra-bar volatility (noise level within the bar)
- Rejection quality (wick structure from 1-min perspective)
- Momentum acceleration (is the move speeding up or slowing?)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_intrabar_features(
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame,
) -> pd.DataFrame:
    """Add 1-min derived features to 5-min DataFrame.

    Matches 1-min bars to their parent 5-min bar by timestamp,
    then computes aggregate features from the 5 constituent bars.
    """
    if df_1m is None or len(df_1m) == 0:
        # No 1-min data available — add empty columns
        for col in INTRABAR_FEATURE_COLS:
            df_5m[col] = 0.0
        return df_5m

    df_5m = df_5m.copy()

    # Ensure timestamps are aligned
    if df_1m["timestamp"].dt.tz is None and df_5m["timestamp"].dt.tz is not None:
        df_1m = df_1m.copy()
        df_1m["timestamp"] = df_1m["timestamp"].dt.tz_localize("UTC")
    elif df_1m["timestamp"].dt.tz is not None and df_5m["timestamp"].dt.tz is None:
        df_1m = df_1m.copy()
        df_1m["timestamp"] = df_1m["timestamp"].dt.tz_localize(None)

    # Create 5-min bucket key for 1-min bars
    df_1m_copy = df_1m.copy()
    df_1m_copy["bar_5m"] = df_1m_copy["timestamp"].dt.floor("5min")

    # Pre-compute all intra-bar features
    intrabar_data = {}

    for bar_ts, group in df_1m_copy.groupby("bar_5m"):
        if len(group) < 2:
            continue

        closes = group["close"].values
        opens = group["open"].values
        highs = group["high"].values
        lows = group["low"].values
        volumes = group["volume"].values
        n = len(group)

        # 1. Trend consistency: how many 1-min bars moved in the same direction?
        returns = np.diff(closes)
        if len(returns) > 0:
            pos = np.sum(returns > 0)
            neg = np.sum(returns < 0)
            trend_consistency = abs(pos - neg) / len(returns)  # 0 = mixed, 1 = all same
        else:
            trend_consistency = 0.0

        # 2. Volume profile: front-loaded (institutional) vs back-loaded (retail)
        if n >= 3 and volumes.sum() > 0:
            front_vol = volumes[:n // 2].sum()
            back_vol = volumes[n // 2:].sum()
            total_vol = volumes.sum()
            vol_front_pct = front_vol / total_vol  # >0.6 = front-loaded (good for breakouts)
        else:
            vol_front_pct = 0.5

        # 3. Intra-bar volatility: std of 1-min returns relative to 5-min range
        bar_range = highs.max() - lows.min()
        if bar_range > 0 and len(returns) > 1:
            micro_vol = np.std(returns) / (bar_range / closes.mean()) if closes.mean() > 0 else 0
        else:
            micro_vol = 0.0

        # 4. Rejection quality: did price spike then reverse within the bar?
        bar_close = closes[-1]
        bar_open = opens[0]
        bar_high = highs.max()
        bar_low = lows.min()
        if bar_range > 0:
            # Upper wick from 1-min perspective
            upper_wick = (bar_high - max(bar_close, bar_open)) / bar_range
            lower_wick = (min(bar_close, bar_open) - bar_low) / bar_range
            body_pct = abs(bar_close - bar_open) / bar_range
        else:
            upper_wick = 0.0
            lower_wick = 0.0
            body_pct = 0.0

        # 5. Momentum acceleration: is the move speeding up?
        if len(returns) >= 3:
            first_half = np.mean(np.abs(returns[:len(returns) // 2]))
            second_half = np.mean(np.abs(returns[len(returns) // 2:]))
            accel = second_half / first_half if first_half > 0 else 1.0
        else:
            accel = 1.0

        # 6. Directional conviction: net move / total absolute movement
        total_abs_move = np.sum(np.abs(returns))
        net_move = closes[-1] - closes[0]
        conviction = abs(net_move) / total_abs_move if total_abs_move > 0 else 0.0

        intrabar_data[bar_ts] = {
            "ib_trend_consistency": trend_consistency,
            "ib_vol_front_pct": vol_front_pct,
            "ib_micro_vol": micro_vol,
            "ib_upper_wick": upper_wick,
            "ib_lower_wick": lower_wick,
            "ib_body_pct": body_pct,
            "ib_momentum_accel": accel,
            "ib_conviction": conviction,
        }

    # Map to 5-min DataFrame
    for col in INTRABAR_FEATURE_COLS:
        df_5m[col] = 0.0

    for idx in range(len(df_5m)):
        ts = df_5m["timestamp"].iloc[idx]
        if hasattr(ts, "floor"):
            key = ts.floor("5min")
        else:
            key = ts
        if key in intrabar_data:
            for col, val in intrabar_data[key].items():
                df_5m.at[df_5m.index[idx], col] = val

    logger.info(f"Added {len(INTRABAR_FEATURE_COLS)} intra-bar features from {len(intrabar_data)} matched bars")
    return df_5m


INTRABAR_FEATURE_COLS = [
    "ib_trend_consistency",   # 0-1: all 1-min bars same direction
    "ib_vol_front_pct",       # 0-1: front-loaded volume (institutional)
    "ib_micro_vol",           # Noise level within the bar
    "ib_upper_wick",          # Upper rejection from 1-min perspective
    "ib_lower_wick",          # Lower rejection
    "ib_body_pct",            # Body as % of range
    "ib_momentum_accel",      # Is move speeding up or slowing?
    "ib_conviction",          # Net move / total absolute movement
]
