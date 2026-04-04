"""Session-aware features for ES futures.

ES has distinct intraday structure:
- Globex overnight: 5:00 PM - 8:30 AM CT
- RTH open: 8:30 AM CT (highest volume, biggest moves)
- Midday: 11:00 AM - 1:00 PM CT (chop, low volume)
- RTH close: 3:00 PM CT (rebalancing flows)

These session dynamics create exploitable patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add session-aware features to bar DataFrame."""
    df = df.copy()

    # Convert to Central Time for session logic
    if df["timestamp"].dt.tz is not None:
        ct = df["timestamp"].dt.tz_convert("US/Central")
    else:
        ct = df["timestamp"]

    minutes = ct.dt.hour * 60 + ct.dt.minute

    # ── Session labels ────────────────────────────────────────────────
    # Pre-market: 5:00 PM (prev day) - 8:30 AM CT
    # RTH: 8:30 AM - 3:00 PM CT
    df["is_premarket"] = ((minutes >= 0) & (minutes < 510)).astype(int)
    df["is_rth"] = ((minutes >= 510) & (minutes < 900)).astype(int)

    # Sub-sessions within RTH
    df["is_open_drive"] = ((minutes >= 510) & (minutes < 570)).astype(int)  # 8:30-9:30
    df["is_midday_chop"] = ((minutes >= 660) & (minutes < 780)).astype(int)  # 11:00-1:00
    df["is_close_drive"] = ((minutes >= 840) & (minutes < 900)).astype(int)  # 2:00-3:00
    df["is_power_hour"] = ((minutes >= 780) & (minutes < 900)).astype(int)  # 1:00-3:00

    # ── VWAP (Volume Weighted Average Price) ──────────────────────────
    # Reset each RTH session
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_x_vol"] = df["typical_price"] * df["volume"]

    # Detect session boundaries (RTH start)
    date_col = ct.dt.date
    df["_session_date"] = date_col

    # Cumulative VWAP per session
    vwap = []
    cum_tp_vol = 0.0
    cum_vol = 0.0
    current_date = None

    for i in range(len(df)):
        d = df["_session_date"].iloc[i]
        is_rth = df["is_rth"].iloc[i]

        if d != current_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            current_date = d

        if is_rth:
            cum_tp_vol += df["tp_x_vol"].iloc[i]
            cum_vol += df["volume"].iloc[i]
            vwap.append(cum_tp_vol / cum_vol if cum_vol > 0 else df["close"].iloc[i])
        else:
            vwap.append(np.nan)

    df["vwap"] = vwap
    df["vwap"] = df["vwap"].ffill()

    # VWAP-based features
    df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    df["vwap_slope"] = df["vwap"].diff(3) / df["vwap"].shift(3).replace(0, np.nan)

    # ── Opening Range (first 30 min of RTH) ──────────────────────────
    or_high = {}
    or_low = {}
    for date, group in df[df["is_open_drive"] == 1].groupby("_session_date"):
        or_high[date] = group["high"].max()
        or_low[date] = group["low"].min()

    df["or_high"] = df["_session_date"].map(or_high)
    df["or_low"] = df["_session_date"].map(or_low)
    df["or_range"] = df["or_high"] - df["or_low"]

    # Price position relative to opening range
    df["above_or"] = (df["close"] > df["or_high"]).astype(int)
    df["below_or"] = (df["close"] < df["or_low"]).astype(int)
    df["or_breakout"] = df["above_or"] | df["below_or"]

    # Distance from OR boundary (normalized)
    or_range_safe = df["or_range"].replace(0, np.nan)
    df["dist_from_or_high"] = (df["close"] - df["or_high"]) / or_range_safe
    df["dist_from_or_low"] = (df["close"] - df["or_low"]) / or_range_safe

    # ── Overnight gap ─────────────────────────────────────────────────
    # Previous RTH close vs current day open
    prev_close = {}
    current_open = {}
    for date, group in df[df["is_rth"] == 1].groupby("_session_date"):
        prev_close[date] = group["close"].iloc[-1]
        current_open[date] = group["open"].iloc[0]

    dates = sorted(prev_close.keys())
    df["overnight_gap"] = 0.0
    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        if today in current_open and yesterday in prev_close:
            gap = (current_open[today] - prev_close[yesterday]) / prev_close[yesterday]
            mask = df["_session_date"] == today
            df.loc[mask, "overnight_gap"] = gap

    # ── Intraday high/low tracking ────────────────────────────────────
    day_high = {}
    day_low = {}
    running_high = {}
    running_low = {}

    for i in range(len(df)):
        d = df["_session_date"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if d not in running_high:
            running_high[d] = h
            running_low[d] = l
        else:
            running_high[d] = max(running_high[d], h)
            running_low[d] = min(running_low[d], l)

    # Map back — this gives the running high/low up to each bar
    # Simplified: use session high/low
    df["session_high"] = df["_session_date"].map(running_high)
    df["session_low"] = df["_session_date"].map(running_low)
    session_range = (df["session_high"] - df["session_low"]).replace(0, np.nan)
    df["intraday_position"] = (df["close"] - df["session_low"]) / session_range

    # Cleanup
    df = df.drop(columns=["typical_price", "tp_x_vol", "_session_date"])

    return df


SESSION_FEATURE_COLS = [
    "is_premarket", "is_rth", "is_open_drive", "is_midday_chop",
    "is_close_drive", "is_power_hour",
    "vwap", "price_vs_vwap", "vwap_slope",
    "or_high", "or_low", "or_range", "above_or", "below_or",
    "or_breakout", "dist_from_or_high", "dist_from_or_low",
    "overnight_gap", "intraday_position",
]
