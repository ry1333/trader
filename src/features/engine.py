"""Feature engineering for the RSIH strategy — candles-only MVP.

All features from the blueprint's "Core (always available)" list:
- Multi-horizon returns + EW momentum
- Realized volatility / ATR
- Range expansion score
- Z-score (overextension from rolling mean)
- Time-of-day / session flags
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all features to a bar DataFrame. Expects OHLCV + timestamp columns."""
    df = df.copy()

    # ── Returns over multiple horizons ────────────────────────────────
    for n in [1, 3, 6, 12]:
        df[f"ret_{n}"] = df["close"].pct_change(n)

    # Exponentially weighted momentum (span ~12 bars = 1 hour)
    df["ew_mom"] = df["close"].ewm(span=12).mean().pct_change()

    # ── Volatility measures ───────────────────────────────────────────
    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_50"] = tr.rolling(50).mean()

    # Realized volatility (rolling std of returns)
    df["rvol_12"] = df["ret_1"].rolling(12).std()
    df["rvol_50"] = df["ret_1"].rolling(50).std()

    # ── Range expansion score ─────────────────────────────────────────
    # Current bar range vs rolling average range
    bar_range = df["high"] - df["low"]
    avg_range = bar_range.rolling(50).mean()
    df["range_expansion"] = bar_range / avg_range.replace(0, np.nan)

    # ── Z-score (overextension from rolling mean) ─────────────────────
    for window in [20, 50]:
        roll_mean = df["close"].rolling(window).mean()
        roll_std = df["close"].rolling(window).std()
        df[f"zscore_{window}"] = (df["close"] - roll_mean) / roll_std.replace(0, np.nan)

    # ── Momentum indicators ───────────────────────────────────────────
    # RSI (14-bar)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── Volume features ───────────────────────────────────────────────
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)

    # ── Time-of-day / session flags ───────────────────────────────────
    if df["timestamp"].dt.tz is not None:
        ct = df["timestamp"].dt.tz_convert("US/Central")
    else:
        ct = df["timestamp"]

    df["hour_ct"] = ct.dt.hour
    df["minute_ct"] = ct.dt.minute
    df["time_sin"] = np.sin(2 * np.pi * (ct.dt.hour * 60 + ct.dt.minute) / (24 * 60))
    df["time_cos"] = np.cos(2 * np.pi * (ct.dt.hour * 60 + ct.dt.minute) / (24 * 60))

    # Session flags (US futures RTH = 8:30 AM - 3:00 PM CT)
    minutes_of_day = ct.dt.hour * 60 + ct.dt.minute
    df["is_rth"] = ((minutes_of_day >= 510) & (minutes_of_day < 900)).astype(int)  # 8:30-15:00
    df["near_close"] = (minutes_of_day >= 870).astype(int)  # After 2:30 PM CT — danger zone
    df["is_open_30m"] = ((minutes_of_day >= 510) & (minutes_of_day < 540)).astype(int)

    # Day of week
    df["dow"] = ct.dt.dayofweek

    return df


FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ew_mom",
    "atr_14", "atr_50", "rvol_12", "rvol_50",
    "range_expansion",
    "zscore_20", "zscore_50",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "vol_ratio",
    "time_sin", "time_cos", "is_rth", "near_close", "is_open_30m", "dow",
]
