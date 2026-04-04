"""Regime classifier — detects trend / mean-reversion / stress regimes.

Uses a rule-based approach for the MVP (no ML training needed).
The regime determines which signal logic to apply on each bar.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Regime(IntEnum):
    STRESS = 0       # High vol, wide spreads — reduce/stand down
    MEAN_REVERT = 1  # Range-bound, overextended — fade moves
    TREND = 2        # Directional momentum — trade with trend


def classify_regimes(df: pd.DataFrame) -> pd.Series:
    """Classify each bar into a regime based on volatility and momentum features.

    Logic:
    - STRESS:      rvol_12 > 2x rvol_50  OR  range_expansion > 3.0
    - TREND:       |ret_12| > 1.5 * atr_14  AND  rvol_12 moderate
    - MEAN_REVERT: everything else (range-bound conditions)
    """
    regime = pd.Series(Regime.MEAN_REVERT, index=df.index, dtype=int)

    # Stress: volatility spike
    vol_spike = df["rvol_12"] > (2.0 * df["rvol_50"])
    range_blow = df["range_expansion"] > 3.0
    stress_mask = vol_spike | range_blow
    regime[stress_mask] = Regime.STRESS

    # Trend: strong directional move without extreme vol
    abs_ret_12 = df["ret_12"].abs()
    # Normalize ATR to return space: atr / close
    atr_pct = df["atr_14"] / df["close"]
    strong_move = abs_ret_12 > (1.5 * atr_pct * 12)  # scale ATR to 12-bar horizon
    trend_mask = strong_move & ~stress_mask
    regime[trend_mask] = Regime.TREND

    return regime


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime column to DataFrame."""
    df = df.copy()
    df["regime"] = classify_regimes(df)
    return df
