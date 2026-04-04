"""Signal generation — regime-dependent entry/exit logic.

RSIH (Regime-Switching Intraday Hybrid):
- Trend regime:       momentum/pullback entries
- Mean-revert regime: fade overextended moves
- Stress regime:      no new entries, exit existing
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

from src.strategy.regime import Regime


class Signal(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Generate trade signals for each bar.

    Returns a Series of Signal values.
    Assumes df has features + regime columns.
    """
    signals = pd.Series(Signal.FLAT, index=df.index, dtype=int)

    for i in range(1, len(df)):
        regime = df["regime"].iloc[i]

        # STRESS: no new entries
        if regime == Regime.STRESS:
            continue

        # Near session close: no new entries (flatten zone)
        if df["near_close"].iloc[i]:
            continue

        if regime == Regime.TREND:
            signals.iloc[i] = _trend_signal(df, i)
        elif regime == Regime.MEAN_REVERT:
            signals.iloc[i] = _mean_revert_signal(df, i)

    return signals


def _trend_signal(df: pd.DataFrame, i: int) -> int:
    """Momentum pullback entry in trend regime.

    Logic:
    - 12-bar return positive + RSI between 40-70 (pullback, not overbought) → LONG
    - 12-bar return negative + RSI between 30-60 (pullback, not oversold)  → SHORT
    - Confirm with MACD histogram direction
    """
    ret_12 = df["ret_12"].iloc[i]
    rsi = df["rsi_14"].iloc[i]
    macd_hist = df["macd_hist"].iloc[i]
    vol_ratio = df["vol_ratio"].iloc[i]

    if pd.isna(ret_12) or pd.isna(rsi) or pd.isna(macd_hist):
        return Signal.FLAT

    # Require minimum volume participation
    if vol_ratio < 0.5:
        return Signal.FLAT

    # Long: uptrend + pullback + MACD turning up
    if ret_12 > 0 and 40 <= rsi <= 70 and macd_hist > 0:
        return Signal.LONG

    # Short: downtrend + pullback + MACD turning down
    if ret_12 < 0 and 30 <= rsi <= 60 and macd_hist < 0:
        return Signal.SHORT

    return Signal.FLAT


def _mean_revert_signal(df: pd.DataFrame, i: int) -> int:
    """Fade overextended moves in mean-reversion regime.

    Logic:
    - Z-score < -2.0 + RSI < 30 → LONG (oversold bounce)
    - Z-score > 2.0 + RSI > 70  → SHORT (overbought fade)
    - Require volume confirmation
    """
    zscore = df["zscore_20"].iloc[i]
    rsi = df["rsi_14"].iloc[i]
    vol_ratio = df["vol_ratio"].iloc[i]

    if pd.isna(zscore) or pd.isna(rsi):
        return Signal.FLAT

    if vol_ratio < 0.5:
        return Signal.FLAT

    # Long: oversold
    if zscore < -2.0 and rsi < 30:
        return Signal.LONG

    # Short: overbought
    if zscore > 2.0 and rsi > 70:
        return Signal.SHORT

    return Signal.FLAT
