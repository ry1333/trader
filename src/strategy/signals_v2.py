"""V2 signal generation — ES-specific intraday patterns.

Three signal types based on regime + session context:

1. Opening Range Breakout (ORB): Trade the first breakout of the 30-min OR
   - Only during first 2 hours of RTH
   - Requires volume confirmation
   - Best edge: catches institutional order flow

2. VWAP Reversion: Fade extended moves back to VWAP
   - Price extended > 1 ATR from VWAP + reversal candle
   - Best in midday chop / range-bound regimes
   - Mean-reversion edge

3. Trend Continuation: Pullback entries in established trends
   - Price above VWAP + above OR + pullback to support
   - RSI not overbought + MACD confirming
   - Momentum edge in trend regimes
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


class SignalType(IntEnum):
    NONE = 0
    ORB = 1
    VWAP_REVERSION = 2
    TREND_CONTINUATION = 3


def generate_signals_v2(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate V2 signals with signal type tracking.

    Returns (signals, signal_types) Series.
    """
    signals = pd.Series(Signal.FLAT, index=df.index, dtype=int)
    signal_types = pd.Series(SignalType.NONE, index=df.index, dtype=int)

    for i in range(2, len(df)):
        regime = df["regime"].iloc[i]

        # No trading in stress regime
        if regime == Regime.STRESS:
            continue

        # No trading near close
        if df.get("near_close", pd.Series()).iloc[i] if "near_close" in df.columns else False:
            continue

        # No new entries during midday chop (low edge)
        if df.get("is_midday_chop", pd.Series()).iloc[i] if "is_midday_chop" in df.columns else False:
            continue

        # Try each signal type in priority order
        sig, stype = _try_orb(df, i)
        if sig != Signal.FLAT:
            signals.iloc[i] = sig
            signal_types.iloc[i] = stype
            continue

        sig, stype = _try_vwap_reversion(df, i)
        if sig != Signal.FLAT:
            signals.iloc[i] = sig
            signal_types.iloc[i] = stype
            continue

        sig, stype = _try_trend_continuation(df, i)
        if sig != Signal.FLAT:
            signals.iloc[i] = sig
            signal_types.iloc[i] = stype

    return signals, signal_types


def _try_orb(df: pd.DataFrame, i: int) -> tuple[int, int]:
    """Opening Range Breakout — first 2 hours of RTH only.

    Long:  close breaks above OR high + volume spike
    Short: close breaks below OR low + volume spike
    """
    # Only during RTH, after opening drive is established
    is_rth = df.get("is_rth", pd.Series()).iloc[i] if "is_rth" in df.columns else 0
    is_open = df.get("is_open_drive", pd.Series()).iloc[i] if "is_open_drive" in df.columns else 0

    if not is_rth or is_open:
        return Signal.FLAT, SignalType.NONE

    # Need OR data
    or_high = df.get("or_high", pd.Series()).iloc[i] if "or_high" in df.columns else None
    or_low = df.get("or_low", pd.Series()).iloc[i] if "or_low" in df.columns else None

    if or_high is None or or_low is None or pd.isna(or_high) or pd.isna(or_low):
        return Signal.FLAT, SignalType.NONE

    close = df["close"].iloc[i]
    prev_close = df["close"].iloc[i - 1]
    vol_ratio = df.get("vol_ratio", pd.Series(1.0)).iloc[i] if "vol_ratio" in df.columns else 1.0

    if pd.isna(vol_ratio):
        vol_ratio = 1.0

    # Require volume above average
    if vol_ratio < 1.0:
        return Signal.FLAT, SignalType.NONE

    # Breakout long: close above OR high with momentum
    if close > or_high and (close - or_high) / or_high < 0.005:  # Within 0.5% of breakout
        return Signal.LONG, SignalType.ORB

    # Breakout short: close below OR low with momentum
    if close < or_low and (or_low - close) / or_low < 0.005:
        return Signal.SHORT, SignalType.ORB

    return Signal.FLAT, SignalType.NONE


def _try_vwap_reversion(df: pd.DataFrame, i: int) -> tuple[int, int]:
    """VWAP Reversion — fade extended moves back to VWAP.

    Long:  price well below VWAP + oversold + reversal candle
    Short: price well above VWAP + overbought + reversal candle
    """
    if "price_vs_vwap" not in df.columns or "vwap" not in df.columns:
        return Signal.FLAT, SignalType.NONE

    pvv = df["price_vs_vwap"].iloc[i]
    rsi = df.get("rsi_14", pd.Series(50.0)).iloc[i] if "rsi_14" in df.columns else 50.0
    atr = df.get("atr_14", pd.Series(0.0)).iloc[i] if "atr_14" in df.columns else 0.0

    if pd.isna(pvv) or pd.isna(rsi) or pd.isna(atr) or atr <= 0:
        return Signal.FLAT, SignalType.NONE

    close = df["close"].iloc[i]
    prev_close = df["close"].iloc[i - 1]
    vwap = df["vwap"].iloc[i]

    if pd.isna(vwap) or vwap <= 0:
        return Signal.FLAT, SignalType.NONE

    # Distance from VWAP in ATR units
    dist_atr = abs(close - vwap) / atr

    # Only in mean-revert regime
    regime = df.get("regime", pd.Series(Regime.MEAN_REVERT)).iloc[i]
    if regime != Regime.MEAN_REVERT:
        return Signal.FLAT, SignalType.NONE

    # Long: price > 1.0 ATR below VWAP + oversold + reversal (close > prev close)
    if close < vwap and dist_atr > 1.0 and rsi < 40 and close > prev_close:
        return Signal.LONG, SignalType.VWAP_REVERSION

    # Short: price > 1.0 ATR above VWAP + overbought + reversal
    if close > vwap and dist_atr > 1.0 and rsi > 60 and close < prev_close:
        return Signal.SHORT, SignalType.VWAP_REVERSION

    return Signal.FLAT, SignalType.NONE


def _try_trend_continuation(df: pd.DataFrame, i: int) -> tuple[int, int]:
    """Trend Continuation — pullback entries in established trends.

    Long:  uptrend (above VWAP + above OR) + pullback (RSI dip) + MACD positive
    Short: downtrend (below VWAP + below OR) + bounce (RSI pop) + MACD negative
    """
    if "vwap" not in df.columns:
        return Signal.FLAT, SignalType.NONE

    # Only in trend regime
    regime = df.get("regime", pd.Series(Regime.MEAN_REVERT)).iloc[i]
    if regime != Regime.TREND:
        return Signal.FLAT, SignalType.NONE

    close = df["close"].iloc[i]
    vwap = df.get("vwap", pd.Series()).iloc[i] if "vwap" in df.columns else np.nan
    above_or = df.get("above_or", pd.Series(0)).iloc[i] if "above_or" in df.columns else 0
    below_or = df.get("below_or", pd.Series(0)).iloc[i] if "below_or" in df.columns else 0
    rsi = df.get("rsi_14", pd.Series(50.0)).iloc[i] if "rsi_14" in df.columns else 50.0
    macd_hist = df.get("macd_hist", pd.Series(0.0)).iloc[i] if "macd_hist" in df.columns else 0.0
    vol_ratio = df.get("vol_ratio", pd.Series(1.0)).iloc[i] if "vol_ratio" in df.columns else 1.0
    ret_12 = df.get("ret_12", pd.Series(0.0)).iloc[i] if "ret_12" in df.columns else 0.0

    if any(pd.isna(x) for x in [vwap, rsi, macd_hist, vol_ratio, ret_12]):
        return Signal.FLAT, SignalType.NONE

    # Minimum volume
    if vol_ratio < 0.7:
        return Signal.FLAT, SignalType.NONE

    # Long: price above VWAP + pullback (RSI 35-65) + MACD up + momentum up
    if close > vwap and 35 <= rsi <= 65 and macd_hist > 0 and ret_12 > 0:
        return Signal.LONG, SignalType.TREND_CONTINUATION

    # Short: price below VWAP + bounce (RSI 35-65) + MACD down + momentum down
    if close < vwap and 35 <= rsi <= 65 and macd_hist < 0 and ret_12 < 0:
        return Signal.SHORT, SignalType.TREND_CONTINUATION

    return Signal.FLAT, SignalType.NONE
