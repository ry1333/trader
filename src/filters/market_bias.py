"""Market bias detector — shifts strategy between bullish/bearish/neutral modes.

Problem: In March 2026 (-3.2% selloff), the system took 10 longs (-$827) vs 8 shorts (-$115).
Longs were buying dips into a waterfall because the 50-EMA updates too slowly.

Solution: Fast-reacting bias indicator that detects bearish conditions and:
1. Blocks or reduces longs in bearish mode
2. Increases short signal weight in bearish mode
3. Tightens stops on longs, widens on shorts

Uses multiple timeframes to detect bias shift:
- 20-bar EMA slope (fast reaction)
- 50-bar EMA slope (medium)
- Price vs 50 EMA position
- Recent high/low breakout direction
- Session trend (open to current)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class MarketBias(IntEnum):
    STRONG_BEAR = -2
    BEAR = -1
    NEUTRAL = 0
    BULL = 1
    STRONG_BULL = 2


def compute_market_bias(df: pd.DataFrame, bar_idx: int) -> tuple[MarketBias, float]:
    """Compute market directional bias at a given bar.

    Returns (bias, confidence) where confidence is 0-1.

    Uses fast-reacting signals that adapt within 1-2 sessions,
    not slow moving averages that take weeks.
    """
    if bar_idx < 60:
        return MarketBias.NEUTRAL, 0.5

    close = df["close"].iloc[bar_idx]
    score = 0.0  # -5 to +5 scale

    # ── 1. 20-bar return (last 100 min) — fast reaction ──────────────
    if bar_idx >= 20:
        ret_20 = (close - df["close"].iloc[bar_idx - 20]) / df["close"].iloc[bar_idx - 20]
        if ret_20 > 0.003:
            score += 1.0
        elif ret_20 < -0.003:
            score -= 1.0
        if ret_20 > 0.008:
            score += 0.5
        elif ret_20 < -0.008:
            score -= 0.5

    # ── 2. 50-bar return (last 250 min, ~1 session) ──────────────────
    if bar_idx >= 50:
        ret_50 = (close - df["close"].iloc[bar_idx - 50]) / df["close"].iloc[bar_idx - 50]
        if ret_50 > 0.005:
            score += 1.0
        elif ret_50 < -0.005:
            score -= 1.0
        if ret_50 > 0.015:
            score += 0.5
        elif ret_50 < -0.015:
            score -= 0.5

    # ── 3. Lower highs / lower lows vs higher highs / higher lows ────
    if bar_idx >= 40:
        recent_high = df["high"].iloc[bar_idx - 15:bar_idx + 1].max()
        prev_high = df["high"].iloc[bar_idx - 40:bar_idx - 15].max()
        recent_low = df["low"].iloc[bar_idx - 15:bar_idx + 1].min()
        prev_low = df["low"].iloc[bar_idx - 40:bar_idx - 15].min()

        if recent_high < prev_high and recent_low < prev_low:
            score -= 1.5  # Clear downtrend structure
        elif recent_high > prev_high and recent_low > prev_low:
            score += 1.5  # Clear uptrend structure

    # ── 4. Price vs recent VWAP trend ─────────────────────────────────
    if "vwap" in df.columns:
        vwap = df["vwap"].iloc[bar_idx]
        if not pd.isna(vwap) and vwap > 0:
            dist = (close - vwap) / vwap
            if dist > 0.003:
                score += 0.5
            elif dist < -0.003:
                score -= 0.5

    # ── 5. Multi-session trend (200-bar = ~2 full sessions) ─────────
    # This catches multi-day selloffs that the fast indicators miss on bounces
    if bar_idx >= 200:
        ret_200 = (close - df["close"].iloc[bar_idx - 200]) / df["close"].iloc[bar_idx - 200]
        if ret_200 > 0.02:
            score += 1.5  # Strong multi-session uptrend
        elif ret_200 > 0.008:
            score += 0.5
        elif ret_200 < -0.02:
            score -= 1.5  # Strong multi-session downtrend
        elif ret_200 < -0.008:
            score -= 0.5

    # ── Convert to bias ───────────────────────────────────────────────
    confidence = min(1.0, abs(score) / 3.0)

    if score >= 3.0:
        return MarketBias.STRONG_BULL, confidence
    elif score >= 1.5:
        return MarketBias.BULL, confidence
    elif score <= -3.0:
        return MarketBias.STRONG_BEAR, confidence
    elif score <= -1.5:
        return MarketBias.BEAR, confidence
    else:
        return MarketBias.NEUTRAL, confidence


def get_direction_filter(bias: MarketBias, signal_direction: int) -> tuple[bool, float]:
    """Filter and size trades based on market bias.

    Returns (allow_trade, size_multiplier).

    Rules:
    - STRONG_BEAR: block all longs, 1.3x shorts
    - BEAR: reduce longs to 0.5x, normal shorts
    - NEUTRAL: normal everything
    - BULL: normal longs, reduce shorts to 0.5x
    - STRONG_BULL: 1.3x longs, block all shorts
    """
    if bias == MarketBias.STRONG_BEAR:
        if signal_direction == 1:  # Long
            return False, 0.0  # Block longs
        return True, 1.3  # Boost shorts

    elif bias == MarketBias.BEAR:
        if signal_direction == 1:
            return True, 0.5  # Reduce longs
        return True, 1.1  # Slight boost shorts

    elif bias == MarketBias.STRONG_BULL:
        if signal_direction == -1:  # Short
            return False, 0.0  # Block shorts
        return True, 1.3  # Boost longs

    elif bias == MarketBias.BULL:
        if signal_direction == -1:
            return True, 0.5  # Reduce shorts
        return True, 1.1  # Slight boost longs

    # NEUTRAL
    return True, 1.0
