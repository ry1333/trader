"""Meta-labeling model — filters false positives from the primary strategy.

Architecture (from research):
1. Primary model finds trade setups (signals_v3 + quality_model)
2. Meta-model decides: take / reduce / skip for each setup
3. Uses DIFFERENT features than primary — focused on regime context

Key insight: the primary model uses 5-min features and is good at finding
local setups. The meta-model uses HIGHER TIMEFRAME context to catch
regime mismatches that 5-min features miss.

Triple-barrier labeling: trades labeled by actual outcome
(hit TP, hit SL, or timed out) not just next-bar return.

Side asymmetry: separate thresholds for longs vs shorts
because bear-bounce longs fail differently than trend shorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_htf_regime_features(df: pd.DataFrame, bar_idx: int) -> dict:
    """Compute higher-timeframe regime features.

    These capture multi-session context that 5-min features miss:
    - 30-min trend (6 bars)
    - 2-hour trend (24 bars)
    - 1-session trend (78 bars, ~6.5 hours RTH)
    - Multi-session trend (200 bars, ~2 sessions)
    - Volatility regime
    - Overnight gap context
    """
    features = {}

    if bar_idx < 200:
        return {
            "htf_30m_slope": 0, "htf_2h_slope": 0, "htf_session_slope": 0,
            "htf_multi_session_slope": 0, "htf_vol_regime": 0,
            "htf_trend_alignment": 0, "htf_bearish_count": 0, "htf_bullish_count": 0,
        }

    close = df["close"].iloc[bar_idx]

    # 30-min trend (6 bars of 5-min)
    ret_6 = (close - df["close"].iloc[bar_idx - 6]) / df["close"].iloc[bar_idx - 6]
    features["htf_30m_slope"] = ret_6

    # 2-hour trend (24 bars)
    ret_24 = (close - df["close"].iloc[bar_idx - 24]) / df["close"].iloc[bar_idx - 24]
    features["htf_2h_slope"] = ret_24

    # 1-session trend (78 bars)
    ret_78 = (close - df["close"].iloc[bar_idx - 78]) / df["close"].iloc[bar_idx - 78]
    features["htf_session_slope"] = ret_78

    # Multi-session trend (200 bars)
    ret_200 = (close - df["close"].iloc[bar_idx - 200]) / df["close"].iloc[bar_idx - 200]
    features["htf_multi_session_slope"] = ret_200

    # Volatility regime: current ATR vs 50-bar ATR median
    if "atr_14" in df.columns and "atr_50" in df.columns:
        atr = df["atr_14"].iloc[bar_idx]
        atr_50 = df["atr_50"].iloc[bar_idx]
        if not pd.isna(atr) and not pd.isna(atr_50) and atr_50 > 0:
            features["htf_vol_regime"] = atr / atr_50  # >1.5 = high vol
        else:
            features["htf_vol_regime"] = 1.0
    else:
        features["htf_vol_regime"] = 1.0

    # Trend alignment: how many timeframes agree on direction?
    slopes = [ret_6, ret_24, ret_78, ret_200]
    bullish = sum(1 for s in slopes if s > 0.001)
    bearish = sum(1 for s in slopes if s < -0.001)
    features["htf_trend_alignment"] = bullish - bearish  # -4 to +4
    features["htf_bearish_count"] = bearish
    features["htf_bullish_count"] = bullish

    return features


def meta_gate_decision(
    features: dict,
    signal_direction: int,
    primary_confidence: float,
) -> tuple[str, float]:
    """Meta-gate: decide take/reduce/skip based on HTF context.

    Returns (action, size_multiplier):
    - ("take", 1.0-1.3) — good alignment
    - ("reduce", 0.3-0.7) — some mismatch
    - ("skip", 0.0) — strong mismatch

    Key rules (from the research plan):
    - Counter-trend longs in bearish multi-timeframe: require much higher confidence
    - Volatility expansion: reduce all size mechanically
    - Perfect alignment: boost size
    """
    htf_align = features.get("htf_trend_alignment", 0)  # -4 to +4
    htf_vol = features.get("htf_vol_regime", 1.0)
    htf_200 = features.get("htf_multi_session_slope", 0)
    htf_78 = features.get("htf_session_slope", 0)
    bearish_count = features.get("htf_bearish_count", 0)
    bullish_count = features.get("htf_bullish_count", 0)

    # ── SKIP conditions ───────────────────────────────────────────────

    # Long in multi-session downtrend: the key March fix
    # If 200-bar (multi-session) slope is significantly negative, block longs
    # regardless of short-term bounce signals
    htf_200 = features.get("htf_multi_session_slope", 0)
    # Use BOTH multi-session (200-bar) and session (78-bar) slopes
    htf_78 = features.get("htf_session_slope", 0)
    if signal_direction == 1 and (htf_200 < -0.005 or htf_78 < -0.005):
        return "skip", 0.0  # Downtrend on session or multi-session — don't buy

    if signal_direction == -1 and (htf_200 > 0.005 or htf_78 > 0.005):
        return "skip", 0.0  # Uptrend on session or multi-session — don't short

    # Also skip if 3+ timeframes agree against trade direction
    if signal_direction == 1 and bearish_count >= 3:
        return "skip", 0.0

    if signal_direction == -1 and bullish_count >= 3:
        return "skip", 0.0

    # Very high volatility (>2x normal) — reduce everything
    if htf_vol > 2.0:
        return "reduce", 0.3

    # ── REDUCE conditions ─────────────────────────────────────────────

    # Long in moderate bear (2 timeframes bearish)
    if signal_direction == 1 and bearish_count >= 2:
        return "reduce", 0.4  # Significant reduction

    # Short in moderate bull
    if signal_direction == -1 and bullish_count >= 2:
        return "reduce", 0.4

    # High vol (1.5-2x normal) — moderate reduction
    if htf_vol > 1.5:
        return "reduce", 0.7

    # ── TAKE / BOOST conditions ───────────────────────────────────────

    # Perfect alignment: all timeframes agree with signal direction
    if signal_direction == 1 and bullish_count >= 3:
        return "take", 1.3  # Boost

    if signal_direction == -1 and bearish_count >= 3:
        return "take", 1.3  # Boost

    # Good alignment: 2+ timeframes agree
    if signal_direction == 1 and bullish_count >= 2 and bearish_count == 0:
        return "take", 1.1

    if signal_direction == -1 and bearish_count >= 2 and bullish_count == 0:
        return "take", 1.1

    # Default: normal
    return "take", 1.0
