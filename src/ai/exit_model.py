"""AI-driven exit model — momentum-aware profit management.

Audit finding: trailing stop captures only 25% of peak profit.
$46,327 left on table. Winners avg 25 bars, losers avg 8 bars.

Key insight from data:
- High-capture trades (>70%) exit via take_profit or session_flatten
- Low-capture trades (25-40%) exit via trailing_stop too early
- Winners that run 25+ bars capture the most — need patience

This model uses momentum + exhaustion signals to decide:
1. HOLD — momentum intact, very wide trail (let winners run)
2. TIGHTEN — clear signs of reversal starting, protect profit
3. EXIT — strong reversal confirmed, take profit now
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class ExitAction(IntEnum):
    HOLD = 0
    TIGHTEN = 1
    EXIT = 2


def compute_exit_features(
    df: pd.DataFrame,
    bar_idx: int,
    trade_entry_price: float,
    trade_direction: int,
    trade_peak_profit: float,
    bars_held: int,
    atr: float,
) -> dict:
    """Compute features for exit decision."""
    features = {}
    close = df["close"].iloc[bar_idx]

    # Profit state
    current_pnl = (close - trade_entry_price) * trade_direction
    features["pnl_vs_peak"] = current_pnl / trade_peak_profit if trade_peak_profit > 0 else 0
    features["pnl_atr_ratio"] = current_pnl / atr if atr > 0 else 0
    features["peak_atr_ratio"] = trade_peak_profit / atr if atr > 0 else 0
    features["drawdown_from_peak"] = 1.0 - features["pnl_vs_peak"]  # 0 = at peak, 1 = back to entry

    # Momentum — multi-bar (more robust than single bar)
    if bar_idx >= 5:
        ret_1 = (close - df["close"].iloc[bar_idx - 1]) / df["close"].iloc[bar_idx - 1]
        ret_3 = (close - df["close"].iloc[bar_idx - 3]) / df["close"].iloc[bar_idx - 3]
        ret_5 = (close - df["close"].iloc[bar_idx - 5]) / df["close"].iloc[bar_idx - 5]
        features["ret_1_aligned"] = ret_1 * trade_direction
        features["ret_3_aligned"] = ret_3 * trade_direction
        features["ret_5_aligned"] = ret_5 * trade_direction

        # Count of adverse bars in last 5 (more robust than single bar)
        adverse_bars = 0
        for j in range(1, 6):
            if bar_idx - j >= 0:
                bar_ret = (df["close"].iloc[bar_idx - j + 1] - df["close"].iloc[bar_idx - j])
                if bar_ret * trade_direction < 0:
                    adverse_bars += 1
        features["adverse_bar_count"] = adverse_bars  # 0-5, higher = more reversal
    else:
        features["ret_1_aligned"] = 0
        features["ret_3_aligned"] = 0
        features["ret_5_aligned"] = 0
        features["adverse_bar_count"] = 0

    # RSI exhaustion
    rsi = df.get("rsi_14", pd.Series(50)).iloc[bar_idx] if "rsi_14" in df.columns else 50
    rsi = rsi if not pd.isna(rsi) else 50
    if trade_direction == 1:
        features["rsi_exhaustion"] = max(0, rsi - 70) / 30
    else:
        features["rsi_exhaustion"] = max(0, 30 - rsi) / 30

    # Volatility
    vol_ratio = df.get("vol_ratio", pd.Series(1)).iloc[bar_idx] if "vol_ratio" in df.columns else 1
    features["vol_ratio"] = vol_ratio if not pd.isna(vol_ratio) else 1.0

    # MACD — use histogram trend (3-bar slope), not single value
    if "macd_hist" in df.columns and bar_idx >= 3:
        mh_now = df["macd_hist"].iloc[bar_idx]
        mh_prev = df["macd_hist"].iloc[bar_idx - 3]
        if not pd.isna(mh_now) and not pd.isna(mh_prev):
            features["macd_hist_trend"] = (mh_now - mh_prev) * trade_direction
        else:
            features["macd_hist_trend"] = 0
    else:
        features["macd_hist_trend"] = 0

    # Time
    features["bars_held"] = bars_held

    # VWAP
    vwap = df.get("vwap", pd.Series()).iloc[bar_idx] if "vwap" in df.columns else None
    if vwap is not None and not pd.isna(vwap) and atr > 0:
        features["vwap_dist_atr"] = (close - vwap) / atr * trade_direction
    else:
        features["vwap_dist_atr"] = 0

    # Regime
    regime = df.get("regime", pd.Series(1)).iloc[bar_idx] if "regime" in df.columns else 1
    features["regime_trend"] = 1 if regime == 2 else 0
    features["regime_stress"] = 1 if regime == 0 else 0

    return features


def decide_exit(
    df: pd.DataFrame,
    bar_idx: int,
    trade_entry_price: float,
    trade_direction: int,
    trade_peak_profit: float,
    bars_held: int,
    atr: float,
    target_distance: float,
) -> tuple[ExitAction, float]:
    """Momentum-aware exit decision.

    Returns (action, trail_pct) where trail_pct = fraction of peak to keep.

    Key design principles (from audit):
    1. Winners average 25 bars — be PATIENT
    2. 25% capture rate was the problem — be MORE patient
    3. Only EXIT on CONFIRMED reversal, not single-bar noise
    4. HOLD should have very wide trail (15-20% of peak) to survive normal pullbacks
    5. TIGHTEN only when multiple reversal signals converge
    """
    feat = compute_exit_features(
        df, bar_idx, trade_entry_price, trade_direction,
        trade_peak_profit, bars_held, atr
    )

    current_pnl = (df["close"].iloc[bar_idx] - trade_entry_price) * trade_direction

    # ═══════════════════════════════════════════════════════════════════
    # EXIT — ONLY on strong, multi-signal confirmed reversal
    # Must have 2+ converging signals to exit. Single indicators lie.
    # ═══════════════════════════════════════════════════════════════════
    exit_signals = 0

    # RSI extreme + reversing (>80 long or <20 short, not just >70)
    if feat["rsi_exhaustion"] > 0.7:
        exit_signals += 1

    # Sustained adverse momentum: 4+ of last 5 bars against us
    if feat["adverse_bar_count"] >= 4:
        exit_signals += 1

    # 5-bar return strongly against us (>0.4% = ~80 NQ points)
    if feat["ret_5_aligned"] < -0.004:
        exit_signals += 1

    # MACD histogram deteriorating (3-bar trend turning)
    if feat["macd_hist_trend"] < -0.5:
        exit_signals += 1

    # Lost VWAP significantly (>1 ATR wrong side)
    if feat["vwap_dist_atr"] < -1.0:
        exit_signals += 1

    # Stress regime with profit pulling back
    if feat["regime_stress"] and feat["drawdown_from_peak"] > 0.40:
        exit_signals += 1

    # REQUIRE 2+ converging signals to exit
    if exit_signals >= 2 and current_pnl > 0:
        return ExitAction.EXIT, 0.0

    # Massive reversal — 3 signals = always exit
    if exit_signals >= 3:
        return ExitAction.EXIT, 0.0

    # ═══════════════════════════════════════════════════════════════════
    # TIGHTEN — some reversal signs, protect but don't exit yet
    # ═══════════════════════════════════════════════════════════════════
    tighten_signals = 0

    # MACD starting to fade (single signal, not yet confirmed)
    if feat["macd_hist_trend"] < -0.2:
        tighten_signals += 1

    # Held very long (>35 bars) — time decay of edge
    if bars_held > 35:
        tighten_signals += 1

    # At or past target — lock it in
    if current_pnl > target_distance * 0.85:
        tighten_signals += 1

    # High volume bar against us (potential reversal starting)
    if feat["vol_ratio"] > 2.5 and feat["ret_1_aligned"] < -0.001:
        tighten_signals += 1

    if tighten_signals >= 2:
        return ExitAction.TIGHTEN, 0.55  # Keep 55% of peak

    if tighten_signals == 1 and feat["drawdown_from_peak"] > 0.35:
        return ExitAction.TIGHTEN, 0.50  # Already pulling back + one warning

    # ═══════════════════════════════════════════════════════════════════
    # HOLD — momentum intact, give maximum room
    # Normal NQ pullbacks during trends are 20-30% of the move.
    # Trail must be wider than that or we capture nothing.
    # ═══════════════════════════════════════════════════════════════════

    # Strong momentum: 5-bar return positive + MACD improving
    if feat["ret_5_aligned"] > 0.001 and feat["macd_hist_trend"] > 0:
        return ExitAction.HOLD, 0.15  # Very wide — only exit on 85% drawdown from peak

    # Trend regime + above VWAP — maximum patience
    if feat["regime_trend"] and feat["vwap_dist_atr"] > 0.5:
        return ExitAction.HOLD, 0.15

    # Decent momentum: 3-bar still positive
    if feat["ret_3_aligned"] > 0:
        return ExitAction.HOLD, 0.20  # Wide trail

    # Default: moderate hold
    return ExitAction.HOLD, 0.25  # Still wider than old 40%
