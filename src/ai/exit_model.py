"""AI-driven exit model — predicts optimal hold behavior for each trade.

The audit showed:
- Trailing stop captures only 25% of peak profit ($138 avg from $524 peak)
- Take-profit captures 119% (trades that hit TP naturally)
- $12,195+ left on table from trailing exits alone

This model predicts at each bar whether a winning trade should:
1. HOLD — momentum intact, let it run (wide trail or no trail)
2. TIGHTEN — momentum fading, tighten trail
3. EXIT — take profit now, reversal likely

Trained on: for each bar of each historical trade, label what the OPTIMAL
action would have been (based on knowing future outcome).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class ExitAction(IntEnum):
    HOLD = 0      # Keep wide trail, let run
    TIGHTEN = 1   # Tighten trail to 50% of peak
    EXIT = 2      # Take profit now


def compute_exit_features(
    df: pd.DataFrame,
    bar_idx: int,
    trade_entry_price: float,
    trade_direction: int,
    trade_peak_profit: float,
    bars_held: int,
    atr: float,
) -> dict:
    """Compute features for exit decision at a specific bar during a trade.

    These features describe the current state of a winning trade:
    - How much profit we have vs peak
    - Momentum indicators (is the move continuing?)
    - Volatility (is vol expanding or contracting?)
    - Time factors (how long have we held?)
    """
    features = {}
    close = df["close"].iloc[bar_idx]

    # Profit state
    current_pnl = (close - trade_entry_price) * trade_direction
    features["pnl_vs_peak"] = current_pnl / trade_peak_profit if trade_peak_profit > 0 else 0
    features["pnl_atr_ratio"] = current_pnl / atr if atr > 0 else 0
    features["peak_atr_ratio"] = trade_peak_profit / atr if atr > 0 else 0

    # Momentum — is the move still going?
    if bar_idx >= 3:
        ret_1 = (close - df["close"].iloc[bar_idx - 1]) / df["close"].iloc[bar_idx - 1]
        ret_3 = (close - df["close"].iloc[bar_idx - 3]) / df["close"].iloc[bar_idx - 3]
        features["ret_1_aligned"] = ret_1 * trade_direction  # Positive = moving in our favor
        features["ret_3_aligned"] = ret_3 * trade_direction
    else:
        features["ret_1_aligned"] = 0
        features["ret_3_aligned"] = 0

    # RSI state
    rsi = df.get("rsi_14", pd.Series(50)).iloc[bar_idx] if "rsi_14" in df.columns else 50
    features["rsi"] = rsi if not pd.isna(rsi) else 50
    # RSI overbought/oversold relative to direction
    if trade_direction == 1:
        features["rsi_exhaustion"] = max(0, rsi - 70) / 30  # 0 = fine, 1 = very overbought
    else:
        features["rsi_exhaustion"] = max(0, 30 - rsi) / 30  # 0 = fine, 1 = very oversold

    # Volatility state
    rvol = df.get("rvol_12", pd.Series(0)).iloc[bar_idx] if "rvol_12" in df.columns else 0
    rvol_50 = df.get("rvol_50", pd.Series(0)).iloc[bar_idx] if "rvol_50" in df.columns else 0
    features["vol_expanding"] = rvol / rvol_50 if rvol_50 > 0 and not pd.isna(rvol) and not pd.isna(rvol_50) else 1.0

    # Volume
    vol_ratio = df.get("vol_ratio", pd.Series(1)).iloc[bar_idx] if "vol_ratio" in df.columns else 1
    features["vol_ratio"] = vol_ratio if not pd.isna(vol_ratio) else 1.0

    # MACD momentum
    macd_hist = df.get("macd_hist", pd.Series(0)).iloc[bar_idx] if "macd_hist" in df.columns else 0
    features["macd_aligned"] = (macd_hist * trade_direction) if not pd.isna(macd_hist) else 0

    # Time in trade
    features["bars_held"] = bars_held
    features["bars_held_pct"] = min(1.0, bars_held / 48)  # Normalized to max hold

    # VWAP relationship
    vwap = df.get("vwap", pd.Series()).iloc[bar_idx] if "vwap" in df.columns else None
    if vwap is not None and not pd.isna(vwap) and atr > 0:
        features["vwap_dist_atr"] = (close - vwap) / atr * trade_direction  # Positive = favorable
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
    """Rule-based exit decision using momentum + exhaustion signals.

    Returns (action, trail_pct) where trail_pct is how much of peak to keep.

    This is a rule-based system (not ML) because:
    1. ML exit models overfit badly (label leakage from future data)
    2. Rule-based is more robust across regimes
    3. The rules are derived from the audit data patterns
    """
    feat = compute_exit_features(
        df, bar_idx, trade_entry_price, trade_direction,
        trade_peak_profit, bars_held, atr
    )

    current_pnl = (df["close"].iloc[bar_idx] - trade_entry_price) * trade_direction

    # ── EXIT signals (take profit now) ────────────────────────────────
    # RSI exhaustion + momentum reversal
    if feat["rsi_exhaustion"] > 0.5 and feat["ret_1_aligned"] < 0:
        return ExitAction.EXIT, 0.0

    # Strong adverse momentum (3-bar reversal)
    if feat["ret_3_aligned"] < -0.002 and current_pnl > 0:
        return ExitAction.EXIT, 0.0

    # Volume spike against us (potential reversal)
    if feat["vol_ratio"] > 2.0 and feat["ret_1_aligned"] < 0:
        return ExitAction.EXIT, 0.0

    # Stress regime — exit winning trades
    if feat["regime_stress"] and bars_held > 3:
        return ExitAction.EXIT, 0.0

    # ── TIGHTEN signals (trail at 60% of peak) ───────────────────────
    # Momentum fading: MACD turning against us
    if feat["macd_aligned"] < 0 and feat["peak_atr_ratio"] > 1.0:
        return ExitAction.TIGHTEN, 0.60

    # Extended in time (>30 bars) — tighten but don't exit
    if bars_held > 30 and feat["ret_1_aligned"] < 0:
        return ExitAction.TIGHTEN, 0.55

    # At or beyond target — tighten to lock in
    if current_pnl > target_distance * 0.90:
        return ExitAction.TIGHTEN, 0.70

    # ── HOLD signals (wide trail at 30% of peak or no trail) ──────────
    # Strong momentum continuing
    if feat["ret_3_aligned"] > 0.001 and feat["macd_aligned"] > 0:
        return ExitAction.HOLD, 0.25  # Very wide trail

    # Trend regime + favorable VWAP — let it run
    if feat["regime_trend"] and feat["vwap_dist_atr"] > 0:
        return ExitAction.HOLD, 0.30

    # Default: moderate trail
    return ExitAction.HOLD, 0.35
