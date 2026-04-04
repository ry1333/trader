"""Extended AI feature extraction — builds on core features with ML-specific additions.

Similar to Morgan bot's 41-feature approach: combines rule-based signals
with engineered features so the ML model learns what distinguishes winners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engine import FEATURE_COLS, compute_features
from src.strategy.regime import Regime, classify_regimes


def extract_ai_features(df: pd.DataFrame, bar_idx: int) -> dict:
    """Extract full AI feature vector for a single bar.

    Returns a dict of ~35 features covering:
    - Core technical features (from engine)
    - Regime context
    - Momentum quality
    - Volatility regime
    - Volume profile
    - Session context
    - Setup quality scoring
    """
    row = df.iloc[bar_idx]
    features: dict = {}

    # ── Core features from engine ─────────────────────────────────────
    for col in FEATURE_COLS:
        val = row.get(col, np.nan)
        features[col] = float(val) if not pd.isna(val) else 0.0

    # ── Regime context ────────────────────────────────────────────────
    features["regime"] = float(row.get("regime", Regime.MEAN_REVERT))
    features["regime_is_trend"] = 1.0 if features["regime"] == Regime.TREND else 0.0
    features["regime_is_stress"] = 1.0 if features["regime"] == Regime.STRESS else 0.0

    # ── Momentum quality (multi-timeframe agreement) ──────────────────
    ret_1 = features.get("ret_1", 0)
    ret_3 = features.get("ret_3", 0)
    ret_6 = features.get("ret_6", 0)
    ret_12 = features.get("ret_12", 0)

    # All returns same direction = strong momentum
    signs = [np.sign(ret_1), np.sign(ret_3), np.sign(ret_6), np.sign(ret_12)]
    features["momentum_agreement"] = abs(sum(signs)) / 4.0

    # Momentum acceleration: recent > older
    features["momentum_accel"] = (abs(ret_1) - abs(ret_3) / 3) if ret_3 != 0 else 0.0

    # ── Volatility regime features ────────────────────────────────────
    rvol_12 = features.get("rvol_12", 0)
    rvol_50 = features.get("rvol_50", 0)
    features["vol_ratio_12_50"] = rvol_12 / rvol_50 if rvol_50 > 0 else 1.0
    features["vol_expanding"] = 1.0 if rvol_12 > rvol_50 * 1.5 else 0.0
    features["vol_contracting"] = 1.0 if rvol_12 < rvol_50 * 0.7 else 0.0

    # ATR relative to price (normalized volatility)
    atr_14 = features.get("atr_14", 0)
    close = row.get("close", 1)
    features["atr_pct"] = atr_14 / close if close > 0 else 0.0

    # ── Volume profile ────────────────────────────────────────────────
    vol_ratio = features.get("vol_ratio", 1.0)
    features["volume_spike"] = 1.0 if vol_ratio > 2.0 else 0.0
    features["volume_dry"] = 1.0 if vol_ratio < 0.5 else 0.0

    # ── Price position features ───────────────────────────────────────
    if bar_idx >= 50:
        lookback = df.iloc[max(0, bar_idx - 50):bar_idx + 1]
        high_50 = lookback["high"].max()
        low_50 = lookback["low"].min()
        rng = high_50 - low_50
        if rng > 0:
            features["price_position_50"] = (close - low_50) / rng  # 0=at low, 1=at high
        else:
            features["price_position_50"] = 0.5

        # Distance from recent high/low
        features["pct_from_50_high"] = (close - high_50) / close if close > 0 else 0.0
        features["pct_from_50_low"] = (close - low_50) / close if close > 0 else 0.0
    else:
        features["price_position_50"] = 0.5
        features["pct_from_50_high"] = 0.0
        features["pct_from_50_low"] = 0.0

    # ── Session context (time-based edge) ─────────────────────────────
    features["is_first_hour"] = features.get("is_open_30m", 0)
    features["is_last_hour"] = features.get("near_close", 0)

    # ── Setup quality score (rule-based, like Morgan's quality bonus) ─
    quality = 0.0

    # Trend alignment: momentum + regime agree
    if features["regime_is_trend"] and features["momentum_agreement"] > 0.5:
        quality += 10.0

    # Volume confirms: above average on signal bar
    if vol_ratio > 1.5:
        quality += 5.0
    elif vol_ratio < 0.5:
        quality -= 5.0

    # Not overextended: z-score within reasonable range
    zscore = abs(features.get("zscore_20", 0))
    if 1.0 < zscore < 2.5:
        quality += 5.0  # Sweet spot for mean reversion
    elif zscore > 3.5:
        quality -= 10.0  # Too extended, risky

    # RSI in favorable zone (not extreme)
    rsi = features.get("rsi_14", 50)
    if 35 <= rsi <= 65:
        quality += 3.0  # Room to move
    elif rsi > 80 or rsi < 20:
        quality -= 5.0  # Extreme

    # Volatility contraction (coiling for breakout)
    if features["vol_contracting"]:
        quality += 5.0

    # MACD histogram momentum
    macd_hist = features.get("macd_hist", 0)
    if abs(macd_hist) > 0:
        quality += 3.0

    features["setup_quality"] = quality

    return features


AI_FEATURE_COLS = (
    FEATURE_COLS
    + [
        "regime", "regime_is_trend", "regime_is_stress",
        "momentum_agreement", "momentum_accel",
        "vol_ratio_12_50", "vol_expanding", "vol_contracting", "atr_pct",
        "volume_spike", "volume_dry",
        "price_position_50", "pct_from_50_high", "pct_from_50_low",
        "is_first_hour", "is_last_hour",
        # setup_quality removed — was double-counted (used in signals AND model.py post-hoc)
    ]
)

# Feature subsets for ensemble models
MOMENTUM_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ew_mom",
    "rsi_14", "macd", "macd_hist", "macd_signal",
    "momentum_agreement", "momentum_accel",
]

MEAN_REVERSION_FEATURES = [
    "zscore_20", "zscore_50", "rsi_14",
    "price_position_50", "pct_from_50_high", "pct_from_50_low",
    "vol_ratio",
]

VOLATILITY_FEATURES = [
    "atr_14", "atr_50", "rvol_12", "rvol_50",
    "vol_ratio_12_50", "vol_expanding", "vol_contracting",
    "range_expansion", "atr_pct",
]
