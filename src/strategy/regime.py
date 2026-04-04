"""Regime classifier — ML-based with hysteresis.

Detects trend / mean-reversion / stress using:
- ADX (trend strength)
- Hurst exponent approximation (trending vs mean-reverting)
- ATR ratio (current vs historical vol)
- Autocorrelation of returns
- Range compression
- Outputs regime + confidence score
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Regime(IntEnum):
    STRESS = 0
    MEAN_REVERT = 1
    TREND = 2


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out when the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx


def _compute_hurst(series: pd.Series, window: int = 50) -> pd.Series:
    """Approximate Hurst exponent using rescaled range (R/S) method."""
    def rs_hurst(x):
        if len(x) < 20:
            return 0.5
        y = x - x.mean()
        cumdev = np.cumsum(y)
        R = cumdev.max() - cumdev.min()
        S = x.std()
        if S == 0 or R == 0:
            return 0.5
        return np.log(R / S) / np.log(len(x))

    return series.rolling(window).apply(rs_hurst, raw=True)


def _compute_autocorr(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling autocorrelation of returns."""
    return returns.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else 0, raw=True
    )


def classify_regimes(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Classify regimes using multiple indicators + hysteresis.

    Returns (regime_series, confidence_series).
    """
    n = len(df)
    regime = pd.Series(Regime.MEAN_REVERT, index=df.index, dtype=int)
    confidence = pd.Series(0.5, index=df.index, dtype=float)

    # Compute regime indicators
    adx = _compute_adx(df, 14)
    ret_1 = df["close"].pct_change()
    hurst = _compute_hurst(ret_1, 50)
    atr_ratio = df["atr_14"] / df["atr_50"].replace(0, np.nan) if "atr_50" in df.columns else pd.Series(1.0, index=df.index)
    autocorr = _compute_autocorr(ret_1, 20)

    # Range compression: 20-bar range / ATR
    rolling_high = df["high"].rolling(20).max()
    rolling_low = df["low"].rolling(20).min()
    range_comp = (rolling_high - rolling_low) / df["atr_14"].replace(0, np.nan) if "atr_14" in df.columns else pd.Series(1.0, index=df.index)

    # Score each regime (0-1 scale)
    # TREND score: ADX high + Hurst > 0.55 + positive autocorrelation
    trend_score = pd.Series(0.0, index=df.index)
    trend_score += (adx / 50).clip(0, 1) * 0.4  # ADX contribution (0-50 → 0-0.4)
    trend_score += ((hurst - 0.45) / 0.2).clip(0, 1) * 0.3  # Hurst contribution
    trend_score += (autocorr.clip(0, 0.5) / 0.5) * 0.3  # Autocorr contribution

    # MEAN_REVERT score: low ADX + Hurst < 0.45 + negative autocorrelation
    mr_score = pd.Series(0.0, index=df.index)
    mr_score += ((50 - adx) / 50).clip(0, 1) * 0.3  # Low ADX
    mr_score += ((0.55 - hurst) / 0.2).clip(0, 1) * 0.3  # Low Hurst
    mr_score += ((-autocorr).clip(0, 0.5) / 0.5) * 0.2  # Negative autocorr
    mr_score += ((range_comp - 5).clip(0, 10) / 10) * 0.2  # Wide range = mean reverting

    # STRESS score: ATR ratio spike + extreme range expansion
    stress_score = pd.Series(0.0, index=df.index)
    stress_score += ((atr_ratio - 1.0) / 1.0).clip(0, 1) * 0.5  # ATR expanding
    rvol_ratio = df["rvol_12"] / df["rvol_50"].replace(0, np.nan) if "rvol_12" in df.columns and "rvol_50" in df.columns else pd.Series(1.0, index=df.index)
    stress_score += ((rvol_ratio - 1.5) / 1.5).clip(0, 1) * 0.5  # Vol spike

    # Assign regime based on highest score
    scores = pd.DataFrame({
        "trend": trend_score,
        "mr": mr_score,
        "stress": stress_score,
    })

    # Fill NaN scores with 0 before computing max
    scores = scores.fillna(0)
    max_score = scores.max(axis=1)
    regime_raw = scores.idxmax(axis=1)
    regime_map = {"trend": Regime.TREND, "mr": Regime.MEAN_REVERT, "stress": Regime.STRESS}

    # Apply hysteresis: require minimum 6 bars and confidence > 0.55 to switch
    MIN_BARS = 6
    MIN_CONFIDENCE = 0.55
    current_regime = Regime.MEAN_REVERT
    bars_in_regime = 0

    for i in range(n):
        proposed = regime_map.get(regime_raw.iloc[i], Regime.MEAN_REVERT) if not pd.isna(regime_raw.iloc[i]) else Regime.MEAN_REVERT
        score = max_score.iloc[i] if not pd.isna(max_score.iloc[i]) else 0.5

        if proposed != current_regime:
            if bars_in_regime >= MIN_BARS and score >= MIN_CONFIDENCE:
                current_regime = proposed
                bars_in_regime = 0
        else:
            bars_in_regime += 1

        regime.iloc[i] = current_regime
        confidence.iloc[i] = score

    return regime, confidence


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime and confidence columns to DataFrame."""
    df = df.copy()
    df["regime"], df["regime_confidence"] = classify_regimes(df)
    return df
