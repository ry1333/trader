"""Session quality filter — classifies trading conditions and manages no-trade zones.

Three layers:
1. Real news calendar — actual scheduled macro event dates from ForexFactory/historical
2. Volatility-based news detection — catches unscheduled events from price action
3. Session quality model — real-time chop/trend assessment

Output: SessionGrade (A/B/C/D) that controls sizing and trade permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

# Lazy import to avoid circular deps
_news_filter = None


def _get_news_filter():
    global _news_filter
    if _news_filter is None:
        from src.filters.news_calendar import NewsFilter
        _news_filter = NewsFilter()
    return _news_filter


class SessionGrade(IntEnum):
    """Session quality grade controlling trading behavior."""
    A = 4  # Strong trend day — full size, all setups
    B = 3  # Normal day — normal size
    C = 2  # Low quality — reduced size, top EV only
    D = 1  # Chop/news — no trading


@dataclass
class SessionAssessment:
    """Assessment of current session quality."""
    grade: SessionGrade = SessionGrade.B
    size_multiplier: float = 1.0
    reason: str = ""
    is_news_blocked: bool = False
    adx_score: float = 0.0
    trend_score: float = 0.0
    chop_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1: NEWS CALENDAR FILTER
# ═══════════════════════════════════════════════════════════════════════

# Major US macro events with typical release times (ET)
# Format: (month, day, hour, minute, name, severity)
# severity: 3=high (FOMC, CPI, NFP), 2=medium (PPI, retail), 1=low (other)
#
# Since we can't know exact dates without a live calendar feed,
# we use day-of-week + week-of-month heuristics for recurring events:
# - NFP: first Friday of month, 8:30 AM ET
# - CPI: ~10th-13th of month, 8:30 AM ET
# - FOMC: 8 times per year, 2:00 PM ET
# - PPI: ~12th-15th of month, 8:30 AM ET
#
# For backtesting, we detect news indirectly via volatility spikes.

def is_likely_news_period(df: pd.DataFrame, bar_idx: int, lookback: int = 6) -> bool:
    """Detect probable news release from sudden volatility spike.

    Instead of hard-coded calendar (unreliable in backtest),
    detect news-like conditions: sudden ATR expansion + volume spike.
    """
    if bar_idx < lookback + 1:
        return False

    atr = df.get("atr_14", pd.Series(0)).iloc[bar_idx]
    atr_prev = df.get("atr_14", pd.Series(0)).iloc[bar_idx - lookback]
    vol_ratio = df.get("vol_ratio", pd.Series(1)).iloc[bar_idx]

    if pd.isna(atr) or pd.isna(atr_prev) or pd.isna(vol_ratio):
        return False

    # News signature: ATR jumps >50% in 6 bars AND volume spikes >3x
    atr_jump = atr / atr_prev if atr_prev > 0 else 1.0
    if atr_jump > 1.5 and vol_ratio > 3.0:
        return True

    return False


def get_news_buffer_bars(severity: int = 3) -> int:
    """Number of bars to block around detected news (5-min bars)."""
    if severity >= 3:
        return 6  # 30 minutes
    elif severity >= 2:
        return 4  # 20 minutes
    return 2  # 10 minutes


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: SESSION QUALITY MODEL
# ═══════════════════════════════════════════════════════════════════════

def compute_session_quality(df: pd.DataFrame, bar_idx: int, lookback: int = 30) -> SessionAssessment:
    """Assess current session quality using multiple indicators.

    Uses rolling window of last 30 bars (~2.5 hours) to classify conditions.

    Features used:
    - ADX: trend strength (>25 = trending, <18 = choppy)
    - EMA slope: directional persistence
    - VWAP crossing frequency: high crossings = chop
    - Wick-to-body ratio: high wicks = indecision/chop
    - Range expansion: current vs historical
    - Directional persistence: consecutive bars in same direction
    - ATR ratio: current vol vs recent average
    """
    if bar_idx < lookback + 10:
        return SessionAssessment(grade=SessionGrade.B, size_multiplier=1.0, reason="warmup")

    assessment = SessionAssessment()

    # Slice lookback window
    start = max(0, bar_idx - lookback)
    window = df.iloc[start:bar_idx + 1]

    # ── ADX Score (0-1, higher = more trending) ───────────────────────
    # Compute simple ADX proxy from directional movement
    if "atr_14" in df.columns:
        closes = window["close"].values
        highs = window["high"].values
        lows = window["low"].values

        if len(closes) >= 14:
            # Simplified ADX: absolute returns vs range
            abs_returns = np.abs(np.diff(closes))
            ranges = highs[1:] - lows[1:]
            ranges = np.where(ranges > 0, ranges, 1)
            directional_ratio = abs_returns / ranges

            adx_proxy = np.mean(directional_ratio[-14:])  # 0-1 scale
            assessment.adx_score = float(adx_proxy)

    # ── Trend Score (0-1, higher = stronger trend) ────────────────────
    if len(window) >= 20:
        # EMA slope over lookback
        ema_20 = window["close"].ewm(span=20).mean()
        if len(ema_20) >= 5:
            slope = (ema_20.iloc[-1] - ema_20.iloc[-5]) / ema_20.iloc[-5] if ema_20.iloc[-5] != 0 else 0
            assessment.trend_score = min(1.0, abs(slope) * 100)  # Normalize

    # ── Chop Score (0-1, higher = more choppy) ────────────────────────
    chop_signals = 0.0
    chop_count = 0

    # 1. VWAP crossing frequency (high = choppy)
    if "vwap" in df.columns:
        vwap = window.get("vwap", pd.Series())
        if len(vwap) >= 10 and not vwap.isna().all():
            closes_arr = window["close"].values
            vwap_arr = vwap.values
            # Count VWAP crossings in last 20 bars
            cross_window = min(20, len(closes_arr) - 1)
            crossings = 0
            for j in range(1, cross_window + 1):
                idx = len(closes_arr) - j
                if idx > 0:
                    if not np.isnan(vwap_arr[idx]) and not np.isnan(vwap_arr[idx - 1]):
                        above_now = closes_arr[idx] > vwap_arr[idx]
                        above_prev = closes_arr[idx - 1] > vwap_arr[idx - 1]
                        if above_now != above_prev:
                            crossings += 1
            # >4 crossings in 20 bars = very choppy
            chop_signals += min(1.0, crossings / 4.0)
            chop_count += 1

    # 2. Wick-to-body ratio (high = indecision)
    if len(window) >= 10:
        recent = window.iloc[-10:]
        bodies = (recent["close"] - recent["open"]).abs()
        ranges = recent["high"] - recent["low"]
        ranges = ranges.replace(0, np.nan)
        wick_ratio = 1.0 - (bodies / ranges).mean()
        if not np.isnan(wick_ratio):
            chop_signals += min(1.0, wick_ratio / 0.7)  # >70% wick = very choppy
            chop_count += 1

    # 3. Directional persistence (low = choppy)
    if len(window) >= 10:
        recent_returns = window["close"].diff().iloc[-10:]
        signs = np.sign(recent_returns.dropna().values)
        if len(signs) >= 5:
            # Count consecutive same-direction bars
            max_run = 1
            current_run = 1
            for j in range(1, len(signs)):
                if signs[j] == signs[j - 1] and signs[j] != 0:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            # Low max run = choppy (1-2 bar runs), high = trending (5+ bar runs)
            persistence = min(1.0, max_run / 5.0)
            chop_signals += (1.0 - persistence)
            chop_count += 1

    # 4. Range compression (tight range = potential chop or squeeze)
    if "range_expansion" in df.columns:
        re = df["range_expansion"].iloc[bar_idx]
        if not pd.isna(re):
            if re < 0.7:
                chop_signals += 0.8  # Compressed range = chop
            elif re > 2.0:
                chop_signals += 0.0  # Expanding = trending
            else:
                chop_signals += 0.3
            chop_count += 1

    if chop_count > 0:
        assessment.chop_score = chop_signals / chop_count

    # ── Check for scheduled news events (real calendar) ─────────────
    timestamp = df["timestamp"].iloc[bar_idx]
    nf = _get_news_filter()
    news_blocked, news_event = nf.is_blocked(timestamp)
    news_impact = nf.get_impact_at(timestamp)

    # Also check volatility-based detection for unscheduled events
    vol_news = is_likely_news_period(df, bar_idx)
    assessment.is_news_blocked = news_blocked or vol_news

    # ── Assign Grade ──────────────────────────────────────────────────
    if news_blocked and news_impact >= 3:
        assessment.grade = SessionGrade.D
        assessment.size_multiplier = 0.0
        assessment.reason = f"high_impact_news:{news_event}"
    elif news_blocked and news_impact >= 2:
        assessment.grade = SessionGrade.C
        assessment.size_multiplier = 0.5
        assessment.reason = f"medium_impact_news:{news_event}"
    elif assessment.is_news_blocked:
        assessment.grade = SessionGrade.D
        assessment.size_multiplier = 0.0
        assessment.reason = "news_volatility"
    elif assessment.chop_score > 0.70:
        assessment.grade = SessionGrade.D
        assessment.size_multiplier = 0.0
        assessment.reason = "extreme_chop"
    elif assessment.chop_score > 0.55:
        assessment.grade = SessionGrade.C
        assessment.size_multiplier = 0.5
        assessment.reason = "moderate_chop"
    elif assessment.trend_score > 0.5 and assessment.chop_score < 0.35:
        assessment.grade = SessionGrade.A
        assessment.size_multiplier = 1.3
        assessment.reason = "strong_trend"
    elif assessment.adx_score > 0.5 and assessment.chop_score < 0.40:
        assessment.grade = SessionGrade.A
        assessment.size_multiplier = 1.2
        assessment.reason = "directional"
    else:
        assessment.grade = SessionGrade.B
        assessment.size_multiplier = 1.0
        assessment.reason = "normal"

    return assessment


def add_session_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add session quality columns to DataFrame for backtesting."""
    df = df.copy()
    grades = []
    multipliers = []
    news_blocked = []

    for i in range(len(df)):
        if i < 60:
            grades.append(SessionGrade.B)
            multipliers.append(1.0)
            news_blocked.append(False)
        else:
            assessment = compute_session_quality(df, i)
            grades.append(assessment.grade)
            multipliers.append(assessment.size_multiplier)
            news_blocked.append(assessment.is_news_blocked)

    df["session_grade"] = grades
    df["session_size_mult"] = multipliers
    df["news_blocked"] = news_blocked
    return df
