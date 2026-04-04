"""V3 signal generation — multi-strategy portfolio.

10 strategy types that each capture a different edge:

Existing (from v2):
1. ORB — Opening Range Breakout
2. VWAP_REVERSION — Fade extended moves back to VWAP
3. TREND_CONTINUATION — Pullback entries in trends

New:
4. EMA_PULLBACK — Buy pullback to 20-EMA in uptrend (or sell bounce in downtrend)
5. RANGE_BREAKOUT — Donchian channel breakout (20-bar high/low)
6. MOMENTUM_IGNITION — Volume spike + price breakout
7. PREV_DAY_LEVEL — Breakout of previous day high/low
8. VOL_CONTRACTION — Bollinger Band squeeze breakout
9. RSI_REVERSAL — RSI extreme + reversal candle in trend direction
10. SESSION_LEVEL — Breakout of current session high/low after midday
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
    EMA_PULLBACK = 4
    RANGE_BREAKOUT = 5
    MOMENTUM_IGNITION = 6
    PREV_DAY_LEVEL = 7
    VOL_CONTRACTION = 8
    RSI_REVERSAL = 9
    SESSION_LEVEL = 10


def generate_signals_v3(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate multi-strategy signals. Returns (signals, signal_types)."""
    signals = pd.Series(Signal.FLAT, index=df.index, dtype=int)
    signal_types = pd.Series(SignalType.NONE, index=df.index, dtype=int)

    # Pre-compute indicators
    ema_20 = df["close"].ewm(span=20).mean()
    ema_50 = df["close"].ewm(span=50).mean()

    # Donchian channel (20-bar)
    donchian_high = df["high"].rolling(20).max()
    donchian_low = df["low"].rolling(20).min()

    # Bollinger Bands for vol contraction
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_width = (2 * bb_std) / bb_mid.replace(0, np.nan)
    bb_width_pct = bb_width.rolling(50).rank(pct=True)  # percentile rank

    # Previous day high/low
    if df["timestamp"].dt.tz is not None:
        ct = df["timestamp"].dt.tz_convert("US/Central")
    else:
        ct = df["timestamp"]
    date_col = ct.dt.date

    prev_day_high = pd.Series(np.nan, index=df.index)
    prev_day_low = pd.Series(np.nan, index=df.index)
    prev_date = None
    prev_h, prev_l = np.nan, np.nan
    cur_h, cur_l = -np.inf, np.inf

    for i in range(len(df)):
        d = date_col.iloc[i]
        if d != prev_date and prev_date is not None:
            prev_h, prev_l = cur_h, cur_l
            cur_h, cur_l = df["high"].iloc[i], df["low"].iloc[i]
        else:
            cur_h = max(cur_h, df["high"].iloc[i])
            cur_l = min(cur_l, df["low"].iloc[i])
        prev_date = d
        prev_day_high.iloc[i] = prev_h
        prev_day_low.iloc[i] = prev_l

    for i in range(60, len(df)):
        regime = df["regime"].iloc[i]
        if regime == Regime.STRESS:
            continue

        regime_conf = df["regime_confidence"].iloc[i] if "regime_confidence" in df.columns else 1.0
        if regime_conf < 0.45:
            continue

        if df.get("near_close", pd.Series()).iloc[i] if "near_close" in df.columns else False:
            continue

        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        rsi = df.get("rsi_14", pd.Series(50.0)).iloc[i] if "rsi_14" in df.columns else 50.0
        vol_ratio = df.get("vol_ratio", pd.Series(1.0)).iloc[i] if "vol_ratio" in df.columns else 1.0
        atr = df.get("atr_14", pd.Series(0.0)).iloc[i] if "atr_14" in df.columns else 0.0
        macd_hist = df.get("macd_hist", pd.Series(0.0)).iloc[i] if "macd_hist" in df.columns else 0.0
        is_rth = df.get("is_rth", pd.Series(0)).iloc[i] if "is_rth" in df.columns else 0
        is_midday = df.get("is_midday_chop", pd.Series(0)).iloc[i] if "is_midday_chop" in df.columns else 0

        if any(pd.isna(x) for x in [close, rsi, vol_ratio, atr]):
            continue

        sig, stype = Signal.FLAT, SignalType.NONE
        ema20 = ema_20.iloc[i]
        ema50_val = ema_50.iloc[i]

        # ── 1. ORB — Opening Range Breakout ───────────────────────────
        if sig == Signal.FLAT and is_rth:
            or_high = df.get("or_high", pd.Series()).iloc[i] if "or_high" in df.columns else None
            or_low = df.get("or_low", pd.Series()).iloc[i] if "or_low" in df.columns else None
            is_open = df.get("is_open_drive", pd.Series(0)).iloc[i] if "is_open_drive" in df.columns else 0
            if or_high is not None and or_low is not None and not pd.isna(or_high) and not is_open:
                if vol_ratio >= 1.0:
                    if close > or_high and (close - or_high) / or_high < 0.005:
                        sig, stype = Signal.LONG, SignalType.ORB
                    elif close < or_low and (or_low - close) / or_low < 0.005:
                        sig, stype = Signal.SHORT, SignalType.ORB

        # ── 2. VWAP Reversion ─────────────────────────────────────────
        if sig == Signal.FLAT and not is_midday and regime == Regime.MEAN_REVERT:
            vwap = df.get("vwap", pd.Series()).iloc[i] if "vwap" in df.columns else None
            if vwap is not None and not pd.isna(vwap) and atr > 0:
                dist_atr = abs(close - vwap) / atr
                if close < vwap and dist_atr > 1.0 and rsi < 40 and close > prev_close:
                    sig, stype = Signal.LONG, SignalType.VWAP_REVERSION
                elif close > vwap and dist_atr > 1.0 and rsi > 60 and close < prev_close:
                    sig, stype = Signal.SHORT, SignalType.VWAP_REVERSION

        # ── 3. Trend Continuation ─────────────────────────────────────
        if sig == Signal.FLAT and regime == Regime.TREND:
            ret_12 = df.get("ret_12", pd.Series(0.0)).iloc[i] if "ret_12" in df.columns else 0.0
            if not pd.isna(ret_12) and vol_ratio >= 0.7:
                vwap = df.get("vwap", pd.Series()).iloc[i] if "vwap" in df.columns else close
                if pd.isna(vwap):
                    vwap = close
                if close > vwap and 35 <= rsi <= 65 and macd_hist > 0 and ret_12 > 0:
                    sig, stype = Signal.LONG, SignalType.TREND_CONTINUATION
                elif close < vwap and 35 <= rsi <= 65 and macd_hist < 0 and ret_12 < 0:
                    sig, stype = Signal.SHORT, SignalType.TREND_CONTINUATION

        # ── 4. EMA Pullback ───────────────────────────────────────────
        # Price pulls back to 20-EMA in established trend (above 50-EMA)
        if sig == Signal.FLAT and not pd.isna(ema20) and not pd.isna(ema50_val):
            near_ema20 = abs(close - ema20) / atr < 0.5 if atr > 0 else False
            if near_ema20:
                # Long: uptrend (20 > 50 EMA) + pullback to 20 + bouncing
                if ema20 > ema50_val and close > ema20 and prev_close <= ema20 and rsi > 40:
                    sig, stype = Signal.LONG, SignalType.EMA_PULLBACK
                # Short: downtrend + bounce to 20 + rejecting
                elif ema20 < ema50_val and close < ema20 and prev_close >= ema20 and rsi < 60:
                    sig, stype = Signal.SHORT, SignalType.EMA_PULLBACK

        # ── 5. Range Breakout (Donchian) ──────────────────────────────
        if sig == Signal.FLAT and not pd.isna(donchian_high.iloc[i]):
            dh = donchian_high.iloc[i - 1]  # Previous bar's channel
            dl = donchian_low.iloc[i - 1]
            if not pd.isna(dh) and not pd.isna(dl):
                if close > dh and prev_close <= dh and vol_ratio > 1.0:
                    sig, stype = Signal.LONG, SignalType.RANGE_BREAKOUT
                elif close < dl and prev_close >= dl and vol_ratio > 1.0:
                    sig, stype = Signal.SHORT, SignalType.RANGE_BREAKOUT

        # ── 6. Momentum Ignition ──────────────────────────────────────
        # Volume spike (>2x avg) + price breakout in same direction
        if sig == Signal.FLAT and vol_ratio > 2.0 and atr > 0:
            bar_move = (close - df["open"].iloc[i])
            if abs(bar_move) > atr * 0.5:
                if bar_move > 0 and rsi < 75:
                    sig, stype = Signal.LONG, SignalType.MOMENTUM_IGNITION
                elif bar_move < 0 and rsi > 25:
                    sig, stype = Signal.SHORT, SignalType.MOMENTUM_IGNITION

        # ── 7. Previous Day Level Breakout ────────────────────────────
        if sig == Signal.FLAT and not pd.isna(prev_day_high.iloc[i]) and is_rth:
            pdh = prev_day_high.iloc[i]
            pdl = prev_day_low.iloc[i]
            if close > pdh and prev_close <= pdh and vol_ratio > 0.8:
                sig, stype = Signal.LONG, SignalType.PREV_DAY_LEVEL
            elif close < pdl and prev_close >= pdl and vol_ratio > 0.8:
                sig, stype = Signal.SHORT, SignalType.PREV_DAY_LEVEL

        # ── 8. Volatility Contraction Breakout ────────────────────────
        # BB width in lowest 20th percentile + breakout
        if sig == Signal.FLAT and not pd.isna(bb_width_pct.iloc[i]):
            if bb_width_pct.iloc[i] < 0.20:  # Squeeze
                upper = bb_mid.iloc[i] + 2 * bb_std.iloc[i]
                lower = bb_mid.iloc[i] - 2 * bb_std.iloc[i]
                if close > upper and vol_ratio > 0.8:
                    sig, stype = Signal.LONG, SignalType.VOL_CONTRACTION
                elif close < lower and vol_ratio > 0.8:
                    sig, stype = Signal.SHORT, SignalType.VOL_CONTRACTION

        # ── 9. RSI Reversal in Trend ──────────────────────────────────
        # RSI hits extreme then reverses, in direction of higher-timeframe trend
        if sig == Signal.FLAT:
            prev_rsi = df.get("rsi_14", pd.Series(50.0)).iloc[i - 1] if "rsi_14" in df.columns else 50.0
            if not pd.isna(prev_rsi):
                # Long: RSI was <30, now crossing back up, in uptrend
                if prev_rsi < 30 and rsi > 30 and close > ema50_val and close > prev_close:
                    sig, stype = Signal.LONG, SignalType.RSI_REVERSAL
                # Short: RSI was >70, now crossing back down, in downtrend
                elif prev_rsi > 70 and rsi < 70 and close < ema50_val and close < prev_close:
                    sig, stype = Signal.SHORT, SignalType.RSI_REVERSAL

        # ── 10. Session Level Breakout ────────────────────────────────
        # Break of current session high/low after midday (power hour breakout)
        if sig == Signal.FLAT and is_rth:
            is_power = df.get("is_power_hour", pd.Series(0)).iloc[i] if "is_power_hour" in df.columns else 0
            session_high = df.get("session_high", pd.Series()).iloc[i] if "session_high" in df.columns else None
            session_low = df.get("session_low", pd.Series()).iloc[i] if "session_low" in df.columns else None
            if is_power and session_high is not None and not pd.isna(session_high):
                if close > session_high and vol_ratio > 1.0:
                    sig, stype = Signal.LONG, SignalType.SESSION_LEVEL
                elif close < session_low and vol_ratio > 1.0:
                    sig, stype = Signal.SHORT, SignalType.SESSION_LEVEL

        if sig == Signal.FLAT:
            continue

        # ── Trend alignment filter ────────────────────────────────────
        if sig == Signal.LONG and close < ema50_val * 0.998:
            continue
        if sig == Signal.SHORT and close > ema50_val * 1.002:
            continue

        signals.iloc[i] = sig
        signal_types.iloc[i] = stype

    return signals, signal_types
