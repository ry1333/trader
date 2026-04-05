"""V3 signal generation — research-backed multi-strategy portfolio.

12 strategies with precise quantitative entry rules:

Original:
1. ORB — Opening Range Breakout
2. VWAP_REVERSION — Fade extended moves back to VWAP
3. TREND_CONTINUATION — Pullback entries in established trends

Research-upgraded:
4. EMA_PULLBACK — 9/21/50 EMA stack + pullback to 21-EMA + bounce confirmation
5. RANGE_BREAKOUT — Donchian channel breakout with volume confirmation
6. MOMENTUM_IGNITION — Volume spike (>2x) + directional price breakout
7. PREV_DAY_LEVEL — Previous RTH high/low breakout with body/wick filter
8. VOL_CONTRACTION — Bollinger squeeze + Keltner channel release
9. RSI_REVERSAL — RSI extreme reversal in trend direction

New from research:
10. FAILED_BREAKOUT — Counter-trend fade of shallow breakout failure
11. VWAP_RECLAIM — Price reclaims VWAP with volume confirmation
12. SESSION_LEVEL — Power hour breakout of session high/low
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
    FAILED_BREAKOUT = 10
    VWAP_RECLAIM = 11
    SESSION_LEVEL = 12


def generate_signals_v3(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate multi-strategy signals. Returns (signals, signal_types)."""
    signals = pd.Series(Signal.FLAT, index=df.index, dtype=int)
    signal_types = pd.Series(SignalType.NONE, index=df.index, dtype=int)

    n = len(df)

    # Pre-compute EMAs
    ema_9 = df["close"].ewm(span=9).mean()
    ema_21 = df["close"].ewm(span=21).mean()
    ema_50 = df["close"].ewm(span=50).mean()

    # Donchian channel (20-bar)
    donchian_high = df["high"].rolling(20).max()
    donchian_low = df["low"].rolling(20).min()

    # Bollinger Bands for vol contraction
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (2 * bb_std) / bb_mid.replace(0, np.nan)
    bb_width_pct = bb_width.rolling(120).rank(pct=True)

    # Keltner Channels for squeeze detection
    kc_mid = ema_21
    kc_atr = df["atr_14"] if "atr_14" in df.columns else pd.Series(0, index=df.index)
    kc_upper = kc_mid + 1.5 * kc_atr
    kc_lower = kc_mid - 1.5 * kc_atr
    # Squeeze: BB inside KC
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

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

    for i in range(n):
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

    # Track recent breakout levels for failed breakout detection
    recent_break_high = pd.Series(np.nan, index=df.index)
    recent_break_low = pd.Series(np.nan, index=df.index)
    recent_break_bar = pd.Series(0, index=df.index)

    for i in range(60, n):
        regime = df["regime"].iloc[i]
        if regime == Regime.STRESS:
            continue

        regime_conf = df["regime_confidence"].iloc[i] if "regime_confidence" in df.columns else 1.0
        if regime_conf < 0.45:
            continue

        if df.get("near_close", pd.Series()).iloc[i] if "near_close" in df.columns else False:
            continue

        close = df["close"].iloc[i]
        open_price = df["open"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        rsi = df.get("rsi_14", pd.Series(50.0)).iloc[i] if "rsi_14" in df.columns else 50.0
        vol_ratio = df.get("vol_ratio", pd.Series(1.0)).iloc[i] if "vol_ratio" in df.columns else 1.0
        atr = df.get("atr_14", pd.Series(0.0)).iloc[i] if "atr_14" in df.columns else 0.0
        macd_hist = df.get("macd_hist", pd.Series(0.0)).iloc[i] if "macd_hist" in df.columns else 0.0
        is_rth = df.get("is_rth", pd.Series(0)).iloc[i] if "is_rth" in df.columns else 0
        is_midday = df.get("is_midday_chop", pd.Series(0)).iloc[i] if "is_midday_chop" in df.columns else 0

        if any(pd.isna(x) for x in [close, rsi, vol_ratio, atr]) or atr <= 0:
            continue

        sig, stype = Signal.FLAT, SignalType.NONE
        e9 = ema_9.iloc[i]
        e21 = ema_21.iloc[i]
        e50 = ema_50.iloc[i]
        bar_range = high - low
        bar_body = abs(close - open_price)

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
            if vwap is not None and not pd.isna(vwap):
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

        # ── 4. EMA Pullback (Research-upgraded) ───────────────────────
        # 9/21/50 EMA stack + pullback to 21-EMA + close back above 9-EMA
        if sig == Signal.FLAT and not pd.isna(e9) and not pd.isna(e21) and not pd.isna(e50):
            # Long: EMA stack bullish + price touched 21-EMA recently + bouncing
            if e9 > e21 > e50:  # Bullish stack
                touched_21 = low <= e21  # Current bar touches 21-EMA
                bouncing = close > e9  # Closes back above 9-EMA
                pullback_depth = (e9 - low) / atr if atr > 0 else 0
                if touched_21 and bouncing and 0.3 < pullback_depth < 2.0 and rsi > 40:
                    sig, stype = Signal.LONG, SignalType.EMA_PULLBACK
            # Short: EMA stack bearish
            elif e9 < e21 < e50:
                touched_21 = high >= e21
                bouncing = close < e9
                pullback_depth = (high - e9) / atr if atr > 0 else 0
                if touched_21 and bouncing and 0.3 < pullback_depth < 2.0 and rsi < 60:
                    sig, stype = Signal.SHORT, SignalType.EMA_PULLBACK

        # ── 5. Range Breakout (Donchian) ──────────────────────────────
        if sig == Signal.FLAT and i > 1 and not pd.isna(donchian_high.iloc[i - 1]):
            dh = donchian_high.iloc[i - 1]
            dl = donchian_low.iloc[i - 1]
            if not pd.isna(dh) and not pd.isna(dl):
                # Body must be >30% of range (not a wick-only break)
                body_pct = bar_body / bar_range if bar_range > 0 else 0
                if close > dh and prev_close <= dh and vol_ratio > 1.2 and body_pct > 0.3:
                    sig, stype = Signal.LONG, SignalType.RANGE_BREAKOUT
                elif close < dl and prev_close >= dl and vol_ratio > 1.2 and body_pct > 0.3:
                    sig, stype = Signal.SHORT, SignalType.RANGE_BREAKOUT

        # ── 6. Momentum Ignition ──────────────────────────────────────
        # Volume spike (>2x avg) + directional bar (body > 50% of ATR)
        if sig == Signal.FLAT and vol_ratio > 2.0:
            bar_move = close - open_price
            if abs(bar_move) > atr * 0.5:
                body_pct = bar_body / bar_range if bar_range > 0 else 0
                if body_pct > 0.5:  # Strong directional bar, not doji
                    if bar_move > 0 and rsi < 75:
                        sig, stype = Signal.LONG, SignalType.MOMENTUM_IGNITION
                    elif bar_move < 0 and rsi > 25:
                        sig, stype = Signal.SHORT, SignalType.MOMENTUM_IGNITION

        # ── 7. Previous Day Level Breakout (Research-upgraded) ────────
        if sig == Signal.FLAT and not pd.isna(prev_day_high.iloc[i]) and is_rth:
            pdh = prev_day_high.iloc[i]
            pdl = prev_day_low.iloc[i]
            if not pd.isna(pdh) and not pd.isna(pdl):
                body_pct = bar_body / bar_range if bar_range > 0 else 0
                # Close above prev day high (not just wick) + body confirmation
                if close > pdh and prev_close <= pdh and vol_ratio > 1.2 and body_pct > 0.3:
                    sig, stype = Signal.LONG, SignalType.PREV_DAY_LEVEL
                elif close < pdl and prev_close >= pdl and vol_ratio > 1.2 and body_pct > 0.3:
                    sig, stype = Signal.SHORT, SignalType.PREV_DAY_LEVEL

        # ── 8. Volatility Contraction Breakout (Research-upgraded) ────
        # Bollinger squeeze (BB inside Keltner) + release + directional
        if sig == Signal.FLAT and i >= 6:
            # Check squeeze was on for at least 6 bars
            if not pd.isna(bb_width_pct.iloc[i]):
                was_squeezed = squeeze_on.iloc[max(0, i - 6):i].all() if i >= 6 else False
                now_released = not squeeze_on.iloc[i] if not pd.isna(squeeze_on.iloc[i]) else False
                if was_squeezed and now_released:
                    # Direction from EMA slope
                    ema_slope = (e9 - ema_9.iloc[i - 3]) / atr if atr > 0 and not pd.isna(ema_9.iloc[i - 3]) else 0
                    if ema_slope > 0.1 and close > bb_upper.iloc[i] and vol_ratio > 0.8:
                        sig, stype = Signal.LONG, SignalType.VOL_CONTRACTION
                    elif ema_slope < -0.1 and close < bb_lower.iloc[i] and vol_ratio > 0.8:
                        sig, stype = Signal.SHORT, SignalType.VOL_CONTRACTION

        # ── 9. RSI Reversal in Trend ──────────────────────────────────
        if sig == Signal.FLAT:
            prev_rsi = df.get("rsi_14", pd.Series(50.0)).iloc[i - 1] if "rsi_14" in df.columns else 50.0
            if not pd.isna(prev_rsi):
                # Long: RSI crosses back above 30 from oversold, in uptrend
                if prev_rsi < 30 and rsi > 30 and close > e50 and close > prev_close:
                    sig, stype = Signal.LONG, SignalType.RSI_REVERSAL
                # Short: RSI crosses back below 70 from overbought, in downtrend
                elif prev_rsi > 70 and rsi < 70 and close < e50 and close < prev_close:
                    sig, stype = Signal.SHORT, SignalType.RSI_REVERSAL

        # ── 10. Failed Breakout Reversal (NEW from research) ──────────
        # Shallow breakout of key level that fails within 3-5 bars → fade
        if sig == Signal.FLAT and is_rth:
            or_high = df.get("or_high", pd.Series()).iloc[i] if "or_high" in df.columns else None
            session_high = df.get("session_high", pd.Series()).iloc[i] if "session_high" in df.columns else None

            # Check for recent failed breakout above session/OR high
            if or_high is not None and not pd.isna(or_high):
                # Look back 3-5 bars for a breakout that failed
                for lb in range(1, min(6, i)):
                    lb_close = df["close"].iloc[i - lb]
                    lb_high = df["high"].iloc[i - lb]
                    if lb_high > or_high and lb_close > or_high:
                        # There was a breakout lb bars ago — did it fail?
                        penetration = (lb_high - or_high) / atr
                        if penetration < 0.5 and close < or_high and close < prev_close:
                            # Shallow breakout that's now back below level
                            body_pct = bar_body / bar_range if bar_range > 0 else 0
                            if body_pct > 0.4:  # Conviction candle
                                sig, stype = Signal.SHORT, SignalType.FAILED_BREAKOUT
                                break

            # Failed breakdown below OR low
            or_low = df.get("or_low", pd.Series()).iloc[i] if "or_low" in df.columns else None
            if sig == Signal.FLAT and or_low is not None and not pd.isna(or_low):
                for lb in range(1, min(6, i)):
                    lb_close = df["close"].iloc[i - lb]
                    lb_low = df["low"].iloc[i - lb]
                    if lb_low < or_low and lb_close < or_low:
                        penetration = (or_low - lb_low) / atr
                        if penetration < 0.5 and close > or_low and close > prev_close:
                            body_pct = bar_body / bar_range if bar_range > 0 else 0
                            if body_pct > 0.4:
                                sig, stype = Signal.LONG, SignalType.FAILED_BREAKOUT
                                break

        # ── 11. VWAP Reclaim/Reject (NEW from research) ───────────────
        if sig == Signal.FLAT and is_rth and not is_midday:
            vwap = df.get("vwap", pd.Series()).iloc[i] if "vwap" in df.columns else None
            if vwap is not None and not pd.isna(vwap) and i >= 30:
                # Check how long price has been on one side of VWAP
                bars_below = 0
                bars_above = 0
                for lb in range(1, min(16, i)):
                    if df["close"].iloc[i - lb] < vwap:
                        bars_below += 1
                    elif df["close"].iloc[i - lb] > vwap:
                        bars_above += 1

                # VWAP Reclaim Long: was below VWAP, now closes above with volume
                if bars_below >= 10 and close > vwap and prev_close < vwap:
                    # Reclaim bar quality: close in upper 60% of range
                    if bar_range > 0 and (close - low) / bar_range >= 0.60 and vol_ratio >= 1.3:
                        sig, stype = Signal.LONG, SignalType.VWAP_RECLAIM

                # VWAP Reject Short: was below, rallies to VWAP, rejected
                elif bars_below >= 8 and high >= vwap and close < vwap:
                    # Reject bar: upper wick >= 60% of range
                    if bar_range > 0 and (high - close) / bar_range >= 0.60 and vol_ratio >= 1.1:
                        sig, stype = Signal.SHORT, SignalType.VWAP_RECLAIM

                # VWAP Reclaim Short (mirror): was above, closes below
                elif bars_above >= 10 and close < vwap and prev_close > vwap:
                    if bar_range > 0 and (high - close) / bar_range >= 0.60 and vol_ratio >= 1.3:
                        sig, stype = Signal.SHORT, SignalType.VWAP_RECLAIM

                # VWAP Reject Long: was above, dips to VWAP, bounces
                elif bars_above >= 8 and low <= vwap and close > vwap:
                    if bar_range > 0 and (close - low) / bar_range >= 0.60 and vol_ratio >= 1.1:
                        sig, stype = Signal.LONG, SignalType.VWAP_RECLAIM

        # ── 12. Session Level Breakout ────────────────────────────────
        if sig == Signal.FLAT and is_rth:
            is_power = df.get("is_power_hour", pd.Series(0)).iloc[i] if "is_power_hour" in df.columns else 0
            session_high = df.get("session_high", pd.Series()).iloc[i] if "session_high" in df.columns else None
            session_low = df.get("session_low", pd.Series()).iloc[i] if "session_low" in df.columns else None
            if is_power and session_high is not None and not pd.isna(session_high):
                body_pct = bar_body / bar_range if bar_range > 0 else 0
                if close > session_high and vol_ratio > 1.0 and body_pct > 0.3:
                    sig, stype = Signal.LONG, SignalType.SESSION_LEVEL
                elif session_low is not None and close < session_low and vol_ratio > 1.0 and body_pct > 0.3:
                    sig, stype = Signal.SHORT, SignalType.SESSION_LEVEL

        if sig == Signal.FLAT:
            continue

        # ── Trend alignment filter ────────────────────────────────────
        # Don't apply to mean-reversion signals (FAILED_BREAKOUT, VWAP_RECLAIM, VWAP_REVERSION)
        mean_reversion_types = {SignalType.FAILED_BREAKOUT, SignalType.VWAP_RECLAIM, SignalType.VWAP_REVERSION}
        if stype not in mean_reversion_types:
            if sig == Signal.LONG and close < e50 * 0.998:
                continue
            if sig == Signal.SHORT and close > e50 * 1.002:
                continue

        signals.iloc[i] = sig
        signal_types.iloc[i] = stype

    return signals, signal_types
