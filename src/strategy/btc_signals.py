"""Bitcoin-specific signal generation — designed for BTC's unique market structure.

BTC is fundamentally different from equity futures:
- Trades 23/6 (CME: Sun 5PM - Fri 4PM ET)
- Trends harder and longer than any traditional market
- Liquidation cascades create momentum that persists
- Mean reversion is DANGEROUS — trends crush faders
- Session structure: Asian → London → US (not RTH/premarket)

5 BTC-specific strategies:
1. BTC_MOMENTUM — trend continuation with EMA stack + ADX filter
2. BTC_LONDON_BREAKOUT — breakout of Asian range during London session
3. BTC_US_ORB — US session opening range breakout
4. BTC_LIQUIDATION — ride volume spike momentum (liquidation cascades)
5. BTC_GAP_FILL — Sunday CME gap fill (77% fill rate)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class BTCSignal(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class BTCSignalType(IntEnum):
    NONE = 0
    BTC_MOMENTUM = 1
    BTC_LONDON_BREAKOUT = 2
    BTC_US_ORB = 3
    BTC_LIQUIDATION = 4
    BTC_GAP_FILL = 5


def compute_btc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BTC-specific features to bar DataFrame."""
    df = df.copy()

    # Session detection based on ET
    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]

    et_hour = et.dt.hour
    et_minute = et.dt.minute
    et_minutes = et_hour * 60 + et_minute

    # BTC Sessions (ET)
    # Asian: 6 PM - 3 AM (1080-1440 + 0-180 minutes)
    # London: 3 AM - 8 AM (180-480 minutes)
    # US: 8 AM - 4 PM (480-960 minutes)
    # Dead zone: 4 PM - 6 PM (960-1080 minutes)
    df["btc_asian"] = ((et_minutes >= 1080) | (et_minutes < 180)).astype(int)
    df["btc_london"] = ((et_minutes >= 180) & (et_minutes < 480)).astype(int)
    df["btc_us"] = ((et_minutes >= 480) & (et_minutes < 960)).astype(int)
    df["btc_dead"] = ((et_minutes >= 960) & (et_minutes < 1080)).astype(int)

    # ADX (critical for BTC — only trade when ADX > 20)
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["btc_adx"] = dx.rolling(14).mean()
    df["btc_plus_di"] = plus_di
    df["btc_minus_di"] = minus_di

    # EMAs (9/20/50 — standard for BTC intraday)
    df["btc_ema9"] = close.ewm(span=9).mean()
    df["btc_ema20"] = close.ewm(span=20).mean()
    df["btc_ema50"] = close.ewm(span=50).mean()

    # ATR for stops
    df["btc_atr14"] = atr14

    # Volume features
    df["btc_vol_sma20"] = df["volume"].rolling(20).mean()
    df["btc_vol_ratio"] = df["volume"] / df["btc_vol_sma20"].replace(0, np.nan)

    # VWAP with 6 PM ET reset
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_x_vol = tp * df["volume"]

    # Detect 6 PM ET session boundary
    dates = et.dt.date
    session_dates = []
    prev_date = None
    session_id = 0
    for i in range(len(df)):
        h = et_hour.iloc[i]
        d = dates.iloc[i]
        if h >= 18 and (prev_date is None or d != prev_date or (i > 0 and et_hour.iloc[i-1] < 18)):
            session_id += 1
        prev_date = d
        session_dates.append(session_id)

    df["_btc_session"] = session_dates

    vwap = []
    cum_tp_vol = 0.0
    cum_vol = 0.0
    cur_session = -1
    for i in range(len(df)):
        s = df["_btc_session"].iloc[i]
        if s != cur_session:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            cur_session = s
        cum_tp_vol += tp_x_vol.iloc[i]
        cum_vol += df["volume"].iloc[i]
        vwap.append(cum_tp_vol / cum_vol if cum_vol > 0 else close.iloc[i])

    df["btc_vwap"] = vwap

    # Asian range (for London breakout)
    # Track session high/low for each session type
    asian_high = {}
    asian_low = {}
    cur_asian_session = -1
    cur_h = -np.inf
    cur_l = np.inf

    for i in range(len(df)):
        s = df["_btc_session"].iloc[i]
        if s != cur_asian_session:
            if cur_asian_session >= 0:
                asian_high[cur_asian_session] = cur_h
                asian_low[cur_asian_session] = cur_l
            cur_h = df["high"].iloc[i]
            cur_l = df["low"].iloc[i]
            cur_asian_session = s
        if df["btc_asian"].iloc[i]:
            cur_h = max(cur_h, df["high"].iloc[i])
            cur_l = min(cur_l, df["low"].iloc[i])

    df["btc_asian_high"] = df["_btc_session"].map(asian_high)
    df["btc_asian_low"] = df["_btc_session"].map(asian_low)
    df["btc_asian_range"] = df["btc_asian_high"] - df["btc_asian_low"]

    # US session opening range (first 12 bars = 1 hour of US session)
    us_or_high = {}
    us_or_low = {}
    us_bar_count = {}

    for i in range(len(df)):
        s = df["_btc_session"].iloc[i]
        if df["btc_us"].iloc[i]:
            if s not in us_bar_count:
                us_bar_count[s] = 0
            us_bar_count[s] += 1
            if us_bar_count[s] <= 12:  # First hour
                if s not in us_or_high:
                    us_or_high[s] = df["high"].iloc[i]
                    us_or_low[s] = df["low"].iloc[i]
                else:
                    us_or_high[s] = max(us_or_high[s], df["high"].iloc[i])
                    us_or_low[s] = min(us_or_low[s], df["low"].iloc[i])

    df["btc_us_or_high"] = df["_btc_session"].map(us_or_high)
    df["btc_us_or_low"] = df["_btc_session"].map(us_or_low)

    # CME gap (Friday close vs current — for Sunday open)
    df["btc_day_of_week"] = et.dt.dayofweek  # 0=Mon, 6=Sun

    # Cleanup
    df = df.drop(columns=["_btc_session"])

    return df


def generate_btc_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate BTC-specific signals. Returns (signals, signal_types)."""
    signals = pd.Series(BTCSignal.FLAT, index=df.index, dtype=int)
    signal_types = pd.Series(BTCSignalType.NONE, index=df.index, dtype=int)

    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]
    et_minutes = et.dt.hour * 60 + et.dt.minute

    for i in range(60, len(df)):
        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        volume = df["volume"].iloc[i]

        adx = df["btc_adx"].iloc[i] if not pd.isna(df["btc_adx"].iloc[i]) else 0
        ema9 = df["btc_ema9"].iloc[i]
        ema20 = df["btc_ema20"].iloc[i]
        ema50 = df["btc_ema50"].iloc[i]
        atr = df["btc_atr14"].iloc[i]
        vol_ratio = df["btc_vol_ratio"].iloc[i]
        vwap = df["btc_vwap"].iloc[i]

        if any(pd.isna(x) for x in [adx, ema9, ema20, ema50, atr, vwap]) or atr <= 0:
            continue

        is_dead = df["btc_dead"].iloc[i]
        is_us = df["btc_us"].iloc[i]
        is_london = df["btc_london"].iloc[i]

        # Skip dead zone (4-6 PM ET)
        if is_dead:
            continue

        sig, stype = BTCSignal.FLAT, BTCSignalType.NONE

        # ── 1. BTC MOMENTUM — trend continuation (primary strategy) ───
        # Only during US + London sessions, ADX > 20
        if sig == BTCSignal.FLAT and (is_us or is_london) and adx > 20:
            # EMA stack bullish: 9 > 20 > 50
            if ema9 > ema20 > ema50:
                # Pullback to 9-EMA or VWAP, then bounce
                touched_ema9 = low <= ema9
                touched_vwap = low <= vwap and close > vwap
                bouncing = close > ema9 and close > prev_close

                if (touched_ema9 or touched_vwap) and bouncing:
                    if vol_ratio >= 1.0:
                        sig, stype = BTCSignal.LONG, BTCSignalType.BTC_MOMENTUM

            # EMA stack bearish: 9 < 20 < 50
            elif ema9 < ema20 < ema50:
                touched_ema9 = high >= ema9
                touched_vwap = high >= vwap and close < vwap
                bouncing = close < ema9 and close < prev_close

                if (touched_ema9 or touched_vwap) and bouncing:
                    if vol_ratio >= 1.0:
                        sig, stype = BTCSignal.SHORT, BTCSignalType.BTC_MOMENTUM

        # ── 2. LONDON BREAKOUT of Asian range ─────────────────────────
        if sig == BTCSignal.FLAT and is_london:
            asian_high = df["btc_asian_high"].iloc[i]
            asian_low = df["btc_asian_low"].iloc[i]
            asian_range = df["btc_asian_range"].iloc[i]

            if not pd.isna(asian_high) and not pd.isna(asian_low) and asian_range > 0:
                # Tight Asian range = better breakout (< 1.5% of price)
                range_pct = asian_range / close
                if range_pct < 0.015:
                    # Breakout filter: close beyond level by 0.15%
                    filter_dist = close * 0.0015

                    if close > asian_high + filter_dist and prev_close <= asian_high:
                        if vol_ratio >= 1.5:
                            sig, stype = BTCSignal.LONG, BTCSignalType.BTC_LONDON_BREAKOUT

                    elif close < asian_low - filter_dist and prev_close >= asian_low:
                        if vol_ratio >= 1.5:
                            sig, stype = BTCSignal.SHORT, BTCSignalType.BTC_LONDON_BREAKOUT

        # ── 3. US SESSION ORB ─────────────────────────────────────────
        if sig == BTCSignal.FLAT and is_us:
            us_or_high = df["btc_us_or_high"].iloc[i]
            us_or_low = df["btc_us_or_low"].iloc[i]

            if not pd.isna(us_or_high) and not pd.isna(us_or_low):
                # Only after opening range is established (>12 bars into US)
                et_min = et_minutes.iloc[i]
                if et_min >= 540:  # After 9 AM ET (1 hour into US session)
                    filter_dist = close * 0.0015

                    # Breakout must align with trend
                    if close > us_or_high + filter_dist and prev_close <= us_or_high:
                        if ema20 > ema50 and vol_ratio >= 1.2:
                            sig, stype = BTCSignal.LONG, BTCSignalType.BTC_US_ORB

                    elif close < us_or_low - filter_dist and prev_close >= us_or_low:
                        if ema20 < ema50 and vol_ratio >= 1.2:
                            sig, stype = BTCSignal.SHORT, BTCSignalType.BTC_US_ORB

        # ── 4. LIQUIDATION CASCADE ────────────────────────────────────
        # Volume spike > 3x + strong directional bar = liquidation in progress
        if sig == BTCSignal.FLAT and vol_ratio > 3.0:
            bar_body = abs(close - df["open"].iloc[i])
            bar_range = high - low
            body_pct = bar_body / bar_range if bar_range > 0 else 0

            # Strong directional bar (body > 60% of range)
            if body_pct > 0.60 and bar_range > atr * 0.5:
                bar_direction = 1 if close > df["open"].iloc[i] else -1

                # DON'T fade — ride the momentum
                if bar_direction == 1 and adx > 15:
                    sig, stype = BTCSignal.LONG, BTCSignalType.BTC_LIQUIDATION
                elif bar_direction == -1 and adx > 15:
                    sig, stype = BTCSignal.SHORT, BTCSignalType.BTC_LIQUIDATION

        # ── 5. CME GAP FILL (Sunday only) ─────────────────────────────
        if sig == BTCSignal.FLAT:
            dow = df["btc_day_of_week"].iloc[i]
            # Sunday evening (day 6) — first bars of new week
            if dow == 6 and i >= 2:
                # Check if there's a gap from Friday close
                # Look back to find Friday's last bar
                for lookback in range(1, min(50, i)):
                    prev_dow = df["btc_day_of_week"].iloc[i - lookback]
                    if prev_dow == 4:  # Friday
                        friday_close = df["close"].iloc[i - lookback]
                        gap_pct = (close - friday_close) / friday_close

                        # Gap > 1% = tradeable gap fill
                        if abs(gap_pct) > 0.01:
                            # Fade the gap (trade toward fill)
                            if gap_pct > 0.01:
                                sig, stype = BTCSignal.SHORT, BTCSignalType.BTC_GAP_FILL
                            elif gap_pct < -0.01:
                                sig, stype = BTCSignal.LONG, BTCSignalType.BTC_GAP_FILL
                        break

        if sig != BTCSignal.FLAT:
            signals.iloc[i] = sig
            signal_types.iloc[i] = stype

    return signals, signal_types
