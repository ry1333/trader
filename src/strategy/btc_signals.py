"""Bitcoin Multi-Timeframe Trend Pullback v2.

Architecture (from Mesíček-Vojtko framework, adapted):
- 4h: Strategic bias (EMA 21/50 alignment + price confirmation)
- 1h: Tactical confirmation (EMA slope for early reversal detection)
- 15m: Entry timing (pullback to EMA zone + reclaim)

Rules:
- 4h bullish (price > EMA21 > EMA50) → only longs allowed
- 4h bearish (price < EMA21 < EMA50) → only shorts allowed
- 4h neutral (mixed signals) → NO TRADES (reversal protection)
- 1h EMA slope flattening → reduce size (caution mode)
- Vol regime: low vol → skip or half size
- Entry: 15m pullback to EMA(21) + reclaim + above/below VWAP
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
    BTC_TREND_PULLBACK = 1
    BTC_US_ORB = 2
    BTC_LONDON_BREAKOUT = 3


def _compute_htf_layers(df_ltf: pd.DataFrame) -> pd.DataFrame:
    """Compute 4h strategic bias + 1h tactical confirmation, map to LTF bars."""

    df_ltf = df_ltf.sort_values("timestamp").reset_index(drop=True)

    # ── 4h Strategic Layer ───────────────────────────────────────────
    df_4h = df_ltf.set_index("timestamp").resample("4h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()

    if len(df_4h) < 60:
        df_ltf["htf_bias"] = 0
        df_ltf["htf_caution"] = 0
        df_ltf["htf_low_vol"] = 0
        return df_ltf

    c4 = df_4h["close"]
    ema21_4h = c4.ewm(span=21).mean()
    ema50_4h = c4.ewm(span=50).mean()

    # 4h bias: strict — price + EMA alignment must agree
    bias_4h = pd.Series(0, index=df_4h.index)
    bias_4h[(c4 > ema21_4h) & (ema21_4h > ema50_4h)] = 1   # Bullish
    bias_4h[(c4 < ema21_4h) & (ema21_4h < ema50_4h)] = -1  # Bearish
    # Everything else = 0 (neutral → no trades)

    # 4h volatility regime: 20-period stdev of returns vs 60-period MA
    ret_4h = c4.pct_change()
    vol_4h = ret_4h.rolling(20).std()
    vol_ma_4h = vol_4h.rolling(60).mean()
    low_vol = (vol_4h < vol_ma_4h).astype(int)

    df_4h["_bias4h"] = bias_4h.values
    df_4h["_lowvol"] = low_vol.values

    # ── 1h Tactical Layer ────────────────────────────────────────────
    df_1h = df_ltf.set_index("timestamp").resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()

    c1 = df_1h["close"]
    ema20_1h = c1.ewm(span=20).mean()

    # 1h EMA slope: rate of change over 3 bars
    slope_1h = (ema20_1h - ema20_1h.shift(3)) / c1 * 100  # As % of price
    # Caution when slope is flattening (abs < 0.05%)
    caution_1h = (slope_1h.abs() < 0.05).astype(int)

    df_1h["_caution"] = caution_1h.values

    # ── Map back to LTF ──────────────────────────────────────────────
    lookup_4h = df_4h[["timestamp", "_bias4h", "_lowvol"]].rename(columns={"timestamp": "_ts4h"})
    lookup_1h = df_1h[["timestamp", "_caution"]].rename(columns={"timestamp": "_ts1h"})

    merged = pd.merge_asof(df_ltf, lookup_4h, left_on="timestamp", right_on="_ts4h", direction="backward")
    merged = pd.merge_asof(merged, lookup_1h, left_on="timestamp", right_on="_ts1h", direction="backward")

    df_ltf["htf_bias"] = merged["_bias4h"].fillna(0).astype(int).values
    df_ltf["htf_caution"] = merged["_caution"].fillna(0).astype(int).values
    df_ltf["htf_low_vol"] = merged["_lowvol"].fillna(0).astype(int).values

    for col in ["_ts4h", "_ts1h"]:
        if col in df_ltf.columns:
            df_ltf = df_ltf.drop(columns=[col])

    return df_ltf


def compute_btc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BTC features: LTF indicators + HTF layers."""
    df = df.copy()

    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]

    et_hour = et.dt.hour
    et_minutes = et_hour * 60 + et.dt.minute

    df["btc_london"] = ((et_minutes >= 180) & (et_minutes < 480)).astype(int)
    df["btc_us"] = ((et_minutes >= 480) & (et_minutes < 960)).astype(int)
    df["btc_dead"] = ((et_minutes >= 960) & (et_minutes < 1080)).astype(int)

    high, low, close = df["high"], df["low"], df["close"]

    # LTF EMAs
    df["btc_ema9"] = close.ewm(span=9).mean()
    df["btc_ema21"] = close.ewm(span=21).mean()

    # LTF ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["btc_atr14"] = tr.rolling(14).mean()

    # Volume
    df["btc_vol_sma20"] = df["volume"].rolling(20).mean()
    df["btc_vol_ratio"] = df["volume"] / df["btc_vol_sma20"].replace(0, np.nan)

    # VWAP (secondary context, 6 PM ET reset)
    tp = (high + low + close) / 3
    tp_x_vol = tp * df["volume"]

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

    vwap_vals = []
    cum_tp_vol = 0.0
    cum_vol = 0.0
    cur_session = -1
    for i in range(len(df)):
        s = session_dates[i]
        if s != cur_session:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            cur_session = s
        cum_tp_vol += tp_x_vol.iloc[i]
        cum_vol += df["volume"].iloc[i]
        vwap_vals.append(cum_tp_vol / cum_vol if cum_vol > 0 else close.iloc[i])

    df["btc_vwap"] = vwap_vals
    df["btc_day_of_week"] = et.dt.dayofweek

    # US session opening range (first 30 min = 8:00-8:30 AM ET on 15m bars = 2 bars)
    # On 15-min: 480 min = 8:00 AM ET = US open, first 2 bars = 30 min OR
    us_or_high = {}
    us_or_low = {}
    us_bar_count = {}
    _session_ids = session_dates  # reuse from VWAP

    for i in range(len(df)):
        s = _session_ids[i]
        if df["btc_us"].iloc[i]:
            if s not in us_bar_count:
                us_bar_count[s] = 0
            us_bar_count[s] += 1
            if us_bar_count[s] <= 2:  # First 30 min on 15m bars
                if s not in us_or_high:
                    us_or_high[s] = df["high"].iloc[i]
                    us_or_low[s] = df["low"].iloc[i]
                else:
                    us_or_high[s] = max(us_or_high[s], df["high"].iloc[i])
                    us_or_low[s] = min(us_or_low[s], df["low"].iloc[i])

    # Map session ID to each bar for lookup
    df["_session_id"] = _session_ids
    df["btc_us_or_high"] = df["_session_id"].map(us_or_high)
    df["btc_us_or_low"] = df["_session_id"].map(us_or_low)

    # ── Asian range for London breakout (tighter: 8 PM - 2 AM ET) ────
    # 8 PM = 1200 min, 2 AM = 120 min (next day)
    is_trimmed_asia = ((et_minutes >= 1200) | (et_minutes < 120)).astype(int)

    asian_high = {}
    asian_low = {}
    for i in range(len(df)):
        s = _session_ids[i]
        if is_trimmed_asia.iloc[i]:
            if s not in asian_high:
                asian_high[s] = df["high"].iloc[i]
                asian_low[s] = df["low"].iloc[i]
            else:
                asian_high[s] = max(asian_high[s], df["high"].iloc[i])
                asian_low[s] = min(asian_low[s], df["low"].iloc[i])

    df["btc_asia_high"] = df["_session_id"].map(asian_high)
    df["btc_asia_low"] = df["_session_id"].map(asian_low)
    df["btc_asia_range"] = df["btc_asia_high"] - df["btc_asia_low"]
    df["btc_asia_mid"] = (df["btc_asia_high"] + df["btc_asia_low"]) / 2

    # Range compression: percentile of current range vs last 20 sessions
    range_by_session = pd.Series(asian_high) - pd.Series(asian_low)
    range_pctile = {}
    sorted_sessions = sorted(range_by_session.keys())
    for idx, s in enumerate(sorted_sessions):
        if idx < 5:
            range_pctile[s] = 50  # Not enough history
        else:
            lookback = [range_by_session[sorted_sessions[j]]
                       for j in range(max(0, idx - 20), idx)
                       if sorted_sessions[j] in range_by_session]
            if lookback:
                current = range_by_session[s]
                pctile = sum(1 for x in lookback if x <= current) / len(lookback) * 100
                range_pctile[s] = pctile
            else:
                range_pctile[s] = 50

    df["btc_asia_range_pctile"] = df["_session_id"].map(range_pctile)
    df = df.drop(columns=["_session_id"])

    # HTF layers (4h + 1h)
    df = _compute_htf_layers(df)

    return df


def generate_btc_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Generate multi-timeframe trend pullback signals.

    Architecture (unchanged):
    1. 4h bias agrees (bullish or bearish, NOT neutral)
    2. Not in low-vol regime
    3. 15m pullback to EMA(21) + reclaim + VWAP context

    Entry quality improvements (v2):
    A. Reclaim filter — close in top/bottom third of bar (conviction, not wick)
    B. No-chase filter — skip if already >1.2 ATR extended from VWAP
    C. Session filter — only best liquidity windows (London open + US morning)
    """
    signals = pd.Series(BTCSignal.FLAT, index=df.index, dtype=int)
    signal_types = pd.Series(BTCSignalType.NONE, index=df.index, dtype=int)

    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]
    et_minutes = et.dt.hour * 60 + et.dt.minute

    for i in range(60, len(df)):
        close = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        ema21 = df["btc_ema21"].iloc[i]
        atr = df["btc_atr14"].iloc[i]
        vol_ratio = df["btc_vol_ratio"].iloc[i]
        vwap = df["btc_vwap"].iloc[i]
        htf_bias = df["htf_bias"].iloc[i]
        htf_low_vol = df["htf_low_vol"].iloc[i]

        if any(pd.isna(x) for x in [ema21, atr, vol_ratio, vwap]):
            continue
        if atr <= 0:
            continue

        # Skip dead zone
        if df["btc_dead"].iloc[i]:
            continue

        # ── SESSION FILTER: best liquidity windows only ──────────────
        # London open: 3-5 AM ET (180-300 min)
        # US morning: 8:30-11:30 AM ET (510-690 min)
        et_min = et_minutes.iloc[i]
        in_london_open = 180 <= et_min < 300
        in_us_morning = 510 <= et_min < 690
        if not (in_london_open or in_us_morning):
            continue

        # ── 4h GATE: neutral = NO TRADES ─────────────────────────────
        if htf_bias == 0:
            continue

        # ── Low vol regime: skip entirely ─────────────────────────────
        if htf_low_vol:
            continue

        bar_range = high - low
        if bar_range <= 0:
            continue

        # ── LONG: 4h bullish + pullback to EMA21 + quality reclaim ───
        if htf_bias == 1:
            # Pullback: price reached EMA21
            pulled_back = low <= ema21

            # Reclaim: close back above EMA21
            reclaimed = close > ema21

            # (A) Conviction: close in top third of bar range
            bar_position = (close - low) / bar_range
            has_conviction = bar_position >= 0.67

            # VWAP context
            above_vwap = close > vwap

            # (B) No-chase: skip if already too extended from VWAP
            not_chasing = close < vwap + atr * 1.2

            # Volume
            vol_ok = vol_ratio >= 1.0

            if pulled_back and reclaimed and has_conviction and above_vwap and not_chasing and vol_ok:
                signals.iloc[i] = BTCSignal.LONG
                signal_types.iloc[i] = BTCSignalType.BTC_TREND_PULLBACK

        # ── SHORT: 4h bearish + rally to EMA21 + quality rejection ───
        elif htf_bias == -1:
            pulled_back = high >= ema21
            reclaimed = close < ema21

            bar_position = (close - low) / bar_range
            has_conviction = bar_position <= 0.33

            below_vwap = close < vwap
            not_chasing = close > vwap - atr * 1.2

            vol_ok = vol_ratio >= 1.0

            if pulled_back and reclaimed and has_conviction and below_vwap and not_chasing and vol_ok:
                signals.iloc[i] = BTCSignal.SHORT
                signal_types.iloc[i] = BTCSignalType.BTC_TREND_PULLBACK

        # ── US ORB: retest-style breakout of 30-min opening range ────
        # Improvements:
        # 1. Retest entry: require prior bar broke out, this bar retests and holds
        # 2. Breakout distance: close beyond OR by 0.25x OR width (scales with range)
        # 3. Time window: first 90 min after OR (8:30-10:00 AM ET = 510-600 min)
        if signals.iloc[i] == BTCSignal.FLAT and 510 <= et_min < 600:
            us_or_h = df["btc_us_or_high"].iloc[i]
            us_or_l = df["btc_us_or_low"].iloc[i]

            if not pd.isna(us_or_h) and not pd.isna(us_or_l) and i >= 2:
                or_range = us_or_h - us_or_l

                or_range_pct = or_range / close if close > 0 else 0
                if 0.003 < or_range_pct < 0.015 and or_range > 0:
                    # Breakout distance: 0.25x OR width beyond the level
                    breakout_dist = or_range * 0.25

                    prev_close = df["close"].iloc[i - 1]
                    prev_high = df["high"].iloc[i - 1]
                    prev_low = df["low"].iloc[i - 1]

                    # ── BULLISH ORB: retest-and-hold ─────────────────
                    if htf_bias == 1:
                        # Prior bar broke above OR high (initial breakout)
                        prior_broke_out = prev_close > us_or_h
                        # This bar retested (low dipped near OR high) but held
                        retested = low <= us_or_h + breakout_dist
                        # And closed back above with distance
                        held = close > us_or_h + breakout_dist

                        # Also accept first clean breakout if strong
                        first_break = (prev_close <= us_or_h and
                                       close > us_or_h + breakout_dist)

                        if (prior_broke_out and retested and held) or first_break:
                            if vol_ratio >= 1.2:
                                bp = (close - low) / bar_range if bar_range > 0 else 0
                                if bp >= 0.55:
                                    signals.iloc[i] = BTCSignal.LONG
                                    signal_types.iloc[i] = BTCSignalType.BTC_US_ORB

                    # ── BEARISH ORB: retest-and-hold ─────────────────
                    elif htf_bias == -1:
                        prior_broke_out = prev_close < us_or_l
                        retested = high >= us_or_l - breakout_dist
                        held = close < us_or_l - breakout_dist

                        first_break = (prev_close >= us_or_l and
                                       close < us_or_l - breakout_dist)

                        if (prior_broke_out and retested and held) or first_break:
                            if vol_ratio >= 1.2:
                                bp = (close - low) / bar_range if bar_range > 0 else 0
                                if bp <= 0.45:
                                    signals.iloc[i] = BTCSignal.SHORT
                                    signal_types.iloc[i] = BTCSignalType.BTC_US_ORB

        # London Breakout removed — doesn't generate enough trades on 15-min bars.
        # 3 trades in 18 months, all losers. The breakout is too gradual
        # for 15-min candles to capture cleanly.

    return signals, signal_types
