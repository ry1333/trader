"""Opening Drive Pullback Continuation (ODPC) — MNQ-optimized.

Research-backed intraday momentum strategy:
When NQ opens with a strong directional move (≥0.30%), the first shallow
pullback (30-50% retracement, ≤3 bars) is a high-probability continuation entry.

Evidence: Intraday momentum studies show opening-session returns predict
later-session returns, stronger during high volume/volatility conditions.

Time window: 9:35-10:15 ET only (peak: 9:50-10:05 ET)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class ODPCState(IntEnum):
    WAITING_FOR_DRIVE = 0
    DRIVE_DETECTED = 1
    WAITING_FOR_PULLBACK = 2
    PULLBACK_IN_PROGRESS = 3
    ENTRY_TRIGGERED = 4


@dataclass
class DriveInfo:
    """Records the opening drive characteristics."""
    direction: int = 0  # +1 long, -1 short
    open_price: float = 0.0
    drive_extreme: float = 0.0  # High for long drive, low for short
    drive_range: float = 0.0
    drive_bars: int = 0
    avg_drive_volume: float = 0.0
    drive_return_pct: float = 0.0
    valid: bool = False


@dataclass
class PullbackInfo:
    """Records pullback characteristics."""
    retrace_pct: float = 0.0
    duration_bars: int = 0
    avg_volume_ratio: float = 0.0  # vs drive volume
    max_wick_ratio: float = 0.0
    pullback_extreme: float = 0.0  # Low for long PB, high for short
    valid: bool = False


def detect_odpc_signals(
    df: pd.DataFrame,
    min_drive_return: float = 0.0025,  # 0.25% minimum opening drive (NQ moves big)
    min_volume_ratio: float = 1.2,  # Drive volume vs 20-bar avg (NQ opens always have high vol)
    min_atr_ratio: float = 1.2,  # Opening range vs ATR
    pb_retrace_min: float = 0.15,  # Min pullback depth (shallower OK for strong drives)
    pb_retrace_max: float = 0.65,  # Max pullback depth (wider to catch more setups)
    pb_max_bars: int = 5,  # Max pullback duration (25 min on 5-min)
    pb_vol_decline: float = 0.80,  # Pullback vol must be ≤ this × drive vol (loosened)
    pb_max_wick_ratio: float = 2.5,  # Max wick/body ratio (loosened)
    entry_vol_min: float = 0.50,  # Entry bar vol vs drive avg (pullback bars naturally lower)
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Detect ODPC setups. Returns (signals, signal_info, drive_quality).

    signals: 0=flat, 1=long, -1=short
    signal_info: dict with stop/target info per signal bar
    drive_quality: 0-1 score of setup quality
    """
    n = len(df)
    signals = pd.Series(0, index=df.index, dtype=int)
    quality = pd.Series(0.0, index=df.index, dtype=float)

    if n < 50 or "timestamp" not in df.columns:
        return signals, quality, quality

    # Convert to ET for time window checks
    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]

    et_minutes = et.dt.hour * 60 + et.dt.minute
    dates = et.dt.date

    # Pre-compute volume average (20-bar)
    vol_sma_20 = df["volume"].rolling(20).mean()
    atr = df.get("atr_14", pd.Series(0, index=df.index))

    # EMA for trend confirmation
    ema_9 = df["close"].ewm(span=9).mean()

    # Track state per session
    prev_date = None
    state = ODPCState.WAITING_FOR_DRIVE
    drive = DriveInfo()
    pullback = PullbackInfo()
    session_open = 0.0
    drive_start_bar = 0
    pb_start_bar = 0
    pb_volumes = []
    signal_fired_today = False
    rth_open_found = False

    for i in range(1, n):
        cur_date = dates.iloc[i]
        cur_et_min = et_minutes.iloc[i]

        # New calendar date reset
        if cur_date != prev_date:
            signal_fired_today = False
            rth_open_found = False
            state = ODPCState.WAITING_FOR_DRIVE
            drive = DriveInfo()
            pullback = PullbackInfo()
            pb_volumes = []
        prev_date = cur_date

        # Detect RTH open (9:30 ET = 570 minutes) — this is the TRUE session open
        if not rth_open_found and cur_et_min >= 570 and cur_et_min <= 575:
            session_open = df["open"].iloc[i]
            drive_start_bar = i
            rth_open_found = True
            state = ODPCState.WAITING_FOR_DRIVE

        # Only active during 9:35-10:15 ET (575-615 minutes)
        if cur_et_min < 575 or cur_et_min > 615:
            continue

        if not rth_open_found or session_open <= 0:
            continue

        # One signal per session max
        if signal_fired_today:
            continue

        close = df["close"].iloc[i]
        open_p = df["open"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        volume = df["volume"].iloc[i]
        bar_body = abs(close - open_p)
        bar_range = high - low
        prev_close = df["close"].iloc[i - 1]
        cur_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        cur_vol_avg = vol_sma_20.iloc[i] if not pd.isna(vol_sma_20.iloc[i]) else 1

        if cur_atr <= 0 or session_open <= 0:
            continue

        # ── State: WAITING_FOR_DRIVE ──────────────────────────────────
        if state == ODPCState.WAITING_FOR_DRIVE:
            # Check if first 1-3 bars form a qualified drive
            # Drive window: 9:30-9:45 ET (570-585 min)
            if cur_et_min <= 585:
                bars_since_open = i - drive_start_bar + 1

                if bars_since_open >= 1 and bars_since_open <= 3:
                    drive_return = (close - session_open) / session_open

                    # Check drive qualification
                    if abs(drive_return) >= min_drive_return:
                        direction = 1 if drive_return > 0 else -1
                        drive_extreme = high if direction == 1 else low

                        # Opening range check
                        or_high = df["high"].iloc[drive_start_bar:i + 1].max()
                        or_low = df["low"].iloc[drive_start_bar:i + 1].min()
                        or_range = or_high - or_low

                        # Volume check
                        drive_vols = df["volume"].iloc[drive_start_bar:i + 1]
                        avg_drive_vol = drive_vols.mean()
                        vol_ratio = avg_drive_vol / cur_vol_avg if cur_vol_avg > 0 else 0

                        # Directional coherence: net move vs total range
                        net_move = abs(close - session_open)
                        coherence = net_move / or_range if or_range > 0 else 0

                        if (vol_ratio >= min_volume_ratio and
                                or_range >= min_atr_ratio * cur_atr and
                                coherence >= 0.50):
                            drive = DriveInfo(
                                direction=direction,
                                open_price=session_open,
                                drive_extreme=drive_extreme,
                                drive_range=or_range,
                                drive_bars=bars_since_open,
                                avg_drive_volume=avg_drive_vol,
                                drive_return_pct=abs(drive_return),
                                valid=True,
                            )
                            state = ODPCState.WAITING_FOR_PULLBACK
                elif bars_since_open > 3:
                    # Drive window expired without qualification
                    state = ODPCState.WAITING_FOR_DRIVE  # Stay waiting (already there)
                    signal_fired_today = True  # No more attempts today

        # ── State: WAITING_FOR_PULLBACK ───────────────────────────────
        elif state == ODPCState.WAITING_FOR_PULLBACK:
            if not drive.valid:
                state = ODPCState.WAITING_FOR_DRIVE
                continue

            # Detect start of pullback: price moves against drive direction
            if drive.direction == 1:
                # Update drive extreme
                if high > drive.drive_extreme:
                    drive.drive_extreme = high

                # Pullback starts when close is below previous close
                if close < prev_close:
                    state = ODPCState.PULLBACK_IN_PROGRESS
                    pb_start_bar = i
                    pb_volumes = [volume]
                    pullback = PullbackInfo(pullback_extreme=low)
            else:
                if low < drive.drive_extreme:
                    drive.drive_extreme = low

                if close > prev_close:
                    state = ODPCState.PULLBACK_IN_PROGRESS
                    pb_start_bar = i
                    pb_volumes = [volume]
                    pullback = PullbackInfo(pullback_extreme=high)

        # ── State: PULLBACK_IN_PROGRESS ───────────────────────────────
        elif state == ODPCState.PULLBACK_IN_PROGRESS:
            pullback.duration_bars = i - pb_start_bar + 1
            pb_volumes.append(volume)

            # Track pullback extreme
            if drive.direction == 1:
                pullback.pullback_extreme = min(pullback.pullback_extreme, low)
            else:
                pullback.pullback_extreme = max(pullback.pullback_extreme, high)

            # Compute retracement
            drive_move = abs(drive.drive_extreme - drive.open_price)
            if drive.direction == 1:
                retrace = (drive.drive_extreme - pullback.pullback_extreme) / drive_move if drive_move > 0 else 0
            else:
                retrace = (pullback.pullback_extreme - drive.drive_extreme) / drive_move if drive_move > 0 else 0
            pullback.retrace_pct = retrace

            # Volume ratio
            pullback.avg_volume_ratio = np.mean(pb_volumes) / drive.avg_drive_volume if drive.avg_drive_volume > 0 else 1.0

            # Wick ratio check
            wick = bar_range - bar_body
            wick_ratio = wick / bar_body if bar_body > 0 else 10
            pullback.max_wick_ratio = max(pullback.max_wick_ratio, wick_ratio)

            # ── Check pullback failure conditions ─────────────────────
            # Too deep
            if retrace > pb_retrace_max:
                state = ODPCState.WAITING_FOR_DRIVE
                signal_fired_today = True
                continue

            # Too long
            if pullback.duration_bars > pb_max_bars:
                state = ODPCState.WAITING_FOR_DRIVE
                signal_fired_today = True
                continue

            # Lost VWAP (critical)
            vwap = df.get("vwap", pd.Series()).iloc[i] if "vwap" in df.columns else None
            if vwap is not None and not pd.isna(vwap):
                if drive.direction == 1 and close < vwap:
                    state = ODPCState.WAITING_FOR_DRIVE
                    signal_fired_today = True
                    continue
                elif drive.direction == -1 and close > vwap:
                    state = ODPCState.WAITING_FOR_DRIVE
                    signal_fired_today = True
                    continue

            # ── Check entry trigger ───────────────────────────────────
            # Pullback resolves back in drive direction
            pb_valid = (
                pb_retrace_min <= retrace <= pb_retrace_max
                and pullback.avg_volume_ratio <= pb_vol_decline
                and pullback.max_wick_ratio <= pb_max_wick_ratio
            )

            if pb_valid:
                # Entry trigger: close reclaims 9-EMA in drive direction
                ema9 = ema_9.iloc[i]
                entry_vol_ok = volume >= entry_vol_min * drive.avg_drive_volume

                triggered = False
                if drive.direction == 1:
                    if close > ema9 and close > prev_close and entry_vol_ok:
                        triggered = True
                else:
                    if close < ema9 and close < prev_close and entry_vol_ok:
                        triggered = True

                if triggered:
                    # ── SIGNAL FIRE ───────────────────────────────────
                    signals.iloc[i] = drive.direction

                    # Quality score (0-1)
                    q = 0.0
                    q += min(0.3, drive.drive_return_pct * 50)  # Stronger drive = higher
                    q += 0.2 * (1.0 - retrace)  # Shallower pullback = higher
                    q += 0.2 * (1.0 - pullback.avg_volume_ratio)  # Lower PB vol = higher
                    q += 0.15 * min(1.0, (drive.avg_drive_volume / cur_vol_avg) / 3.0)  # Volume
                    q += 0.15 * (1.0 - min(1.0, pullback.max_wick_ratio / 3.0))  # Clean bars
                    quality.iloc[i] = min(1.0, max(0.0, q))

                    signal_fired_today = True
                    state = ODPCState.WAITING_FOR_DRIVE

    return signals, quality, quality  # Third return for compatibility


def get_odpc_stop_target(
    entry_price: float,
    direction: int,
    pullback_extreme: float,
    atr: float,
    tick_size: float = 0.25,
    buffer_ticks: int = 8,  # 2 NQ points
) -> tuple[float, float, float]:
    """Compute stop and targets for ODPC trade.

    Returns (stop_price, target1_price, target2_price).
    """
    # Structural stop: pullback extreme + buffer
    if direction == 1:
        structural_stop = pullback_extreme - buffer_ticks * tick_size
    else:
        structural_stop = pullback_extreme + buffer_ticks * tick_size

    # ATR backstop: at least 1.0 ATR, at most 2.0 ATR
    atr_stop_dist = max(1.0 * atr, abs(entry_price - structural_stop))
    if atr_stop_dist > 2.0 * atr:
        return 0, 0, 0  # Skip: stop too wide

    stop_price = entry_price - atr_stop_dist * direction
    stop_dist = abs(entry_price - stop_price)

    # Targets
    target1 = entry_price + stop_dist * 1.5 * direction  # 1.5R
    target2 = entry_price + stop_dist * 3.0 * direction  # 3.0R (runner)

    return stop_price, target1, target2
