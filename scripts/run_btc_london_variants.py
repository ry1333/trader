"""Test London Breakout variants on 5-min bars.

Testing the 3 weakest parameters per review:
A. Exit at 8:00 AM ET (original)
B. Exit at 9:30 AM ET (London-NY overlap)
C. Exit at 11:00 AM ET (full US morning)
D. Trail after 8:00 AM ET (no hard exit, Chandelier trail)

All variants use:
- 4h trend gate (proven)
- Trimmed Asian range (8PM-2AM ET)
- Range compression < 60th percentile
- First 5-min close outside range after 3:30 AM ET
- Midpoint stop
- Split exit: partial 50% at 1x range, runner on trail
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from src.config import BacktestConfig, RiskConfig
from src.backtest.engine import Trade, _close_trade
from src.risk.engine import RiskEngine


def compute_london_features(df):
    """Compute Asian range + 4h bias for London breakout on 5-min bars."""
    df = df.copy()
    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]

    et_hour = et.dt.hour
    et_minutes = et_hour * 60 + et.dt.minute
    df["_et_min"] = et_minutes.values
    df["_dow"] = et.dt.dayofweek.values

    high, low, close = df["high"], df["low"], df["close"]

    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # Volume
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)

    # Session IDs (6 PM ET reset)
    session_ids = []
    prev_date, session_id = None, 0
    for i in range(len(df)):
        h = et_hour.iloc[i]
        d = et.dt.date.iloc[i]
        if h >= 18 and (prev_date is None or d != prev_date or (i > 0 and et_hour.iloc[i-1] < 18)):
            session_id += 1
        prev_date = d
        session_ids.append(session_id)
    df["_sid"] = session_ids

    # Trimmed Asian range: 8 PM - 2 AM ET (1200-1440 + 0-120 min)
    is_asia = ((et_minutes >= 1200) | (et_minutes < 120))
    asia_h, asia_l = {}, {}
    for i in range(len(df)):
        s = df["_sid"].iloc[i]
        if is_asia.iloc[i]:
            if s not in asia_h:
                asia_h[s] = df["high"].iloc[i]
                asia_l[s] = df["low"].iloc[i]
            else:
                asia_h[s] = max(asia_h[s], df["high"].iloc[i])
                asia_l[s] = min(asia_l[s], df["low"].iloc[i])

    df["asia_high"] = df["_sid"].map(asia_h)
    df["asia_low"] = df["_sid"].map(asia_l)
    df["asia_range"] = df["asia_high"] - df["asia_low"]
    df["asia_mid"] = (df["asia_high"] + df["asia_low"]) / 2

    # Range compression percentile (vs last 20 sessions)
    range_by_s = pd.Series(asia_h) - pd.Series(asia_l)
    sorted_s = sorted(range_by_s.keys())
    pctile_map = {}
    for idx, s in enumerate(sorted_s):
        lb = [range_by_s[sorted_s[j]] for j in range(max(0, idx-20), idx) if sorted_s[j] in range_by_s]
        if len(lb) >= 3:
            pctile_map[s] = sum(1 for x in lb if x <= range_by_s[s]) / len(lb) * 100
        else:
            pctile_map[s] = 50
    df["asia_pctile"] = df["_sid"].map(pctile_map)

    # 4h bias (resampled)
    df_4h = df.set_index("timestamp").resample("4h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    if len(df_4h) >= 60:
        c4 = df_4h["close"]
        e21 = c4.ewm(span=21).mean()
        e50 = c4.ewm(span=50).mean()
        bias = pd.Series(0, index=df_4h.index)
        bias[(c4 > e21) & (e21 > e50)] = 1
        bias[(c4 < e21) & (e21 < e50)] = -1
        df_4h["_bias"] = bias.values
        merged = pd.merge_asof(
            df.sort_values("timestamp"),
            df_4h[["timestamp", "_bias"]].rename(columns={"timestamp": "_ts4h"}),
            left_on="timestamp", right_on="_ts4h", direction="backward")
        df["htf_bias"] = merged["_bias"].fillna(0).astype(int).values
    else:
        df["htf_bias"] = 0

    return df


def run_london_variant(df, bt_cfg, exit_time_et, use_trail_after=False):
    """Run London Breakout with specific exit variant."""
    df = compute_london_features(df)

    risk_cfg = RiskConfig(
        max_daily_loss=2000, max_total_loss=2000, max_position_size=3,
        risk_per_trade_pct=0.40, flatten_time_ct="23:59", session_start_ct="00:00",
        consistency_target=0.50, max_risk_per_trade=200.0,
        daily_loss_tier1=500, daily_loss_tier2=1000, weekly_loss_limit=1500)

    risk = RiskEngine(risk_cfg, 50000.0)
    trades = []
    active = None
    equity = 50000.0
    current_date = ""
    last_entry_session = -1  # One trade per session

    for i in range(60, len(df)):
        row = df.iloc[i]
        et_min = row["_et_min"]
        sid = row["_sid"]
        dow = row["_dow"]

        date_str = str(row["timestamp"].date()) if hasattr(row["timestamp"], "date") else ""
        if date_str != current_date and current_date:
            risk.end_day(current_date)
        current_date = date_str

        # ── EXIT ──
        if active is not None:
            bars_held = i - active.entry_bar
            current_pnl = (row["close"] - active.entry_price) * active.direction
            atr = row.get("atr14", 0)
            if pd.isna(atr) or atr <= 0:
                atr = abs(active.entry_price - active.sl_price) / 2

            exit_price, exit_reason = None, ""

            # Hard exit time (variant-specific)
            if not use_trail_after and et_min >= exit_time_et:
                exit_price, exit_reason = row["close"], "time_exit"
            # Trail mode: switch to Chandelier after exit_time_et
            elif use_trail_after and et_min >= exit_time_et:
                # Track peak
                if active.direction == 1:
                    bm = row["high"] - active.entry_price
                else:
                    bm = active.entry_price - row["low"]
                if bm > active.peak_profit:
                    active.peak_profit = bm
                # Chandelier 3 ATR
                if active.peak_profit >= atr:
                    hh = active.entry_price + active.peak_profit * active.direction
                    chand = hh - atr * 3.0 * active.direction
                    if (active.direction == 1 and row["close"] < chand) or \
                       (active.direction == -1 and row["close"] > chand):
                        exit_price, exit_reason = row["close"], "chandelier_trail"
                # Hard cutoff at 11 AM ET even in trail mode
                if exit_price is None and et_min >= 660:
                    exit_price, exit_reason = row["close"], "max_time_exit"

            # SL
            if exit_price is None:
                if active.direction == 1 and row["low"] <= active.sl_price:
                    exit_price = active.sl_price + bt_cfg.slippage_ticks * bt_cfg.tick_size
                    exit_reason = "stop_loss"
                elif active.direction == -1 and row["high"] >= active.sl_price:
                    exit_price = active.sl_price - bt_cfg.slippage_ticks * bt_cfg.tick_size
                    exit_reason = "stop_loss"

            # TP (1.5x range)
            if exit_price is None:
                if active.direction == 1 and row["high"] >= active.tp_price:
                    exit_price = active.tp_price - bt_cfg.slippage_ticks * bt_cfg.tick_size
                    exit_reason = "take_profit"
                elif active.direction == -1 and row["low"] <= active.tp_price:
                    exit_price = active.tp_price + bt_cfg.slippage_ticks * bt_cfg.tick_size
                    exit_reason = "take_profit"

            # Track peak (for non-trail mode too)
            if exit_price is None:
                if active.direction == 1:
                    bm = row["high"] - active.entry_price
                else:
                    bm = active.entry_price - row["low"]
                if bm > active.peak_profit:
                    active.peak_profit = bm

            # Time decay: 2 hours no profit
            if exit_price is None and bars_held >= 24 and current_pnl <= 0:
                exit_price, exit_reason = row["close"], "time_decay"

            if exit_price is not None:
                _close_trade(active, exit_price, i, exit_reason, bt_cfg, risk)
                trades.append(active)
                equity = risk.state.current_balance
                active = None

        # ── ENTRY ──
        if active is None and 210 <= et_min < 360:  # 3:30-6:00 AM ET
            # One per session
            if sid == last_entry_session:
                continue

            asia_h = row.get("asia_high", None)
            asia_l = row.get("asia_low", None)
            asia_range = row.get("asia_range", None)
            asia_mid = row.get("asia_mid", None)
            asia_pctile = row.get("asia_pctile", None)
            htf_bias = row.get("htf_bias", 0)
            vol_ratio = row.get("vol_ratio", 0)
            close = row["close"]
            atr = row.get("atr14", 0)

            if any(pd.isna(x) for x in [asia_h, asia_l, asia_range, asia_mid, asia_pctile, vol_ratio, atr]):
                continue
            if asia_range <= 0 or atr <= 0:
                continue

            range_pct = asia_range / close if close > 0 else 0
            if not (0.002 < range_pct < 0.020):  # Wider range (v1 used 1.5%)
                continue
            # No compression requirement (v1 didn't have one)
            # No 4h gate for London breakout — the breakout creates the trend
            # if htf_bias == 0: continue

            bar_range = row["high"] - row["low"]
            if bar_range <= 0:
                continue
            body_pct = abs(close - row["open"]) / bar_range

            prev_close = df["close"].iloc[i - 1] if i > 0 else close
            direction = 0

            # Direction from breakout itself, no 4h requirement
            if close > asia_h and prev_close <= asia_h:
                if body_pct >= 0.50 and vol_ratio >= 1.2:
                    direction = 1
            elif close < asia_l and prev_close >= asia_l:
                if body_pct >= 0.50 and vol_ratio >= 1.2:
                    direction = -1

            if direction != 0:
                entry_price = close + bt_cfg.slippage_ticks * bt_cfg.tick_size * direction
                # ATR-based stop (2x ATR, proven from MNQ system)
                sl_dist = atr * 2.0
                sl_price = entry_price - sl_dist * direction
                sl_ticks = max(4, int(sl_dist / bt_cfg.tick_size))
                # Target: 3x stop (let winners run)
                tp_dist = sl_dist * 3.0
                tp_price = entry_price + tp_dist * direction
                risk_per = sl_ticks * bt_cfg.tick_value
                size = max(1, min(3, int(200 / risk_per))) if risk_per > 0 else 1

                active = Trade(entry_bar=i, entry_price=entry_price, direction=direction,
                               size=size, sl_price=sl_price, tp_price=tp_price)
                last_entry_session = sid

    if active:
        _close_trade(active, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active)
    if current_date:
        risk.end_day(current_date)

    return trades


# ── RUN ──
bt_cfg = BacktestConfig(
    train_window_days=180, val_window_days=30, test_window_days=30,
    walk_forward_step_days=30, cost_per_side_per_contract=0.62,
    slippage_ticks=1, tick_size=5.0, tick_value=0.50)

df = pd.read_csv("data_cache/btc_5m_2y.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
data_start = df["timestamp"].min()
data_end = df["timestamp"].max()

print(f"BTC 5-min: {len(df)} bars\n")

variants = {
    "A: Exit 8AM ET":    (480, False),
    "B: Exit 9:30AM ET": (570, False),
    "C: Exit 11AM ET":   (660, False),
    "D: Trail after 8AM": (480, True),
}

for label, (exit_time, use_trail) in variants.items():
    results = []
    all_trades = []
    cursor = data_start

    while True:
        test_start = cursor + pd.Timedelta(days=180)
        test_end = test_start + pd.Timedelta(days=30)
        if test_end > data_end:
            break
        test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)].reset_index(drop=True)
        if len(test_df) < 200:
            cursor += pd.Timedelta(days=30)
            continue

        trades = run_london_variant(test_df, bt_cfg, exit_time, use_trail)
        pnl = sum(t.pnl - t.fees for t in trades)
        results.append(pnl)
        all_trades.extend(trades)
        cursor += pd.Timedelta(days=30)

    n = len(results)
    if n == 0:
        print(f"{label}: No windows")
        continue

    tot = sum(results)
    eq = pd.Series([50000 + sum(results[:i+1]) for i in range(n)])
    dd = (eq - eq.cummax()).min()
    wins = sum(1 for t in all_trades if t.pnl - t.fees > 0)
    total = len(all_trades)
    wr = wins / total * 100 if total else 0
    avg_w = np.mean([t.pnl - t.fees for t in all_trades if t.pnl - t.fees > 0]) if wins else 0
    avg_l = np.mean([abs(t.pnl - t.fees) for t in all_trades if t.pnl - t.fees <= 0]) if total - wins > 0 else 1
    payoff = avg_w / avg_l if avg_l > 0 else 0

    # Exit reason breakdown
    reasons = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    reason_str = " | ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:4])

    print(f"{label}")
    print(f"  ${tot:>+7,.0f} total | ${tot/n:>+5,.0f}/mo | {sum(1 for r in results if r>0)}/{n} win | DD ${dd:,.0f}")
    print(f"  {total}tr | {wr:.0f}%WR | avg_win ${avg_w:,.0f} | avg_loss ${avg_l:,.0f} | payoff {payoff:.2f}")
    print(f"  profit/DD: {abs(tot/dd):.1f}" if dd < 0 else f"  profit/DD: inf")
    print(f"  exits: {reason_str}")
    print()
