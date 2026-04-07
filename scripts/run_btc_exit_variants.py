"""Test 4 exit variants on BTC 15-min trend pullback.

A: Current Chandelier, but trail only after 1R
B: Trail only after 1.5R
C: Take 1/3 off at 1R, trail rest with 3 ATR
D: No partial, trail with 3 ATR (wider), only after 1R
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from src.config import BacktestConfig, RiskConfig
from src.strategy.btc_signals import compute_btc_features, generate_btc_signals, BTCSignal, BTCSignalType
from src.backtest.engine import BacktestResult, Trade, _close_trade
from src.risk.engine import RiskEngine


def resample_bars(df, freq):
    df = df.set_index("timestamp")
    r = df.resample(freq, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return r.reset_index()


def run_variant(df, risk_cfg, bt_cfg, variant, training_mode=False):
    """Run backtest with specific exit variant."""
    df = compute_btc_features(df)
    df["signal"], df["signal_type"] = generate_btc_signals(df)

    bar_minutes = max(5, int((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 60)) if len(df) >= 2 else 15
    max_hold = max(4, int(360 / bar_minutes))

    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]
    et_minutes_series = et.dt.hour * 60 + et.dt.minute

    risk = RiskEngine(risk_cfg, 50000.0)
    trades = []
    active = None
    equity = 50000.0
    eq_curve = []
    current_date = ""
    consec_loss = 0
    partial_taken = False  # For variant C

    for i in range(len(df)):
        row = df.iloc[i]
        et_min = et_minutes_series.iloc[i]
        et_time = et.iloc[i]
        date_str = str(et_time.date())
        dow = row.get("btc_day_of_week", 0)

        if date_str != current_date and current_date:
            risk.end_day(current_date)
            if training_mode:
                risk.state.is_killed = False
                risk.state.consecutive_losses = 0
            consec_loss = 0
        current_date = date_str

        # ── EXIT ──
        if active is not None:
            bars_held = i - active.entry_bar
            current_pnl = (row["close"] - active.entry_price) * active.direction
            atr = row.get("btc_atr14", 0)
            if pd.isna(atr) or atr <= 0:
                atr = abs(active.entry_price - active.sl_price) / 2

            time_decay = max_hold // 2
            exit_price, exit_reason = None, ""

            # Flatten
            if 955 <= et_min < 1080:
                exit_price, exit_reason = row["close"], "maintenance_flatten"
            elif dow == 4 and et_min >= 950:
                exit_price, exit_reason = row["close"], "friday_flatten"
            # Max hold
            elif bars_held >= max_hold:
                if current_pnl <= 0:
                    exit_price, exit_reason = row["close"], "max_hold"
                elif active.peak_profit > 0 and current_pnl < active.peak_profit * 0.40:
                    exit_price, exit_reason = row["close"], "max_hold_trail"
            # Time decay
            elif bars_held >= time_decay and current_pnl <= 0:
                exit_price, exit_reason = row["close"], "time_decay"

            # Initial SL
            if exit_price is None:
                if active.direction == 1 and row["low"] <= active.sl_price:
                    exit_price = active.sl_price + bt_cfg.slippage_ticks * bt_cfg.tick_size
                    exit_reason = "stop_loss"
                elif active.direction == -1 and row["high"] >= active.sl_price:
                    exit_price = active.sl_price - bt_cfg.slippage_ticks * bt_cfg.tick_size
                    exit_reason = "stop_loss"

            # Track peak
            if exit_price is None:
                if active.direction == 1:
                    bar_max = row["high"] - active.entry_price
                else:
                    bar_max = active.entry_price - row["low"]
                if bar_max > active.peak_profit:
                    active.peak_profit = bar_max

                stop_dist = abs(active.entry_price - active.sl_price)
                r_multiple = active.peak_profit / stop_dist if stop_dist > 0 else 0
                current_r = current_pnl / stop_dist if stop_dist > 0 else 0

                # ── VARIANT-SPECIFIC EXIT LOGIC ──

                if variant == "A":
                    # Chandelier trail only after 1R
                    if r_multiple >= 1.0:
                        # Breakeven at 1R
                        if current_pnl <= 0:
                            exit_price, exit_reason = row["close"], "breakeven_stop"
                        # Chandelier: 2.5 ATR from highest high
                        elif active.peak_profit >= atr:
                            hh = active.entry_price + active.peak_profit * active.direction
                            chand = hh - atr * 2.5 * active.direction
                            if (active.direction == 1 and row["close"] < chand) or \
                               (active.direction == -1 and row["close"] > chand):
                                exit_price, exit_reason = row["close"], "chandelier_trail"

                elif variant == "B":
                    # Chandelier trail only after 1.5R
                    if r_multiple >= 1.5:
                        if current_pnl <= 0:
                            exit_price, exit_reason = row["close"], "breakeven_stop"
                        elif active.peak_profit >= atr:
                            hh = active.entry_price + active.peak_profit * active.direction
                            chand = hh - atr * 2.5 * active.direction
                            if (active.direction == 1 and row["close"] < chand) or \
                               (active.direction == -1 and row["close"] > chand):
                                exit_price, exit_reason = row["close"], "chandelier_trail"
                    elif r_multiple >= 1.0 and current_pnl <= 0:
                        exit_price, exit_reason = row["close"], "breakeven_stop"

                elif variant == "C":
                    # Partial at 1R, wider trail (3 ATR) on rest
                    if r_multiple >= 1.0 and not partial_taken:
                        # Simulate partial: reduce size by 1/3
                        old_size = active.size
                        partial_size = max(1, old_size // 3)
                        active.size = max(1, old_size - partial_size)
                        partial_taken = True
                        # Book partial PnL (approximate)
                    if r_multiple >= 1.0 and current_pnl <= 0:
                        exit_price, exit_reason = row["close"], "breakeven_stop"
                    elif r_multiple >= 1.0 and active.peak_profit >= atr:
                        hh = active.entry_price + active.peak_profit * active.direction
                        chand = hh - atr * 3.0 * active.direction  # Wider: 3 ATR
                        if (active.direction == 1 and row["close"] < chand) or \
                           (active.direction == -1 and row["close"] > chand):
                            exit_price, exit_reason = row["close"], "chandelier_trail"

                elif variant == "D":
                    # No partial, wider trail (3 ATR), only after 1R
                    if r_multiple >= 1.0:
                        if current_pnl <= 0:
                            exit_price, exit_reason = row["close"], "breakeven_stop"
                        elif active.peak_profit >= atr:
                            hh = active.entry_price + active.peak_profit * active.direction
                            chand = hh - atr * 3.0 * active.direction  # 3 ATR
                            if (active.direction == 1 and row["close"] < chand) or \
                               (active.direction == -1 and row["close"] > chand):
                                exit_price, exit_reason = row["close"], "chandelier_trail"

            if exit_price is not None:
                _close_trade(active, exit_price, i, exit_reason, bt_cfg, risk)
                active._signal_type = BTCSignalType.BTC_TREND_PULLBACK
                trades.append(active)
                equity = risk.state.current_balance
                net = active.pnl - active.fees
                consec_loss = consec_loss + 1 if net <= 0 else 0
                active = None
                partial_taken = False

        # ── ENTRY ──
        if active is None and row["signal"] != BTCSignal.FLAT:
            ct_min = (et_min - 60) % 1440
            if risk.can_trade(ct_min, 8):
                atr = row.get("btc_atr14", 0)
                if pd.isna(atr) or atr <= 0:
                    eq_curve.append(equity)
                    continue
                if not training_mode and consec_loss >= 3:
                    eq_curve.append(equity)
                    continue

                direction = 1 if row["signal"] == BTCSignal.LONG else -1
                entry_price = row["close"] + bt_cfg.slippage_ticks * bt_cfg.tick_size * direction
                sl_ticks = max(4, int(atr * 2.0 / bt_cfg.tick_size))
                tp_ticks = max(4, int(sl_ticks * 4.0))  # Wide target — let trail handle exit
                sl_price = entry_price - sl_ticks * bt_cfg.tick_size * direction
                tp_price = entry_price + tp_ticks * bt_cfg.tick_size * direction

                risk_per = sl_ticks * bt_cfg.tick_value
                size = max(1, min(3, int(200 / risk_per))) if risk_per > 0 else 1

                htf_caution = row.get("htf_caution", 0)
                if htf_caution and size > 1:
                    size = max(1, size // 2)

                active = Trade(entry_bar=i, entry_price=entry_price, direction=direction,
                               size=size, sl_price=sl_price, tp_price=tp_price)
                partial_taken = False

        eq_curve.append(equity)

    if active:
        active._signal_type = BTCSignalType.BTC_TREND_PULLBACK
        _close_trade(active, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active)
    if current_date:
        risk.end_day(current_date)

    return trades


# ── CONFIG ──
risk_cfg = RiskConfig(
    max_daily_loss=2000, max_total_loss=2000, max_position_size=3,
    risk_per_trade_pct=0.40, flatten_time_ct="23:59", session_start_ct="00:00",
    consistency_target=0.50, max_risk_per_trade=200.0,
    daily_loss_tier1=500, daily_loss_tier2=1000, weekly_loss_limit=1500,
)
bt_cfg = BacktestConfig(
    train_window_days=180, val_window_days=30, test_window_days=30,
    walk_forward_step_days=30, cost_per_side_per_contract=0.62,
    slippage_ticks=1, tick_size=5.0, tick_value=0.50,
)

df_5m = pd.read_csv("data_cache/btc_5m_2y.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], utc=True)
df_15m = resample_bars(df_5m, "15min")

variants = {
    "A": "Trail after 1R, Chandelier 2.5 ATR",
    "B": "Trail after 1.5R, Chandelier 2.5 ATR",
    "C": "Partial 1/3 at 1R, trail rest 3 ATR",
    "D": "No partial, trail 3 ATR after 1R",
}

print(f"15-min: {len(df_15m)} bars\n")

for var_id, var_desc in variants.items():
    data_start = df_15m["timestamp"].min()
    data_end = df_15m["timestamp"].max()
    results = []
    cursor = data_start
    total_trades_list = []

    while True:
        train_end = cursor + pd.Timedelta(days=180)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=30)
        if test_end > data_end:
            break

        test_df = df_15m[(df_15m["timestamp"] >= test_start) & (df_15m["timestamp"] < test_end)].reset_index(drop=True)
        if len(test_df) < 50:
            cursor += pd.Timedelta(days=30)
            continue

        trades = run_variant(test_df, risk_cfg, bt_cfg, var_id)
        pnl = sum(t.pnl - t.fees for t in trades)
        results.append(pnl)
        total_trades_list.extend(trades)

        cursor += pd.Timedelta(days=30)

    n = len(results)
    if n == 0:
        print(f"Variant {var_id}: No windows")
        continue

    tot = sum(results)
    eq = pd.Series([50000 + sum(results[:i+1]) for i in range(n)])
    dd = (eq - eq.cummax()).min()
    total_trades = len(total_trades_list)
    wins = sum(1 for t in total_trades_list if t.pnl - t.fees > 0)
    wr = wins / total_trades * 100 if total_trades else 0
    avg_win = np.mean([t.pnl - t.fees for t in total_trades_list if t.pnl - t.fees > 0]) if wins > 0 else 0
    avg_loss = np.mean([abs(t.pnl - t.fees) for t in total_trades_list if t.pnl - t.fees <= 0]) if total_trades - wins > 0 else 1
    payoff = avg_win / avg_loss if avg_loss > 0 else 0

    print(f"Variant {var_id}: {var_desc}")
    print(f"  ${tot:>+7,.0f} total | ${tot/n:>+5,.0f}/mo | {sum(1 for r in results if r>0)}/{n} win | DD ${dd:,.0f}")
    print(f"  {total_trades}tr | {wr:.0f}%WR | avg_win ${avg_win:,.0f} | avg_loss ${avg_loss:,.0f} | payoff {payoff:.2f}")
    print(f"  profit/DD: {abs(tot/dd):.2f}" if dd < 0 else f"  profit/DD: inf")
    print()
