"""BTC Final: Variant D exits + stats-optimized stops/targets.

Walk-forward:
- Train 6 months → collect trades → build stats bank (optimal stop/target per direction)
- Test 1 month → use stats bank stop/target + Variant D Chandelier trail (3 ATR after 1R)
- Compare: default params vs stats-optimized
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from src.config import BacktestConfig, RiskConfig
from src.strategy.btc_signals import compute_btc_features, generate_btc_signals, BTCSignal, BTCSignalType
from src.backtest.engine import Trade, _close_trade
from src.backtest.engine_btc import compute_btc_strategy_stats
from src.risk.engine import RiskEngine


def resample_bars(df, freq):
    df = df.set_index("timestamp")
    r = df.resample(freq, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return r.reset_index()


def run_btc_variant_d(df, risk_cfg, bt_cfg, stats_bank=None, training_mode=False):
    """Variant D: No partial, 3 ATR Chandelier after 1R, with optional stats bank."""
    df = compute_btc_features(df)
    df["signal"], df["signal_type"] = generate_btc_signals(df)

    bar_min = max(5, int((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 60)) if len(df) >= 2 else 15
    max_hold = max(4, int(360 / bar_min))
    time_decay = max_hold // 2

    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]
    et_min_series = et.dt.hour * 60 + et.dt.minute

    risk = RiskEngine(risk_cfg, 50000.0)
    trades = []
    active = None
    equity = 50000.0
    eq_curve = []
    current_date = ""
    consec_loss = 0

    for i in range(len(df)):
        row = df.iloc[i]
        et_min = et_min_series.iloc[i]
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

            exit_price, exit_reason = None, ""
            is_london_bo = getattr(active, "_sig_type", 0) == BTCSignalType.BTC_LONDON_BREAKOUT

            # London breakout hard exit: 8:00 AM ET (480 min)
            if is_london_bo and et_min >= 480:
                exit_price, exit_reason = row["close"], "london_time_exit"

            # Flatten
            if exit_price is None and 955 <= et_min < 1080:
                exit_price, exit_reason = row["close"], "maintenance_flatten"
            elif exit_price is None and dow == 4 and et_min >= 950:
                exit_price, exit_reason = row["close"], "friday_flatten"
            elif exit_price is None and bars_held >= max_hold:
                if current_pnl <= 0:
                    exit_price, exit_reason = row["close"], "max_hold"
                elif active.peak_profit > 0 and current_pnl < active.peak_profit * 0.40:
                    exit_price, exit_reason = row["close"], "max_hold_trail"
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

            # Trail after 1R — strategy-specific Chandelier width
            if exit_price is None:
                if active.direction == 1:
                    bar_max = row["high"] - active.entry_price
                else:
                    bar_max = active.entry_price - row["low"]
                if bar_max > active.peak_profit:
                    active.peak_profit = bar_max

                stop_dist = abs(active.entry_price - active.sl_price)
                r_mult = active.peak_profit / stop_dist if stop_dist > 0 else 0

                # ORB-specific: wider trail (3.5 ATR), time stop if no follow-through
                is_orb = getattr(active, "_sig_type", 0) == BTCSignalType.BTC_US_ORB
                trail_atr = 3.5 if is_orb else 3.0

                # ORB time stop: 6 bars (~90 min on 15m) with no profit = failed breakout
                if is_orb and bars_held >= 6 and current_pnl <= 0:
                    exit_price, exit_reason = row["close"], "orb_time_stop"

                elif r_mult >= 1.0:
                    # Breakeven
                    if current_pnl <= 0:
                        exit_price, exit_reason = row["close"], "breakeven_stop"
                    # Chandelier: trail_atr from highest high
                    elif active.peak_profit >= atr:
                        hh = active.entry_price + active.peak_profit * active.direction
                        chand = hh - atr * trail_atr * active.direction
                        if (active.direction == 1 and row["close"] < chand) or \
                           (active.direction == -1 and row["close"] > chand):
                            exit_price, exit_reason = row["close"], "chandelier_trail"

            if exit_price is not None:
                _close_trade(active, exit_price, i, exit_reason, bt_cfg, risk)
                active._signal_type = getattr(active, "_sig_type", BTCSignalType.BTC_TREND_PULLBACK)
                trades.append(active)
                equity = risk.state.current_balance
                net = active.pnl - active.fees
                consec_loss = consec_loss + 1 if net <= 0 else 0
                active = None

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
                sig_type = int(row["signal_type"])
                strategy_name = BTCSignalType(sig_type).name

                # Stats-optimized or default
                is_london_bo = sig_type == BTCSignalType.BTC_LONDON_BREAKOUT

                if stats_bank:
                    sl_mult, rr = stats_bank.get_exit_params(strategy_name, direction)
                    size_mult = stats_bank.get_size_multiplier(strategy_name, direction)
                    should, _ = stats_bank.should_trade(
                        strategy_name=strategy_name, direction=direction)
                    if not should:
                        eq_curve.append(equity)
                        continue
                else:
                    sl_mult = 2.0
                    rr = 4.0  # Wide — let Chandelier handle exits
                    size_mult = 1.0

                entry_price = row["close"] + bt_cfg.slippage_ticks * bt_cfg.tick_size * direction

                if is_london_bo:
                    # London breakout: stop at Asian range midpoint
                    asia_mid = row.get("btc_asia_mid", None)
                    asia_range = row.get("btc_asia_range", None)
                    if pd.isna(asia_mid) or pd.isna(asia_range) or asia_range <= 0:
                        eq_curve.append(equity)
                        continue
                    sl_price = asia_mid  # Midpoint stop
                    stop_dist = abs(entry_price - sl_price)
                    if stop_dist <= 0 or stop_dist / row["close"] > 0.01:  # Max 1% stop
                        eq_curve.append(equity)
                        continue
                    sl_ticks = max(4, int(stop_dist / bt_cfg.tick_size))
                    # Target: 1.5x range width from entry
                    tp_dist = asia_range * 1.5
                    tp_ticks = max(4, int(tp_dist / bt_cfg.tick_size))
                    tp_price = entry_price + tp_ticks * bt_cfg.tick_size * direction
                else:
                    # Trend pullback / US ORB: ATR-based stops
                    sl_ticks = max(4, int(atr * sl_mult / bt_cfg.tick_size))
                    tp_ticks = max(4, int(sl_ticks * rr))
                    sl_price = entry_price - sl_ticks * bt_cfg.tick_size * direction
                    tp_price = entry_price + tp_ticks * bt_cfg.tick_size * direction

                risk_per = sl_ticks * bt_cfg.tick_value
                size = max(1, min(3, int(200 / risk_per * size_mult))) if risk_per > 0 else 1

                htf_caution = row.get("htf_caution", 0)
                if htf_caution and size > 1:
                    size = max(1, size // 2)

                active = Trade(entry_bar=i, entry_price=entry_price, direction=direction,
                               size=size, sl_price=sl_price, tp_price=tp_price)
                active._sig_type = sig_type

        eq_curve.append(equity)

    if active:
        active._signal_type = getattr(active, "_sig_type", BTCSignalType.BTC_TREND_PULLBACK)
        _close_trade(active, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active)
    if current_date:
        risk.end_day(current_date)

    return trades, equity


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
data_start = df_15m["timestamp"].min()
data_end = df_15m["timestamp"].max()

print(f"BTC 15-min: {len(df_15m)} bars | {data_start.date()} to {data_end.date()}\n")

results_default = []
results_stats = []
all_details = []
cursor = data_start
wid = 0

while True:
    train_end = cursor + pd.Timedelta(days=180)
    test_start = train_end
    test_end = test_start + pd.Timedelta(days=30)
    if test_end > data_end:
        break

    train_df = df_15m[(df_15m["timestamp"] >= cursor) & (df_15m["timestamp"] < train_end)].reset_index(drop=True)
    test_df = df_15m[(df_15m["timestamp"] >= test_start) & (df_15m["timestamp"] < test_end)].reset_index(drop=True)

    if len(train_df) < 200 or len(test_df) < 50:
        cursor += pd.Timedelta(days=30)
        continue

    # ── TRAIN: collect trades, build stats bank ──
    train_trades, _ = run_btc_variant_d(train_df, risk_cfg, bt_cfg, training_mode=True)

    if len(train_trades) < 5:
        cursor += pd.Timedelta(days=30)
        continue

    stats_path = f"data/models/btc_final_stats_{wid}.pkl"
    bank = compute_btc_strategy_stats(
        train_trades, tick_size=bt_cfg.tick_size, tick_value=bt_cfg.tick_value,
        min_trades=5, output_path=stats_path)

    # ── TEST: default params ──
    def_trades, _ = run_btc_variant_d(test_df, risk_cfg, bt_cfg)
    def_pnl = sum(t.pnl - t.fees for t in def_trades)
    results_default.append(def_pnl)

    # ── TEST: stats-optimized ──
    stats_trades, _ = run_btc_variant_d(test_df, risk_cfg, bt_cfg, stats_bank=bank)
    stats_pnl = sum(t.pnl - t.fees for t in stats_trades)
    results_stats.append(stats_pnl)

    # Metrics
    def_wins = sum(1 for t in def_trades if t.pnl - t.fees > 0)
    stats_wins = sum(1 for t in stats_trades if t.pnl - t.fees > 0)
    def_wr = def_wins / len(def_trades) * 100 if def_trades else 0
    stats_wr = stats_wins / len(stats_trades) * 100 if stats_trades else 0

    # Per-strategy breakdown
    strat_pnl = {}
    for t in def_trades:
        stype = getattr(t, "_signal_type", 0)
        try:
            sname = BTCSignalType(int(stype)).name
        except (ValueError, AttributeError):
            sname = "UNKNOWN"
        net = t.pnl - t.fees
        if sname not in strat_pnl:
            strat_pnl[sname] = {"pnl": 0, "n": 0, "wins": 0}
        strat_pnl[sname]["pnl"] += net
        strat_pnl[sname]["n"] += 1
        if net > 0:
            strat_pnl[sname]["wins"] += 1

    all_details.append({
        "wid": wid, "start": test_start.date(), "end": test_end.date(),
        "def_pnl": def_pnl, "stats_pnl": stats_pnl,
        "def_tr": len(def_trades), "stats_tr": len(stats_trades),
        "def_wr": def_wr, "stats_wr": stats_wr,
        "train_tr": len(train_trades), "strat_pnl": strat_pnl,
    })

    print(f"  #{wid}: def=${def_pnl:>+7,.0f}({len(def_trades)}tr,{def_wr:.0f}%) "
          f"stats=${stats_pnl:>+7,.0f}({len(stats_trades)}tr,{stats_wr:.0f}%) "
          f"[train:{len(train_trades)}tr, {len(bank.profiles)} profiles]")

    wid += 1
    cursor += pd.Timedelta(days=30)

n = len(results_default)
if n == 0:
    print("No windows!")
    sys.exit(1)

def _summary(label, results):
    tot = sum(results)
    eq = pd.Series([50000 + sum(results[:i+1]) for i in range(n)])
    dd = (eq - eq.cummax()).min()
    return f"  {label}: ${tot:>+7,.0f} total | ${tot/n:>+5,.0f}/mo | {sum(1 for r in results if r>0)}/{n} profitable | DD ${dd:,.0f} | profit/DD {abs(tot/dd):.1f}" if dd < 0 else f"  {label}: ${tot:>+7,.0f} total | ${tot/n:>+5,.0f}/mo | {sum(1 for r in results if r>0)}/{n} profitable | DD $0"

# Aggregate trade stats
all_def_trades = sum(d["def_tr"] for d in all_details)
all_stats_trades = sum(d["stats_tr"] for d in all_details)
all_def_wins = sum(int(d["def_wr"] * d["def_tr"] / 100) for d in all_details)
all_stats_wins = sum(int(d["stats_wr"] * d["stats_tr"] / 100) for d in all_details)

print(f"""
{'='*65}
BTC FINAL: Variant D + Stats Model ({n} windows, 15-min bars)
{'='*65}

{_summary("DEFAULT (2.0x ATR stop, 4:1 target)", results_default)}
  {all_def_trades}tr total | {all_def_wins}/{all_def_trades} wins ({all_def_wins/all_def_trades*100:.0f}%WR) | {all_def_trades/n:.1f}tr/mo

{_summary("STATS-OPTIMIZED", results_stats)}
  {all_stats_trades}tr total | {all_stats_wins}/{all_stats_trades} wins ({all_stats_wins/all_stats_trades*100:.0f}%WR) | {all_stats_trades/n:.1f}tr/mo

Window Details:""")

for d in all_details:
    flag = " ★" if d["stats_pnl"] > d["def_pnl"] else ""
    print(f"  #{d['wid']}: {d['start']}→{d['end']} "
          f"def=${d['def_pnl']:>+7,.0f} stats=${d['stats_pnl']:>+7,.0f}{flag}")

def_tot = sum(results_default)
stats_tot = sum(results_stats)

# Per-strategy aggregate
agg = {}
for d in all_details:
    for sname, data in d.get("strat_pnl", {}).items():
        if sname not in agg:
            agg[sname] = {"pnl": 0, "n": 0, "wins": 0}
        agg[sname]["pnl"] += data["pnl"]
        agg[sname]["n"] += data["n"]
        agg[sname]["wins"] += data["wins"]

print(f"""
Per-Strategy Breakdown (default):""")
for k, v in sorted(agg.items(), key=lambda x: x[1]["pnl"], reverse=True):
    wr = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
    print(f"  {k}: ${v['pnl']:>+7,.0f} ({v['n']}tr, {wr:.0f}%WR, ${v['pnl']/v['n']:>+.0f}/trade)")

print(f"""
Combined with MNQ ($1,175/mo):
  MNQ + BTC(Default): ${1175 + def_tot/n:,.0f}/mo
  MNQ + BTC(Stats):   ${1175 + stats_tot/n:,.0f}/mo
""")
