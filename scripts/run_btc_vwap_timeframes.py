"""Test BTC Multi-TF Trend Pullback on 15-min bars (4h bias + 1h tactical + 15m entry)."""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from src.config import BacktestConfig, RiskConfig
from src.backtest.engine_btc import run_backtest_btc, compute_btc_strategy_stats


def resample_bars(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.set_index("timestamp")
    resampled = df.resample(freq, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index()


risk_cfg = RiskConfig(
    max_daily_loss=2000, max_total_loss=2000, max_position_size=3,
    risk_per_trade_pct=0.40, flatten_time_ct="23:59",
    session_start_ct="00:00",
    consistency_target=0.50,
    max_risk_per_trade=200.0,
    daily_loss_tier1=500, daily_loss_tier2=1000, weekly_loss_limit=1500,
)

bt_cfg = BacktestConfig(
    train_window_days=180, val_window_days=30, test_window_days=30,
    walk_forward_step_days=30,
    cost_per_side_per_contract=0.62, slippage_ticks=1,
    tick_size=5.0, tick_value=0.50,
)

df_5m = pd.read_csv("data_cache/btc_5m_2y.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], utc=True)
df_15m = resample_bars(df_5m, "15min")

print(f"15-min: {len(df_15m)} bars (from {len(df_5m)} 5-min bars)\n")

for label, df in [("15-MIN", df_15m)]:
    print(f"{'='*60}")
    print(f"  {label} BTC TREND PULLBACK (4h+1h+15m, Chandelier trail)")
    print(f"{'='*60}")

    data_start = df["timestamp"].min()
    data_end = df["timestamp"].max()
    results = []
    all_details = []
    cursor = data_start
    wid = 0

    while True:
        train_end = cursor + pd.Timedelta(days=180)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=30)
        if test_end > data_end:
            break

        train_df = df[(df["timestamp"] >= cursor) & (df["timestamp"] < train_end)].reset_index(drop=True)
        test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)].reset_index(drop=True)

        if len(train_df) < 200 or len(test_df) < 50:
            cursor += pd.Timedelta(days=30)
            continue

        # Train (just to collect stats, but we use default VWAP-based stops)
        r_train, train_trades = run_backtest_btc(
            train_df, risk_cfg, bt_cfg, stats_bank=None, training_mode=True)

        # Test with pure research parameters (no stats optimization)
        r_test, test_trades = run_backtest_btc(
            test_df, risk_cfg, bt_cfg, stats_bank=None)

        results.append(r_test.net_pnl)

        wins = sum(1 for t in test_trades if t.pnl - t.fees > 0)
        wr = wins / len(test_trades) * 100 if test_trades else 0

        # Exit reason breakdown
        reasons = {}
        for t in test_trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        all_details.append({
            "wid": wid, "test_start": test_start.date(), "test_end": test_end.date(),
            "pnl": r_test.net_pnl, "trades": len(test_trades),
            "wins": wins, "wr": wr, "reasons": reasons,
            "train_trades": len(train_trades),
        })

        reason_str = " ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:3])
        print(f"  #{wid}: ${r_test.net_pnl:>+7,.0f} | {len(test_trades)}tr {wr:.0f}%WR | train:{len(train_trades)}tr | {reason_str}")

        wid += 1
        cursor += pd.Timedelta(days=30)

    n = len(results)
    if n == 0:
        print(f"  No windows!\n")
        continue

    tot = sum(results)
    eq = pd.Series([50000 + sum(results[:i+1]) for i in range(n)])
    dd = (eq - eq.cummax()).min()

    total_trades = sum(d["trades"] for d in all_details)
    total_wins = sum(d["wins"] for d in all_details)
    agg_wr = total_wins / total_trades * 100 if total_trades else 0

    # Exit reason totals
    all_reasons = {}
    for d in all_details:
        for k, v in d["reasons"].items():
            all_reasons[k] = all_reasons.get(k, 0) + v

    print(f"""
  RESULTS ({n} windows):
  Total: ${tot:,.0f} | Avg/mo: ${tot/n:,.0f}
  Profitable windows: {sum(1 for r in results if r>0)}/{n}
  Max DD: ${dd:,.0f}
  Win Rate: {agg_wr:.1f}% ({total_wins}/{total_trades})
  Avg trades/mo: {total_trades/n:.1f}

  Exit reasons: {' | '.join(f'{k}:{v}' for k,v in sorted(all_reasons.items(), key=lambda x:-x[1]))}

  Combined with MNQ ($1,175/mo): ${1175 + tot/n:,.0f}/mo
""")
