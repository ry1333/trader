"""BTC Walk-Forward Test: Stats-based exits on BTC-specific strategies.

Walk-forward windows:
- Train: 6 months (build stats bank from trade history)
- Test: 1 month (true OOS with stats-based stops/targets)
- Roll: 1 month forward

BTC-specific parameters:
- MBT: tick_size=$5, tick_value=$0.50, cost=$0.62/side
- No time filter (trades 23/6)
- Dead zone blocking (4-6 PM ET) built into signals
- 5 BTC strategies: Momentum, London Breakout, US ORB, Liquidation, Gap Fill
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from src.config import BacktestConfig, RiskConfig
from src.backtest.engine_btc import run_backtest_btc, compute_btc_strategy_stats
import httpx

# BTC config
bt_cfg = BacktestConfig(
    train_window_days=180, val_window_days=30, test_window_days=30,
    walk_forward_step_days=30,
    cost_per_side_per_contract=0.62,
    slippage_ticks=1,
    tick_size=5.0,    # MBT: $5 per tick
    tick_value=0.50,  # MBT: $0.50 per tick per contract
)

risk_cfg = RiskConfig(
    max_daily_loss=2000, max_total_loss=2000, max_position_size=3,
    risk_per_trade_pct=0.40, flatten_time_ct="23:59",  # BTC: no CT flatten (handled in exit logic)
    session_start_ct="00:00",  # BTC trades all day
    consistency_target=0.50,
    max_risk_per_trade=200.0,
    daily_loss_tier1=500, daily_loss_tier2=1000, weekly_loss_limit=1500,
)

# Load BTC data
df = pd.read_csv("data_cache/btc_5m_2y.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
data_start = df["timestamp"].min()
data_end = df["timestamp"].max()

print(f"BTC data: {len(df)} bars, {data_start.date()} to {data_end.date()}")
print(f"Config: tick={bt_cfg.tick_size} val={bt_cfg.tick_value} cost={bt_cfg.cost_per_side_per_contract}")
print()

results_default = []  # Default stops
results_stats = []    # Stats-optimized stops
all_window_details = []

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

    if len(train_df) < 500 or len(test_df) < 100:
        cursor += pd.Timedelta(days=30)
        continue

    print(f"Window #{wid}: train {cursor.date()}→{train_end.date()} | test {test_start.date()}→{test_end.date()}")

    # ── 1. Train: run backtest to collect trades for stats ──
    r_train, train_trades = run_backtest_btc(
        train_df, risk_cfg, bt_cfg, stats_bank=None,
        training_mode=True)

    if len(train_trades) < 5:
        print(f"  Only {len(train_trades)} training trades — skipping")
        cursor += pd.Timedelta(days=30)
        continue

    # ── 2. Build stats bank from training trades ──
    stats_path = f"data/models/btc_stats_{wid}.pkl"
    bank = compute_btc_strategy_stats(
        train_trades, tick_size=bt_cfg.tick_size, tick_value=bt_cfg.tick_value,
        min_trades=8, output_path=stats_path)

    print(f"  Train: {len(train_trades)}tr, ${r_train.net_pnl:,.0f}, "
          f"stats bank: {len(bank.profiles)} profiles")

    # ── 3. Test with DEFAULT stops (baseline) ──
    r_default, default_trades = run_backtest_btc(
        test_df, risk_cfg, bt_cfg, stats_bank=None)
    results_default.append(r_default.net_pnl)

    # ── 4. Test with STATS-optimized stops ──
    r_stats, stats_trades = run_backtest_btc(
        test_df, risk_cfg, bt_cfg, stats_bank=bank)
    results_stats.append(r_stats.net_pnl)

    # Per-strategy breakdown
    strat_pnl = {}
    for t in stats_trades:
        stype = getattr(t, "_signal_type", 0)
        try:
            from src.strategy.btc_signals import BTCSignalType
            sname = BTCSignalType(int(stype)).name
        except Exception:
            sname = "UNKNOWN"
        net = t.pnl - t.fees
        if sname not in strat_pnl:
            strat_pnl[sname] = {"pnl": 0, "n": 0, "wins": 0}
        strat_pnl[sname]["pnl"] += net
        strat_pnl[sname]["n"] += 1
        if net > 0:
            strat_pnl[sname]["wins"] += 1

    all_window_details.append({
        "wid": wid,
        "test_start": test_start.date(),
        "test_end": test_end.date(),
        "default_pnl": r_default.net_pnl,
        "stats_pnl": r_stats.net_pnl,
        "n_trades": len(stats_trades),
        "strat_pnl": strat_pnl,
    })

    strat_str = " | ".join(f"{k}:{v['pnl']:+.0f}({v['n']})" for k, v in strat_pnl.items())
    print(f"  Default: ${r_default.net_pnl:>7,.0f} ({len(default_trades)}tr)")
    print(f"  Stats:   ${r_stats.net_pnl:>7,.0f} ({len(stats_trades)}tr)")
    print(f"  Strats:  {strat_str}")
    print()

    wid += 1
    cursor += pd.Timedelta(days=30)

# ── Summary ──────────────────────────────────────────────────────
n = len(results_default)
if n == 0:
    print("No windows completed!")
    sys.exit(1)

def_tot = sum(results_default)
stats_tot = sum(results_stats)
def_eq = pd.Series([50000 + sum(results_default[:i+1]) for i in range(n)])
stats_eq = pd.Series([50000 + sum(results_stats[:i+1]) for i in range(n)])
def_dd = (def_eq - def_eq.cummax()).min()
stats_dd = (stats_eq - stats_eq.cummax()).min()

# Strategy-level aggregation
agg_strats = {}
for w in all_window_details:
    for sname, data in w["strat_pnl"].items():
        if sname not in agg_strats:
            agg_strats[sname] = {"pnl": 0, "n": 0, "wins": 0}
        agg_strats[sname]["pnl"] += data["pnl"]
        agg_strats[sname]["n"] += data["n"]
        agg_strats[sname]["wins"] += data["wins"]

strat_table = "\n".join(
    f"  {k}: ${v['pnl']:>+8,.0f} ({v['n']}tr, {v['wins']/v['n']*100:.0f}%WR)"
    for k, v in sorted(agg_strats.items(), key=lambda x: x[1]["pnl"], reverse=True)
)

msg = f"""BTC WALK-FORWARD RESULTS ({n} windows, MBT tick=$5 val=$0.50)

DEFAULT stops:
  Total: ${def_tot:,.0f} | Avg/mo: ${def_tot/n:,.0f}
  Profitable: {sum(1 for r in results_default if r>0)}/{n}
  Max DD: ${def_dd:,.0f}

STATS-OPTIMIZED stops:
  Total: ${stats_tot:,.0f} | Avg/mo: ${stats_tot/n:,.0f}
  Profitable: {sum(1 for r in results_stats if r>0)}/{n}
  Max DD: ${stats_dd:,.0f}

Per-Strategy Breakdown (stats):
{strat_table}

Window Details:
""" + "\n".join(
    f"  #{w['wid']}: {w['test_start']}→{w['test_end']} "
    f"def=${w['default_pnl']:>+7,.0f} stats=${w['stats_pnl']:>+7,.0f} ({w['n_trades']}tr)"
    for w in all_window_details
)

# Combined with MNQ if profitable
combined = f"""
Combined Portfolio (if BTC + MNQ $1,175/mo):
  MNQ + BTC(Default): ${1175 + def_tot/n:,.0f}/mo
  MNQ + BTC(Stats):   ${1175 + stats_tot/n:,.0f}/mo"""

msg += combined
print(msg)

# Send to Discord
url = "https://discord.com/api/webhooks/1490413866168094880/SS77VOuzQzypVeqBpzR414hJpmrBkgMgH4K_SehAo1p1kqN4BkhH9U5lbQmLxyl8sR07"
# Discord embed max is 4096 chars, truncate if needed
desc = msg[:4000]
try:
    httpx.post(url, json={"embeds": [{"title": "BTC Walk-Forward Done", "description": desc, "color": 16750848}]}, timeout=15)
except Exception as e:
    print(f"Discord notification failed: {e}")
