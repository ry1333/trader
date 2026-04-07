"""MGC True Hybrid Test: AI blocking + stats exits."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/root/trader")
import pandas as pd, numpy as np
from src.config import load_settings, BacktestConfig
from src.backtest.multi_instrument import load_instruments
from src.backtest.engine_stats import run_backtest_stats, _check_exit_stats
from src.backtest.engine_v2 import run_backtest_v2
from src.backtest.engine import Trade, _close_trade
from src.ai.quality_model import QualityRiskScorer, train_quality_risk_model
from src.ai.strategy_stats import compute_strategy_stats
from src.ai.features import extract_ai_features
from src.features.engine import compute_features
from src.features.session import add_session_features
from src.strategy.regime import add_regime
from src.strategy.signals_v3 import Signal, SignalType, generate_signals_v3
from src.filters.session_quality import SessionGrade, compute_session_quality
from src.risk.engine import RiskEngine
import httpx

settings = load_settings()
inst = [i for i in load_instruments() if i.symbol == "MGC"][0]
bt_cfg = BacktestConfig(
    train_window_days=180, val_window_days=30, test_window_days=30, walk_forward_step_days=30,
    cost_per_side_per_contract=inst.cost_per_side, slippage_ticks=1,
    tick_size=inst.tick_size, tick_value=inst.tick_value)

df = pd.read_csv("data_cache/gc_5m_2y_databento.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
data_start = df["timestamp"].min()
data_end = df["timestamp"].max()

results_ai = []
results_hybrid = []
cursor = data_start
wid = 0

while True:
    train_end = cursor + pd.Timedelta(days=180)
    val_start = train_end
    val_end = val_start + pd.Timedelta(days=30)
    test_start = val_end
    test_end = test_start + pd.Timedelta(days=30)
    if test_end > data_end:
        break

    train_df = df[(df["timestamp"] >= cursor) & (df["timestamp"] < train_end)].reset_index(drop=True)
    val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)].reset_index(drop=True)
    test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)].reset_index(drop=True)

    if len(train_df) < 200 or len(test_df) < 50:
        cursor += pd.Timedelta(days=30)
        continue

    # Train AI
    r_ai_train, features = run_backtest_v2(
        train_df, settings.strategy, settings.risk, bt_cfg,
        collect_features=True, training_mode=True)

    # Train stats
    r_stats_train, stats_trades = run_backtest_stats(
        train_df, settings.strategy, settings.risk, bt_cfg, training_mode=True)

    if len(r_ai_train.trades) < 20 or len(stats_trades) < 20:
        cursor += pd.Timedelta(days=30)
        continue

    # AI model
    trades_data = pd.DataFrame([{
        "entry_bar": t.entry_bar, "net_pnl": t.pnl - t.fees,
        "pnl": t.pnl, "fees": t.fees,
    } for t in r_ai_train.trades])
    model_path = "data/models/mgc_hyb_ai_" + str(wid) + ".pkl"
    train_quality_risk_model(trades_data, features, model_path)
    scorer = QualityRiskScorer(model_path)

    # Sweep threshold
    best_score = float("-inf")
    best_ev, best_skip = 0, 0.50
    for ev_t in [-10, 0, 10, 20, 30]:
        for skip_t in [0.40, 0.50, 0.60]:
            scorer.ev_threshold = ev_t
            scorer.skip_threshold = skip_t
            vr, _ = run_backtest_v2(val_df, settings.strategy, settings.risk, bt_cfg, scorer=scorer)
            score = vr.net_pnl + (300 if not vr.risk_summary.get("is_killed") else 0)
            if score > best_score:
                best_score = score
                best_ev, best_skip = ev_t, skip_t
    scorer.ev_threshold = best_ev
    scorer.skip_threshold = best_skip

    # AI-only test
    r_ai, _ = run_backtest_v2(test_df, settings.strategy, settings.risk, bt_cfg, scorer=scorer)
    results_ai.append(r_ai.net_pnl)

    # Stats bank
    bank = compute_strategy_stats(stats_trades, inst.tick_size, inst.tick_value,
        output_path="data/models/mgc_hyb_stats_" + str(wid) + ".pkl")

    # HYBRID: AI blocking + stats exits
    test_proc = compute_features(test_df.copy())
    test_proc = add_session_features(test_proc)
    test_proc = add_regime(test_proc)
    test_proc["signal"], test_proc["signal_type"] = generate_signals_v3(test_proc)

    risk = RiskEngine(settings.risk, 50000.0)
    trades = []
    active_trade = None
    active_signal_type = SignalType.NONE
    signal_type_losses = {}
    current_date = ""

    for i in range(len(test_proc)):
        row = test_proc.iloc[i]
        ts = row["timestamp"]
        ct = ts.tz_convert("US/Central") if hasattr(ts, "tz_convert") else ts
        ct_minutes = ct.hour * 60 + ct.minute
        date_str = str(ct.date())
        if date_str != current_date and current_date:
            risk.end_day(current_date)
            signal_type_losses.clear()
        current_date = date_str

        if active_trade is not None:
            exit_price, exit_reason = _check_exit_stats(
                active_trade, row, i, ct_minutes, settings.strategy, bt_cfg, active_signal_type)
            if exit_price is not None:
                _close_trade(active_trade, exit_price, i, exit_reason, bt_cfg, risk)
                active_trade._signal_type = active_signal_type
                trades.append(active_trade)
                net = active_trade.pnl - active_trade.fees
                if net <= 0:
                    signal_type_losses[active_signal_type] = signal_type_losses.get(active_signal_type, 0) + 1
                else:
                    signal_type_losses[active_signal_type] = 0
                active_trade = None
                active_signal_type = SignalType.NONE

        if active_trade is None and row["signal"] != Signal.FLAT:
            if risk.can_trade(ct_minutes, settings.strategy.max_trades_per_day):
                atr = row.get("atr_14", 0)
                atr_50 = row.get("atr_50", 0)
                if i >= 200 and not pd.isna(atr):
                    atr_lb = test_proc["atr_14"].iloc[max(0, i - 200):i].dropna()
                    if len(atr_lb) > 0 and atr < atr_lb.quantile(0.50):
                        continue
                if ct_minutes < 510 or ct_minutes >= 780:
                    continue
                sig_type = int(row["signal_type"])
                if signal_type_losses.get(sig_type, 0) >= 3:
                    continue
                session = compute_session_quality(test_proc, i)
                if session.grade == SessionGrade.D:
                    continue
                vol_gated = not pd.isna(atr_50) and atr_50 > 0 and atr > atr_50 * 2.0
                if not vol_gated and not pd.isna(atr) and atr > 0:
                    direction = 1 if row["signal"] == Signal.LONG else -1
                    strategy_name = SignalType(sig_type).name

                    # AI BLOCKING
                    ai_features = extract_ai_features(test_proc, i)
                    should_take, prob = scorer.should_trade(ai_features)
                    if not should_take or prob < 0.50:
                        continue

                    # STATS EXITS
                    sl_mult, rr_ratio = bank.get_exit_params(strategy_name, direction)
                    size_mult = bank.get_size_multiplier(strategy_name, direction)

                    size = risk.compute_position_size(atr, bt_cfg.tick_size, bt_cfg.tick_value)
                    size = max(1, int(size * size_mult * session.size_multiplier))

                    sl_ticks = risk.compute_stop_ticks(atr, bt_cfg.tick_size, sl_mult)
                    tp_ticks = risk.compute_target_ticks(sl_ticks, rr_ratio)
                    entry_price = row["close"] + (bt_cfg.slippage_ticks * bt_cfg.tick_size * direction)
                    sl_price = entry_price - (sl_ticks * bt_cfg.tick_size * direction)
                    tp_price = entry_price + (tp_ticks * bt_cfg.tick_size * direction)

                    active_trade = Trade(entry_bar=i, entry_price=entry_price, direction=direction,
                        size=size, sl_price=sl_price, tp_price=tp_price)
                    active_signal_type = sig_type

    if active_trade:
        active_trade._signal_type = active_signal_type
        _close_trade(active_trade, test_proc.iloc[-1]["close"], len(test_proc) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active_trade)
    if current_date:
        risk.end_day(current_date)

    hybrid_pnl = sum(t.pnl - t.fees for t in trades)
    results_hybrid.append(hybrid_pnl)

    print(f"#{wid}: ai=${results_ai[-1]:>6,.0f} hybrid=${hybrid_pnl:>6,.0f} ({len(trades)}tr)", flush=True)
    wid += 1
    cursor += pd.Timedelta(days=30)

n = len(results_ai)
ai_tot = sum(results_ai)
hyb_tot = sum(results_hybrid)
ai_eq = pd.Series([50000 + sum(results_ai[:i+1]) for i in range(n)])
hyb_eq = pd.Series([50000 + sum(results_hybrid[:i+1]) for i in range(n)])

msg = f"""MGC TRUE HYBRID ({n} windows)

AI Only: ${ai_tot:,.0f} total, ${ai_tot/n:,.0f}/mo, {sum(1 for r in results_ai if r>0)}/{n} profitable, DD ${(ai_eq-ai_eq.cummax()).min():,.0f}

HYBRID (AI blocking + Stats exits): ${hyb_tot:,.0f} total, ${hyb_tot/n:,.0f}/mo, {sum(1 for r in results_hybrid if r>0)}/{n} profitable, DD ${(hyb_eq-hyb_eq.cummax()).min():,.0f}

With MNQ stats ($1,175/mo):
  MNQ+MGC(AI): ${1175 + ai_tot/n:,.0f}/mo
  MNQ+MGC(Hybrid): ${1175 + hyb_tot/n:,.0f}/mo"""

print(msg)
url = "https://discord.com/api/webhooks/1490413866168094880/SS77VOuzQzypVeqBpzR414hJpmrBkgMgH4K_SehAo1p1kqN4BkhH9U5lbQmLxyl8sR07"
httpx.post(url, json={"embeds": [{"title": "MGC True Hybrid Done", "description": msg, "color": 3447003}]})
