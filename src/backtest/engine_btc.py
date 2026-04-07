"""BTC-specific backtester — adapted for Bitcoin's 24-hour market structure.

Key differences from engine_stats.py (equity futures):
- No 8:30-1:00 CT time restriction — BTC trades 23/6
- Dead zone blocking (4-6 PM ET daily maintenance)
- Friday flatten before 4 PM ET (CME close)
- Higher max hold (BTC trends persist longer: 72 bars = 6 hours)
- BTC-specific signals from btc_signals.py
- Session-aware exits (Asian consolidation vs US momentum)
- No equity session quality filter — replaced with BTC volatility regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.strategy_stats import StrategyStatsBank, compute_strategy_stats
from src.backtest.engine import BacktestResult, Trade, _close_trade
from src.config import BacktestConfig, RiskConfig, StrategyConfig
from src.risk.engine import RiskEngine
from src.strategy.btc_signals import (
    BTCSignal,
    BTCSignalType,
    compute_btc_features,
    generate_btc_signals,
)


# Default R:R per BTC strategy (used when no stats bank yet)
BTC_DEFAULT_RR = {
    BTCSignalType.BTC_MOMENTUM: (2.0, 2.5),       # Wider stop, bigger target (trends run)
    BTCSignalType.BTC_LONDON_BREAKOUT: (1.5, 3.0), # Tight stop, big target (breakouts)
    BTCSignalType.BTC_US_ORB: (1.5, 2.5),          # Similar to London breakout
    BTCSignalType.BTC_LIQUIDATION: (1.0, 3.0),     # Tight stop, ride the cascade
    BTCSignalType.BTC_GAP_FILL: (1.5, 1.5),        # Conservative R:R for gap fills
}

# Max hold bars per strategy (BTC trends last longer)
BTC_MAX_HOLD = {
    BTCSignalType.BTC_MOMENTUM: 72,       # 6 hours — trends persist
    BTCSignalType.BTC_LONDON_BREAKOUT: 48, # 4 hours
    BTCSignalType.BTC_US_ORB: 36,          # 3 hours
    BTCSignalType.BTC_LIQUIDATION: 24,     # 2 hours — fast move
    BTCSignalType.BTC_GAP_FILL: 96,        # 8 hours — gaps can take time to fill
}


def _check_exit_btc(
    trade: Trade,
    row: pd.Series,
    bar_idx: int,
    et_minutes: int,
    bt_cfg: BacktestConfig,
    signal_type: int,
    day_of_week: int,
) -> tuple[float | None, str]:
    """BTC-specific exit logic."""

    bars_held = bar_idx - trade.entry_bar
    current_pnl = (row["close"] - trade.entry_price) * trade.direction
    atr = row.get("btc_atr14", 0)
    if pd.isna(atr):
        atr = 0

    # ── Flatten before CME daily maintenance (3:55 PM ET = 955 min) ──
    if et_minutes >= 955 and et_minutes < 1080:
        return row["close"], "maintenance_flatten"

    # ── Friday flatten before CME close (3:50 PM ET on Friday) ───────
    if day_of_week == 4 and et_minutes >= 950:
        return row["close"], "friday_flatten"

    # ── Strategy-specific max hold ───────────────────────────────────
    max_hold = BTC_MAX_HOLD.get(signal_type, 48)
    if bars_held >= max_hold:
        if current_pnl <= 0:
            return row["close"], "max_hold"
        if trade.peak_profit > 0 and current_pnl < trade.peak_profit * 0.40:
            return row["close"], "max_hold_trail"

    # ── Time decay: 24 bars (2 hours) with no profit → close ────────
    if bars_held >= 24 and current_pnl <= 0:
        return row["close"], "time_decay"

    # ── Standard SL/TP ───────────────────────────────────────────────
    if trade.direction == 1:
        if row["low"] <= trade.sl_price:
            return trade.sl_price + (bt_cfg.slippage_ticks * bt_cfg.tick_size), "stop_loss"
        if row["high"] >= trade.tp_price:
            return trade.tp_price - (bt_cfg.slippage_ticks * bt_cfg.tick_size), "take_profit"
    else:
        if row["high"] >= trade.sl_price:
            return trade.sl_price - (bt_cfg.slippage_ticks * bt_cfg.tick_size), "stop_loss"
        if row["low"] <= trade.tp_price:
            return trade.tp_price + (bt_cfg.slippage_ticks * bt_cfg.tick_size), "take_profit"

    # ── Track peak profit ────────────────────────────────────────────
    if trade.direction == 1:
        bar_max = row["high"] - trade.entry_price
    else:
        bar_max = trade.entry_price - row["low"]
    if bar_max > trade.peak_profit:
        trade.peak_profit = bar_max

    target_distance = abs(trade.tp_price - trade.entry_price)

    # ── Trailing stop (activate at 60% of target, keep 40% of peak) ─
    if target_distance > 0 and trade.peak_profit > target_distance * 0.60:
        trail_keep = trade.peak_profit * 0.40
        if current_pnl < trail_keep:
            return row["close"], "trailing_stop"

    # ── Breakeven stop: 1 ATR profit reached, pull back to entry ────
    if atr > 0 and trade.peak_profit >= atr and current_pnl <= 0:
        return row["close"], "breakeven_stop"

    # ── Liquidation trades: tighter trail (move fast or get out) ─────
    if signal_type == BTCSignalType.BTC_LIQUIDATION:
        if bars_held >= 6 and current_pnl <= 0:
            return row["close"], "liq_time_decay"
        if trade.peak_profit > target_distance * 0.30:
            trail_keep = trade.peak_profit * 0.50
            if current_pnl < trail_keep:
                return row["close"], "liq_trailing"

    # ── Gap fill: exit if gap fills (price returns to Friday close) ──
    if signal_type == BTCSignalType.BTC_GAP_FILL:
        # Gap fill target is tighter — check if price crossed entry toward target
        if bars_held >= 6 and current_pnl > 0:
            # Take quick profits on gap fills
            if current_pnl >= atr * 0.5:
                return row["close"], "gap_fill_target"

    return None, ""


def run_backtest_btc(
    df: pd.DataFrame,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    stats_bank: StrategyStatsBank | None = None,
    starting_balance: float = 50_000.0,
    training_mode: bool = False,
) -> tuple[BacktestResult, list]:
    """Backtest BTC-specific strategies with stats-based exits.

    No time filter (BTC trades 23/6). Dead zone blocking built into signals.
    """

    # Compute BTC features and signals
    df = compute_btc_features(df)
    df["signal"], df["signal_type"] = generate_btc_signals(df)

    # ET time for exit logic
    if df["timestamp"].dt.tz is not None:
        et = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        et = df["timestamp"]
    et_minutes = et.dt.hour * 60 + et.dt.minute

    risk = RiskEngine(risk_cfg, starting_balance)
    trades: list[Trade] = []
    active_trade: Trade | None = None
    active_signal_type: int = BTCSignalType.NONE
    equity = starting_balance
    equity_curve = []
    current_date = ""
    signal_type_losses: dict[int, int] = {}

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        et_time = et.iloc[i] if hasattr(et, "iloc") else et
        et_min = et_minutes.iloc[i]
        date_str = str(et_time.date()) if hasattr(et_time, "date") else str(ts.date())
        dow = row.get("btc_day_of_week", et_time.dayofweek if hasattr(et_time, "dayofweek") else 0)

        if date_str != current_date and current_date:
            risk.end_day(current_date)
            if training_mode:
                risk.state.is_killed = False
                risk.state.consecutive_losses = 0
            signal_type_losses.clear()
        current_date = date_str

        # ── Exit ─────────────────────────────────────────────────────
        if active_trade is not None:
            exit_price, exit_reason = _check_exit_btc(
                active_trade, row, i, et_min, bt_cfg, active_signal_type, dow)
            if exit_price is not None:
                _close_trade(active_trade, exit_price, i, exit_reason, bt_cfg, risk)
                active_trade._signal_type = active_signal_type
                trades.append(active_trade)
                equity = risk.state.current_balance
                net = active_trade.pnl - active_trade.fees
                if net <= 0:
                    signal_type_losses[active_signal_type] = signal_type_losses.get(active_signal_type, 0) + 1
                else:
                    signal_type_losses[active_signal_type] = 0
                active_trade = None
                active_signal_type = BTCSignalType.NONE

        # ── Entry ────────────────────────────────────────────────────
        if active_trade is None and row["signal"] != BTCSignal.FLAT:
            # BTC doesn't use CT time filter — dead zone blocking is in signals
            # RiskEngine still enforces daily loss limits
            # ET→CT: CT = ET - 1h (both are US Eastern / Central)
            ct_min = (et_min - 60) % 1440
            if risk.can_trade(ct_min, 10):  # 10 max trades/day for BTC
                atr = row.get("btc_atr14", 0)
                if pd.isna(atr) or atr <= 0:
                    equity_curve.append(equity)
                    continue

                sig_type = int(row["signal_type"])

                # Circuit breaker: 3 consecutive losses on same signal type
                if not training_mode and signal_type_losses.get(sig_type, 0) >= 3:
                    equity_curve.append(equity)
                    continue

                # Volume filter: skip low-vol bars (ADX < 15)
                adx = row.get("btc_adx", 0)
                if pd.isna(adx) or adx < 12:
                    equity_curve.append(equity)
                    continue

                direction = 1 if row["signal"] == BTCSignal.LONG else -1
                strategy_name = BTCSignalType(sig_type).name

                # Get stop/target from stats bank or defaults
                if stats_bank:
                    should_take, _ = stats_bank.should_trade(
                        strategy_name=strategy_name, direction=direction)
                    if not should_take:
                        equity_curve.append(equity)
                        continue
                    sl_mult, rr_ratio = stats_bank.get_exit_params(strategy_name, direction)
                    size_mult = stats_bank.get_size_multiplier(strategy_name, direction)
                else:
                    sl_mult, rr_ratio = BTC_DEFAULT_RR.get(sig_type, (2.0, 2.0))
                    size_mult = 1.0

                # Position sizing
                size = risk.compute_position_size(atr, bt_cfg.tick_size, bt_cfg.tick_value)
                size = max(1, int(size * size_mult))
                size = min(size, 3)  # Cap at 3 contracts for MBT

                # Compute stops
                sl_ticks = risk.compute_stop_ticks(atr, bt_cfg.tick_size, sl_mult)
                tp_ticks = risk.compute_target_ticks(sl_ticks, rr_ratio)

                entry_price = row["close"] + (bt_cfg.slippage_ticks * bt_cfg.tick_size * direction)
                sl_price = entry_price - (sl_ticks * bt_cfg.tick_size * direction)
                tp_price = entry_price + (tp_ticks * bt_cfg.tick_size * direction)

                active_trade = Trade(
                    entry_bar=i, entry_price=entry_price, direction=direction,
                    size=size, sl_price=sl_price, tp_price=tp_price)
                active_signal_type = sig_type

        equity_curve.append(equity)

    # Close any remaining trade
    if active_trade:
        active_trade._signal_type = active_signal_type
        _close_trade(active_trade, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active_trade)
    if current_date:
        risk.end_day(current_date)

    eq_series = pd.Series(equity_curve, index=df["timestamp"].values[:len(equity_curve)])
    daily_pnl = pd.Series(
        [d.pnl for d in risk.state.daily_history],
        index=[d.date for d in risk.state.daily_history])

    result = BacktestResult(
        trades=trades, equity_curve=eq_series, daily_pnl=daily_pnl,
        risk_summary=risk.summary, df=df)

    return result, trades


def compute_btc_strategy_stats(
    trades: list,
    tick_size: float = 5.0,
    tick_value: float = 0.50,
    min_trades: int = 10,
    output_path: str = "data/models/btc_strategy_stats.pkl",
) -> StrategyStatsBank:
    """Compute strategy stats for BTC trades.

    Lower min_trades (10 vs 15) because BTC has fewer trades per window.
    Uses BTCSignalType names instead of SignalType.
    """
    bank = StrategyStatsBank()

    records = []
    for t in trades:
        net = t.pnl - t.fees
        bars = t.exit_bar - t.entry_bar if t.exit_bar else 0
        stop_dist = abs(t.entry_price - t.sl_price)
        if stop_dist <= 0:
            continue

        atr_approx = stop_dist / 2.0
        mae_atr = stop_dist / atr_approx if atr_approx > 0 else 2.0
        mfe_atr = t.peak_profit / atr_approx if atr_approx > 0 and t.peak_profit > 0 else 0
        actual_move = (t.exit_price - t.entry_price) * t.direction
        r_mult = actual_move / stop_dist

        sig_type = getattr(t, "_signal_type", 0)
        try:
            sig_name = BTCSignalType(int(sig_type)).name
        except (ValueError, AttributeError):
            sig_name = "UNKNOWN"

        records.append({
            "net": net, "direction": t.direction, "bars": bars,
            "mae_atr": mae_atr, "mfe_atr": mfe_atr, "r_mult": r_mult,
            "exit_reason": t.exit_reason, "size": t.size,
            "atr": atr_approx, "strategy": sig_name,
            "is_winner": net > 0,
        })

    if not records:
        bank.save(output_path)
        return bank

    df = pd.DataFrame(records)

    for strategy in df["strategy"].unique():
        if strategy == "NONE":
            continue
        for direction, side_label in [(1, "LONG"), (-1, "SHORT")]:
            mask = (df["strategy"] == strategy) & (df["direction"] == direction)
            grp = df[mask]
            n = len(grp)

            if n < min_trades:
                continue

            winners = grp[grp["is_winner"]]
            losers = grp[~grp["is_winner"]]
            win_rate = len(winners) / n
            avg_win = winners["net"].mean() if len(winners) > 0 else 0
            avg_loss = abs(losers["net"].mean()) if len(losers) > 0 else 1
            payoff = avg_win / avg_loss if avg_loss > 0 else 0
            ev = win_rate * avg_win - (1 - win_rate) * avg_loss

            maes = grp["mae_atr"].values
            mfes = grp[grp["mfe_atr"] > 0]["mfe_atr"].values
            winner_maes = winners["mae_atr"].values if len(winners) > 0 else []

            # Optimal stop via grid search
            best_stop, best_stop_pnl, best_stop_surv = 2.0, float("-inf"), 0
            for test_stop in np.arange(0.75, 4.0, 0.25):
                sim_pnl = 0
                for _, r in grp.iterrows():
                    if r["mfe_atr"] >= 0 and r["mae_atr"] <= test_stop:
                        sim_pnl += r["net"]
                    else:
                        sim_pnl -= test_stop * r["atr"] * r["size"] * tick_value / tick_size
                if sim_pnl > best_stop_pnl:
                    best_stop_pnl = sim_pnl
                    best_stop = test_stop
                    if len(winner_maes) > 0:
                        best_stop_surv = np.mean(winner_maes <= test_stop)

            # Optimal target via grid search
            best_target, best_target_pnl, best_target_hit = 3.0, float("-inf"), 0
            for test_target in np.arange(0.75, 6.0, 0.25):
                sim_pnl = 0
                hits = 0
                for _, r in grp.iterrows():
                    if r["mfe_atr"] >= test_target:
                        sim_pnl += test_target * r["atr"] * r["size"] * tick_value / tick_size
                        hits += 1
                    else:
                        sim_pnl += r["net"]
                if sim_pnl > best_target_pnl:
                    best_target_pnl = sim_pnl
                    best_target = test_target
                    best_target_hit = hits / n

            optimal_rr = best_target / best_stop if best_stop > 0 else 1.5

            if payoff > 0 and win_rate > 0:
                kelly = (win_rate * payoff - (1 - win_rate)) / payoff
                kelly = max(0, kelly)
            else:
                kelly = 0
            quarter_kelly = kelly / 4

            from src.ai.strategy_stats import ExitProfile
            key = f"{strategy}_{side_label}_ALL"
            bank.profiles[key] = ExitProfile(
                strategy_name=strategy, direction=side_label, regime="ALL",
                n_trades=n, win_rate=round(win_rate, 3),
                avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2),
                payoff_ratio=round(payoff, 3),
                optimal_stop_atr=round(best_stop, 2),
                stop_survivors=round(best_stop_surv, 3),
                optimal_target_atr=round(best_target, 2),
                target_hit_rate=round(best_target_hit, 3),
                optimal_rr=round(optimal_rr, 2),
                quarter_kelly=round(quarter_kelly, 4),
                ev_per_trade=round(ev, 2),
                mae_50pct=round(np.percentile(maes, 50), 2) if len(maes) > 0 else 0,
                mae_75pct=round(np.percentile(maes, 75), 2) if len(maes) > 0 else 0,
                mfe_50pct=round(np.percentile(mfes, 50), 2) if len(mfes) > 0 else 0,
                mfe_75pct=round(np.percentile(mfes, 75), 2) if len(mfes) > 0 else 0,
            )
            logger.info(
                f"  {key}: {n}tr WR={win_rate:.0%} EV=${ev:.0f} "
                f"stop={best_stop:.1f} target={best_target:.1f} "
                f"R:R={optimal_rr:.1f} qKelly={quarter_kelly:.3f}"
            )

    bank.save(output_path)
    return bank
