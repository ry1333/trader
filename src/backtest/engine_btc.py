"""BTC Multi-Timeframe Trend Pullback backtester v2.

Exits:
- Initial stop: 2.0x ATR(14) (standard for BTC intraday)
- Chandelier trail: 2.5x ATR(14) from highest high since entry (wider than initial)
- Breakeven: move to entry when 1.5x ATR profit reached
- No hard profit target — let Chandelier trail capture fat tails
- Time decay: ~2 hours no profit → close
- Max hold: ~6 hours

Sizing:
- Normal: ATR-based risk budget
- 1h caution mode: 50% size
- Low vol regime: skip (handled in signals)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.strategy_stats import StrategyStatsBank
from src.backtest.engine import BacktestResult, Trade, _close_trade
from src.config import BacktestConfig, RiskConfig
from src.risk.engine import RiskEngine
from src.strategy.btc_signals import (
    BTCSignal,
    BTCSignalType,
    compute_btc_features,
    generate_btc_signals,
)


def _check_exit_btc(
    trade: Trade,
    row: pd.Series,
    bar_idx: int,
    et_minutes: int,
    bt_cfg: BacktestConfig,
    day_of_week: int,
    max_hold: int,
) -> tuple[float | None, str]:
    """Chandelier-style exit logic."""

    bars_held = bar_idx - trade.entry_bar
    current_pnl = (row["close"] - trade.entry_price) * trade.direction
    atr = row.get("btc_atr14", 0)
    if pd.isna(atr) or atr <= 0:
        atr = abs(trade.entry_price - trade.sl_price) / 2  # fallback

    time_decay = max_hold // 2

    # CME maintenance flatten
    if 955 <= et_minutes < 1080:
        return row["close"], "maintenance_flatten"

    # Friday flatten
    if day_of_week == 4 and et_minutes >= 950:
        return row["close"], "friday_flatten"

    # Max hold
    if bars_held >= max_hold:
        if current_pnl <= 0:
            return row["close"], "max_hold"
        if trade.peak_profit > 0 and current_pnl < trade.peak_profit * 0.40:
            return row["close"], "max_hold_trail"

    # Time decay
    if bars_held >= time_decay and current_pnl <= 0:
        return row["close"], "time_decay"

    # Initial SL (fixed from entry)
    if trade.direction == 1:
        if row["low"] <= trade.sl_price:
            return trade.sl_price + (bt_cfg.slippage_ticks * bt_cfg.tick_size), "stop_loss"
    else:
        if row["high"] >= trade.sl_price:
            return trade.sl_price - (bt_cfg.slippage_ticks * bt_cfg.tick_size), "stop_loss"

    # Track highest high / lowest low since entry
    if trade.direction == 1:
        bar_max = row["high"] - trade.entry_price
    else:
        bar_max = trade.entry_price - row["low"]
    if bar_max > trade.peak_profit:
        trade.peak_profit = bar_max

    # Breakeven: once 1.5x ATR profit reached, exit if back to entry
    if trade.peak_profit >= atr * 1.5 and current_pnl <= 0:
        return row["close"], "breakeven_stop"

    # Chandelier trailing stop: 2.5x ATR from highest high / lowest low
    # Only activate once we have meaningful profit (> 1 ATR)
    if trade.peak_profit >= atr:
        chandelier_dist = atr * 2.5
        if trade.direction == 1:
            highest_high = trade.entry_price + trade.peak_profit
            chandelier_stop = highest_high - chandelier_dist
            if row["close"] < chandelier_stop:
                return row["close"], "chandelier_trail"
        else:
            lowest_low = trade.entry_price - trade.peak_profit
            chandelier_stop = lowest_low + chandelier_dist
            if row["close"] > chandelier_stop:
                return row["close"], "chandelier_trail"

    return None, ""


def run_backtest_btc(
    df: pd.DataFrame,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    stats_bank: StrategyStatsBank | None = None,
    starting_balance: float = 50_000.0,
    training_mode: bool = False,
) -> tuple[BacktestResult, list]:
    """Backtest BTC trend pullback with Chandelier exits."""

    df = compute_btc_features(df)
    df["signal"], df["signal_type"] = generate_btc_signals(df)

    # Timeframe-aware holds (~6 hours for 15m bars)
    if len(df) >= 2:
        bar_minutes = max(5, int((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 60))
    else:
        bar_minutes = 15
    max_hold_bars = max(4, int(360 / bar_minutes))

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
    consecutive_losses = 0

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        et_time = et.iloc[i] if hasattr(et, "iloc") else et
        et_min = et_minutes.iloc[i]
        date_str = str(et_time.date()) if hasattr(et_time, "date") else str(ts.date())
        dow = row.get("btc_day_of_week", 0)

        if date_str != current_date and current_date:
            risk.end_day(current_date)
            if training_mode:
                risk.state.is_killed = False
                risk.state.consecutive_losses = 0
            consecutive_losses = 0
        current_date = date_str

        # Exit
        if active_trade is not None:
            exit_price, exit_reason = _check_exit_btc(
                active_trade, row, i, et_min, bt_cfg, dow, max_hold_bars)
            if exit_price is not None:
                _close_trade(active_trade, exit_price, i, exit_reason, bt_cfg, risk)
                active_trade._signal_type = active_signal_type
                trades.append(active_trade)
                equity = risk.state.current_balance
                net = active_trade.pnl - active_trade.fees
                if net <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                active_trade = None
                active_signal_type = BTCSignalType.NONE

        # Entry
        if active_trade is None and row["signal"] != BTCSignal.FLAT:
            ct_min = (et_min - 60) % 1440
            if risk.can_trade(ct_min, 8):
                atr = row.get("btc_atr14", 0)
                if pd.isna(atr) or atr <= 0:
                    equity_curve.append(equity)
                    continue

                if not training_mode and consecutive_losses >= 3:
                    equity_curve.append(equity)
                    continue

                direction = 1 if row["signal"] == BTCSignal.LONG else -1
                strategy_name = "BTC_TREND_PULLBACK"

                # ATR-based stops
                if stats_bank:
                    sl_mult, rr_ratio = stats_bank.get_exit_params(strategy_name, direction)
                    size_mult = stats_bank.get_size_multiplier(strategy_name, direction)
                    should_take, _ = stats_bank.should_trade(
                        strategy_name=strategy_name, direction=direction)
                    if not should_take:
                        equity_curve.append(equity)
                        continue
                else:
                    sl_mult = 2.0
                    rr_ratio = 3.0  # Wide target — let Chandelier trail handle exits
                    size_mult = 1.0

                entry_price = row["close"] + (bt_cfg.slippage_ticks * bt_cfg.tick_size * direction)
                sl_ticks = max(4, int(atr * sl_mult / bt_cfg.tick_size))
                tp_ticks = max(4, int(sl_ticks * rr_ratio))

                sl_price = entry_price - (sl_ticks * bt_cfg.tick_size * direction)
                tp_price = entry_price + (tp_ticks * bt_cfg.tick_size * direction)

                # Position sizing
                risk_per_contract = sl_ticks * bt_cfg.tick_value
                if risk_per_contract > 0:
                    max_risk = min(200, risk_cfg.max_risk_per_trade)
                    size = max(1, min(3, int(max_risk / risk_per_contract * size_mult)))
                else:
                    size = 1

                # Caution mode: halve size when 1h EMA slope is flat
                htf_caution = row.get("htf_caution", 0)
                if htf_caution and size > 1:
                    size = max(1, size // 2)

                active_trade = Trade(
                    entry_bar=i, entry_price=entry_price, direction=direction,
                    size=size, sl_price=sl_price, tp_price=tp_price)
                active_signal_type = BTCSignalType.BTC_TREND_PULLBACK

        equity_curve.append(equity)

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
    min_trades: int = 8,
    output_path: str = "data/models/btc_strategy_stats.pkl",
) -> StrategyStatsBank:
    """Compute strategy stats for BTC trades."""
    bank = StrategyStatsBank()

    records = []
    for t in trades:
        net = t.pnl - t.fees
        stop_dist = abs(t.entry_price - t.sl_price)
        if stop_dist <= 0:
            continue
        atr_approx = stop_dist / 2.0
        mae_atr = stop_dist / atr_approx if atr_approx > 0 else 2.0
        mfe_atr = t.peak_profit / atr_approx if atr_approx > 0 and t.peak_profit > 0 else 0

        sig_type = getattr(t, "_signal_type", 0)
        try:
            sig_name = BTCSignalType(int(sig_type)).name
        except (ValueError, AttributeError):
            sig_name = "UNKNOWN"

        records.append({
            "net": net, "direction": t.direction,
            "bars": (t.exit_bar or 0) - t.entry_bar,
            "mae_atr": mae_atr, "mfe_atr": mfe_atr,
            "exit_reason": t.exit_reason, "size": t.size,
            "atr": atr_approx, "strategy": sig_name,
            "is_winner": net > 0,
        })

    if not records:
        bank.save(output_path)
        return bank

    rdf = pd.DataFrame(records)
    for strategy in rdf["strategy"].unique():
        if strategy in ("NONE", "UNKNOWN"):
            continue
        for direction, side_label in [(1, "LONG"), (-1, "SHORT")]:
            mask = (rdf["strategy"] == strategy) & (rdf["direction"] == direction)
            grp = rdf[mask]
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

            best_stop, best_pnl = 2.0, float("-inf")
            for test_stop in np.arange(0.75, 4.0, 0.25):
                sim_pnl = sum(
                    r["net"] if r["mae_atr"] <= test_stop
                    else -(test_stop * r["atr"] * r["size"] * tick_value / tick_size)
                    for _, r in grp.iterrows())
                if sim_pnl > best_pnl:
                    best_pnl = sim_pnl
                    best_stop = test_stop

            best_target, best_t_pnl = 3.0, float("-inf")
            for test_target in np.arange(0.75, 6.0, 0.25):
                sim_pnl = sum(
                    test_target * r["atr"] * r["size"] * tick_value / tick_size if r["mfe_atr"] >= test_target
                    else r["net"]
                    for _, r in grp.iterrows())
                if sim_pnl > best_t_pnl:
                    best_t_pnl = sim_pnl
                    best_target = test_target

            optimal_rr = best_target / best_stop if best_stop > 0 else 2.0
            kelly = max(0, (win_rate * payoff - (1 - win_rate)) / payoff) if payoff > 0 and win_rate > 0 else 0

            from src.ai.strategy_stats import ExitProfile
            key = f"{strategy}_{side_label}_ALL"
            bank.profiles[key] = ExitProfile(
                strategy_name=strategy, direction=side_label, regime="ALL",
                n_trades=n, win_rate=round(win_rate, 3),
                avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2),
                payoff_ratio=round(payoff, 3),
                optimal_stop_atr=round(best_stop, 2),
                optimal_target_atr=round(best_target, 2),
                optimal_rr=round(optimal_rr, 2),
                quarter_kelly=round(kelly / 4, 4),
                ev_per_trade=round(ev, 2),
            )
            logger.info(f"  {key}: {n}tr WR={win_rate:.0%} EV=${ev:.0f} "
                        f"stop={best_stop:.1f} target={best_target:.1f} R:R={optimal_rr:.1f}")

    bank.save(output_path)
    return bank
