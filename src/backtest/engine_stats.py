"""Backtester using data-driven strategy statistics instead of ML prediction.

Key difference from engine_v2:
- No AI take/skip decision — all signals are taken
- Stop/target/size determined by historical statistics per strategy
- Kelly criterion sizing: bigger on high-edge strategies, smaller on low-edge
- Rule-based filters still active (time, session quality, vol gate, circuit breaker)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.strategy_stats import StrategyStatsBank
from src.backtest.engine import BacktestResult, Trade, _close_trade
from src.config import BacktestConfig, RiskConfig, StrategyConfig
from src.features.engine import compute_features
from src.features.session import add_session_features
from src.filters.session_quality import SessionGrade, compute_session_quality
from src.risk.engine import RiskEngine
from src.strategy.regime import add_regime
from src.strategy.signals_v3 import Signal, SignalType, generate_signals_v3


def _check_exit_stats(
    trade: Trade,
    row: pd.Series,
    bar_idx: int,
    ct_minutes: int,
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    signal_type: int,
) -> tuple[float | None, str]:
    """Exit logic — same proven logic from Sharpe 1.77 run."""

    bars_held = bar_idx - trade.entry_bar
    current_pnl = (row["close"] - trade.entry_price) * trade.direction
    atr = row.get("atr_14", 0)
    if pd.isna(atr):
        atr = 0

    if ct_minutes >= 900:
        return row["close"], "session_flatten"

    if row.get("regime") == 0:
        if bars_held > 10:
            return row["close"], "stress_exit"
        if current_pnl < -atr * 0.5:
            return row["close"], "stress_exit"

    if bars_held >= 30 and current_pnl < -atr:
        return row["close"], "time_decay"

    if bars_held >= strategy_cfg.max_hold_bars:
        if current_pnl <= 0:
            return row["close"], "max_hold"
        if current_pnl > 0 and trade.peak_profit > 0:
            if current_pnl < trade.peak_profit * 0.40:
                return row["close"], "max_hold_trail"

    # Standard SL/TP
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

    # Track peak profit
    if trade.direction == 1:
        bar_max = row["high"] - trade.entry_price
    else:
        bar_max = trade.entry_price - row["low"]
    if bar_max > trade.peak_profit:
        trade.peak_profit = bar_max

    target_distance = abs(trade.tp_price - trade.entry_price)

    # Trailing stop
    if target_distance > 0 and trade.peak_profit > target_distance * 0.60:
        trail_keep = trade.peak_profit * 0.40
        if current_pnl < trail_keep:
            return row["close"], "trailing_stop"

    # VWAP target for reversion trades
    REVERSION_TYPES = {SignalType.VWAP_REVERSION, SignalType.VWAP_RECLAIM,
                       SignalType.FAILED_BREAKOUT, SignalType.RSI_REVERSAL}
    if signal_type in REVERSION_TYPES:
        vwap = row.get("vwap", None)
        if vwap is not None and not pd.isna(vwap) and atr > 0 and current_pnl > 0:
            vwap_band = 0.5 * atr
            if trade.direction == 1 and row["close"] >= (vwap - vwap_band):
                return row["close"], "vwap_target"
            if trade.direction == -1 and row["close"] <= (vwap + vwap_band):
                return row["close"], "vwap_target"

    return None, ""


def run_backtest_stats(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    stats_bank: StrategyStatsBank | None = None,
    starting_balance: float = 50_000.0,
    training_mode: bool = False,
) -> tuple[BacktestResult, list]:
    """Backtest using strategy statistics for stop/target/sizing.

    No ML prediction. All signals taken. Stats bank provides:
    - Optimal stop (ATR multiple per strategy)
    - Optimal target (ATR multiple per strategy)
    - Kelly-based position size per strategy
    - EV filter (skip negative EV strategies)
    """

    df = compute_features(df)
    df = add_session_features(df)
    df = add_regime(df)
    df["signal"], df["signal_type"] = generate_signals_v3(df)

    risk = RiskEngine(risk_cfg, starting_balance)
    trades: list[Trade] = []
    active_trade: Trade | None = None
    active_signal_type: int = SignalType.NONE
    equity = starting_balance
    equity_curve = []
    current_date = ""
    signal_type_losses: dict[int, int] = {}

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        ct = ts.tz_convert("US/Central") if hasattr(ts, "tz_convert") else ts
        ct_minutes = ct.hour * 60 + ct.minute
        date_str = str(ct.date())

        if date_str != current_date and current_date:
            risk.end_day(current_date)
            if training_mode:
                risk.state.is_killed = False
                risk.state.consecutive_losses = 0
            signal_type_losses.clear()
        current_date = date_str

        # Exit
        if active_trade is not None:
            exit_price, exit_reason = _check_exit_stats(
                active_trade, row, i, ct_minutes, strategy_cfg, bt_cfg, active_signal_type)
            if exit_price is not None:
                _close_trade(active_trade, exit_price, i, exit_reason, bt_cfg, risk)
                # Tag the trade with its signal type for stats collection
                active_trade._signal_type = active_signal_type
                trades.append(active_trade)
                equity = risk.state.current_balance
                net = active_trade.pnl - active_trade.fees
                if net <= 0:
                    signal_type_losses[active_signal_type] = signal_type_losses.get(active_signal_type, 0) + 1
                else:
                    signal_type_losses[active_signal_type] = 0
                active_trade = None
                active_signal_type = SignalType.NONE

        # Entry
        if active_trade is None and row["signal"] != Signal.FLAT:
            if risk.can_trade(ct_minutes, strategy_cfg.max_trades_per_day):
                atr = row.get("atr_14", 0)
                atr_50 = row.get("atr_50", 0)

                # Rule-based filters (proven to work)
                if i >= 200 and not pd.isna(atr):
                    atr_lb = df["atr_14"].iloc[max(0, i - 200):i].dropna()
                    if len(atr_lb) > 0 and atr < atr_lb.quantile(0.50):
                        equity_curve.append(equity)
                        continue

                if ct_minutes < 510 or ct_minutes >= 780:
                    equity_curve.append(equity)
                    continue

                sig_type = int(row["signal_type"])
                if signal_type_losses.get(sig_type, 0) >= 3:
                    equity_curve.append(equity)
                    continue

                session = compute_session_quality(df, i)
                if session.grade == SessionGrade.D:
                    equity_curve.append(equity)
                    continue

                vol_gated = not pd.isna(atr_50) and atr_50 > 0 and atr > atr_50 * 2.0
                if not vol_gated and not pd.isna(atr) and atr > 0:
                    direction = 1 if row["signal"] == Signal.LONG else -1
                    strategy_name = SignalType(sig_type).name

                    # Get optimal parameters from stats bank
                    if stats_bank:
                        # Skip negative EV strategies
                        should_take, _ = stats_bank.should_trade(
                            strategy_name=strategy_name, direction=direction)
                        if not should_take:
                            equity_curve.append(equity)
                            continue

                        sl_mult, rr_ratio = stats_bank.get_exit_params(strategy_name, direction)
                        size_mult = stats_bank.get_size_multiplier(strategy_name, direction)
                    else:
                        # Default R:R (no stats yet — first training pass)
                        default_rr = {
                            SignalType.ORB: (2.5, 2.0), SignalType.VWAP_REVERSION: (2.0, 1.5),
                            SignalType.TREND_CONTINUATION: (2.5, 3.0), SignalType.EMA_PULLBACK: (1.5, 2.0),
                            SignalType.RANGE_BREAKOUT: (2.0, 2.5), SignalType.MOMENTUM_IGNITION: (2.0, 2.0),
                            SignalType.VOL_CONTRACTION: (2.0, 3.0), SignalType.RSI_REVERSAL: (1.5, 2.0),
                            SignalType.FAILED_BREAKOUT: (1.0, 1.5), SignalType.VWAP_RECLAIM: (1.5, 2.0),
                            SignalType.ODPC: (1.5, 2.0),
                        }
                        sl_mult, rr_ratio = default_rr.get(sig_type, (2.0, 2.0))
                        size_mult = 1.0

                    size = risk.compute_position_size(atr, bt_cfg.tick_size, bt_cfg.tick_value)
                    size = max(1, int(size * size_mult * session.size_multiplier))

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

    if active_trade:
        active_trade._signal_type = active_signal_type
        _close_trade(active_trade, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active_trade)
    if current_date:
        risk.end_day(current_date)

    eq_series = pd.Series(equity_curve, index=df["timestamp"].values)
    daily_pnl = pd.Series(
        [d.pnl for d in risk.state.daily_history],
        index=[d.date for d in risk.state.daily_history])

    result = BacktestResult(
        trades=trades, equity_curve=eq_series, daily_pnl=daily_pnl,
        risk_summary=risk.summary, df=df)

    return result, trades
