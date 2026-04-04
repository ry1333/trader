"""V2 backtester — uses session features + V2 signals + optional AI scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.features import extract_ai_features
from src.ai.model import TradeScorer
from src.backtest.engine import BacktestResult, Trade, _close_trade
from src.config import BacktestConfig, RiskConfig, StrategyConfig
from src.features.engine import compute_features
from src.features.session import add_session_features
from src.risk.engine import RiskEngine
from src.strategy.regime import add_regime
from src.strategy.signals_v2 import Signal, SignalType, generate_signals_v2


def _check_exit_v2(
    trade: Trade,
    row: pd.Series,
    bar_idx: int,
    ct_minutes: int,
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    signal_type: int,
) -> tuple[float | None, str]:
    """V2 exit logic with trailing stop to let winners run."""

    # Forced flatten before session close
    if ct_minutes >= 900:
        return row["close"], "session_flatten"

    # Max hold time
    bars_held = bar_idx - trade.entry_bar
    if bars_held >= strategy_cfg.max_hold_bars:
        return row["close"], "max_hold"

    # Stress regime: only exit if trade is losing
    if row.get("regime") == 0:
        current_pnl = (row["close"] - trade.entry_price) * trade.direction
        if current_pnl < 0:
            return row["close"], "stress_exit"

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

    # ── Trailing stop: lock in profits ────────────────────────────────
    # Track max favorable excursion
    if trade.direction == 1:
        current_profit = row["high"] - trade.entry_price
    else:
        current_profit = trade.entry_price - row["low"]

    if current_profit > trade.peak_profit:
        trade.peak_profit = current_profit

    # Once profit exceeds 50% of target, trail at 40% of peak
    stop_distance = abs(trade.entry_price - trade.sl_price)
    target_distance = abs(trade.tp_price - trade.entry_price)
    if trade.peak_profit > target_distance * 0.50:
        trail_level = trade.peak_profit * 0.60  # Keep 60% of peak profit
        current_pnl = (row["close"] - trade.entry_price) * trade.direction
        if current_pnl < trail_level * 0.40:
            # Profit has pulled back more than 60% from peak — trail out
            return row["close"], "trailing_stop"

    # VWAP reversion: exit when price returns to VWAP (but only if profitable)
    if signal_type == SignalType.VWAP_REVERSION:
        vwap = row.get("vwap", None)
        if vwap is not None and not pd.isna(vwap):
            current_pnl = (row["close"] - trade.entry_price) * trade.direction
            if current_pnl > 0:
                if trade.direction == 1 and row["close"] >= vwap:
                    return row["close"], "vwap_target"
                if trade.direction == -1 and row["close"] <= vwap:
                    return row["close"], "vwap_target"

    return None, ""


def run_backtest_v2(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    scorer: TradeScorer | None = None,
    starting_balance: float = 50_000.0,
    collect_features: bool = False,
    immortal: bool = False,
) -> tuple[BacktestResult, pd.DataFrame]:
    """V2 backtest with session features + improved signals.

    immortal: if True, resets balance each day (never killed). Use for training data collection.
    """

    # Compute all features
    df = compute_features(df)
    df = add_session_features(df)
    df = add_regime(df)
    df["signal"], df["signal_type"] = generate_signals_v2(df)

    risk = RiskEngine(risk_cfg, starting_balance)
    trades: list[Trade] = []
    active_trade: Trade | None = None
    active_signal_type: int = SignalType.NONE
    equity = starting_balance
    equity_curve = []
    current_date = ""
    feature_records: list[dict] = []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]

        if hasattr(ts, "tz_convert"):
            ct = ts.tz_convert("US/Central")
        else:
            ct = ts
        ct_minutes = ct.hour * 60 + ct.minute
        date_str = str(ct.date())

        if date_str != current_date and current_date:
            risk.end_day(current_date)
            if immortal:
                # Reset for next day — collect unlimited training data
                risk.state.is_killed = False
                risk.state.current_balance = starting_balance
                risk.state.total_pnl = 0.0
                risk.state.consecutive_losses = 0
                equity = starting_balance
        current_date = date_str

        # ── Check exit ────────────────────────────────────────────────
        if active_trade is not None:
            exit_price, exit_reason = _check_exit_v2(
                active_trade, row, i, ct_minutes, strategy_cfg, bt_cfg, active_signal_type
            )
            if exit_price is not None:
                _close_trade(active_trade, exit_price, i, exit_reason, bt_cfg, risk)
                trades.append(active_trade)
                equity = risk.state.current_balance
                active_trade = None
                active_signal_type = SignalType.NONE

        # ── Check entry ───────────────────────────────────────────────
        if active_trade is None and row["signal"] != Signal.FLAT:
            if risk.can_trade(ct_minutes, strategy_cfg.max_trades_per_day):
                atr = row.get("atr_14", 0)
                atr_50 = row.get("atr_50", 0)
                # Volatility gate: skip when ATR is 2x+ its 50-bar average (crash days)
                vol_gated = not pd.isna(atr_50) and atr_50 > 0 and atr > atr_50 * 2.0
                if not vol_gated and not pd.isna(atr) and atr > 0:
                    # AI scoring
                    ai_features = {}
                    should_take = True
                    win_prob = 0.5

                    if scorer or collect_features:
                        ai_features = extract_ai_features(df, i)
                        if scorer and scorer.model is not None:
                            should_take, win_prob = scorer.should_trade(ai_features)

                    if should_take:
                        sig_type = int(row["signal_type"])

                        # Signal-type-specific stop/target sizing
                        # Wider stops to survive noise, higher R:R to compensate
                        if sig_type == SignalType.ORB:
                            sl_mult, rr_ratio = 2.5, 2.0  # Wide stop for breakout noise
                        elif sig_type == SignalType.VWAP_REVERSION:
                            sl_mult, rr_ratio = 2.0, 1.5  # VWAP target is the exit anyway
                        else:  # TREND_CONTINUATION
                            sl_mult, rr_ratio = 2.5, 3.0  # Let winners run big

                        size = risk.compute_position_size(atr, bt_cfg.tick_size, bt_cfg.tick_value)
                        sl_ticks = risk.compute_stop_ticks(atr, bt_cfg.tick_size, sl_mult)
                        tp_ticks = risk.compute_target_ticks(sl_ticks, rr_ratio)

                        direction = 1 if row["signal"] == Signal.LONG else -1
                        entry_price = row["close"] + (bt_cfg.slippage_ticks * bt_cfg.tick_size * direction)
                        sl_price = entry_price - (sl_ticks * bt_cfg.tick_size * direction)
                        tp_price = entry_price + (tp_ticks * bt_cfg.tick_size * direction)

                        active_trade = Trade(
                            entry_bar=i,
                            entry_price=entry_price,
                            direction=direction,
                            size=size,
                            sl_price=sl_price,
                            tp_price=tp_price,
                        )
                        active_signal_type = sig_type

                    if collect_features:
                        feature_records.append({
                            "entry_bar": i,
                            "signal": int(row["signal"]),
                            "signal_type": int(row["signal_type"]),
                            "ai_approved": should_take,
                            "win_prob": win_prob,
                            **ai_features,
                        })

        equity_curve.append(equity)

    if active_trade is not None:
        _close_trade(active_trade, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active_trade)

    if current_date:
        risk.end_day(current_date)

    eq_series = pd.Series(equity_curve, index=df["timestamp"].values)
    daily_pnl = pd.Series(
        [d.pnl for d in risk.state.daily_history],
        index=[d.date for d in risk.state.daily_history],
    )

    result = BacktestResult(
        trades=trades,
        equity_curve=eq_series,
        daily_pnl=daily_pnl,
        risk_summary=risk.summary,
        df=df,
    )

    features_df = pd.DataFrame(feature_records)
    if not features_df.empty and trades:
        trade_pnls = {t.entry_bar: t.pnl - t.fees for t in trades}
        features_df["net_pnl"] = features_df["entry_bar"].map(trade_pnls).fillna(0.0)
        features_df["was_winner"] = (features_df["net_pnl"] > 0).astype(int)

    return result, features_df
