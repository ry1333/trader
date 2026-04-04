"""V2 backtester — uses session features + V2 signals + optional AI scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.features import extract_ai_features
from src.ai.ev_model import EVScorer
from src.ai.model import EnsembleScorer, TradeScorer
from src.backtest.engine import BacktestResult, Trade, _close_trade
from src.config import BacktestConfig, RiskConfig, StrategyConfig
from src.features.engine import compute_features
from src.features.session import add_session_features
from src.risk.engine import RiskEngine
from src.strategy.regime import add_regime
from src.strategy.signals_v3 import Signal, SignalType, generate_signals_v3


def _check_exit_v2(
    trade: Trade,
    row: pd.Series,
    bar_idx: int,
    ct_minutes: int,
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    signal_type: int,
) -> tuple[float | None, str]:
    """V3 exit logic — trailing stop, breakeven, time-decay, improved stress."""

    bars_held = bar_idx - trade.entry_bar
    current_pnl = (row["close"] - trade.entry_price) * trade.direction
    atr = row.get("atr_14", 0)
    if pd.isna(atr):
        atr = 0

    # ── Forced flatten before session close ───────────────────────────
    if ct_minutes >= 900:
        return row["close"], "session_flatten"

    # ── Stress regime: exit after 6 bars always, or immediately if losing
    if row.get("regime") == 0:
        if bars_held > 6:
            return row["close"], "stress_exit"
        if current_pnl < 0:
            return row["close"], "stress_exit"

    # ── Time-decay exit: if significantly losing after 24 bars (2 hours), cut
    # Audit: 106 time_decay exits at -$6,660. Loosened from 18 to 24 bars.
    if bars_held >= 24 and current_pnl < -atr * 0.5:
        return row["close"], "time_decay"

    # ── Max hold time ─────────────────────────────────────────────────
    if bars_held >= strategy_cfg.max_hold_bars:
        return row["close"], "max_hold"

    # ── Standard SL/TP ────────────────────────────────────────────────
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

    # ── Track peak profit for trailing stop ───────────────────────────
    if trade.direction == 1:
        bar_max_profit = row["high"] - trade.entry_price
    else:
        bar_max_profit = trade.entry_price - row["low"]

    if bar_max_profit > trade.peak_profit:
        trade.peak_profit = bar_max_profit

    target_distance = abs(trade.tp_price - trade.entry_price)

    # ── Breakeven stop: DISABLED — audit showed 144 trades, -$5,419, 0% WR
    # The stop was triggering on noise pullbacks, not real reversals
    # Replaced by trailing stop which is more adaptive

    # ── Trailing stop: activate at 75% of target, keep 50% of peak ───
    if target_distance > 0 and trade.peak_profit > target_distance * 0.75:
        trail_level = trade.peak_profit * 0.50  # Keep 50% of peak profit
        if current_pnl < trail_level * 0.50:
            # Profit pulled back >50% from trail level
            return row["close"], "trailing_stop"

    # ── VWAP reversion: exit near VWAP band (±0.5 ATR) if profitable ─
    if signal_type == SignalType.VWAP_REVERSION:
        vwap = row.get("vwap", None)
        if vwap is not None and not pd.isna(vwap) and atr > 0 and current_pnl > 0:
            vwap_band = 0.5 * atr
            if trade.direction == 1 and row["close"] >= (vwap - vwap_band):
                return row["close"], "vwap_target"
            if trade.direction == -1 and row["close"] <= (vwap + vwap_band):
                return row["close"], "vwap_target"

    return None, ""


def run_backtest_v2(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    scorer: TradeScorer | EnsembleScorer | EVScorer | None = None,
    starting_balance: float = 50_000.0,
    collect_features: bool = False,
    training_mode: bool = False,
    risk_engine: RiskEngine | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """V2 backtest with session features + improved signals.

    training_mode: if True, resets is_killed each day for data collection.
                   Does NOT reset balance (preserves realistic equity dynamics).
    risk_engine: optional shared RiskEngine (for multi-instrument portfolio heat).
    """

    # Compute all features
    df = compute_features(df)
    df = add_session_features(df)
    df = add_regime(df)
    df["signal"], df["signal_type"] = generate_signals_v3(df)

    risk = risk_engine if risk_engine is not None else RiskEngine(risk_cfg, starting_balance)
    trades: list[Trade] = []
    active_trade: Trade | None = None
    active_signal_type: int = SignalType.NONE
    equity = risk.state.current_balance
    equity_curve = []
    current_date = ""
    feature_records: list[dict] = []

    # Signal-type consecutive loss tracking for circuit breaker
    signal_type_losses: dict[int, int] = {}

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
            if training_mode:
                # Only reset kill flag — preserve balance for realistic dynamics
                risk.state.is_killed = False
                risk.state.consecutive_losses = 0
            # Reset signal-type circuit breakers daily
            signal_type_losses.clear()
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

                # Track signal-type losses for circuit breaker
                net_pnl = active_trade.pnl - active_trade.fees
                if net_pnl <= 0:
                    signal_type_losses[active_signal_type] = signal_type_losses.get(active_signal_type, 0) + 1
                else:
                    signal_type_losses[active_signal_type] = 0

                active_trade = None
                active_signal_type = SignalType.NONE

        # ── Check entry ───────────────────────────────────────────────
        if active_trade is None and row["signal"] != Signal.FLAT:
            if risk.can_trade(ct_minutes, strategy_cfg.max_trades_per_day):
                atr = row.get("atr_14", 0)
                atr_50 = row.get("atr_50", 0)

                # ── Trade filters ─────────────────────────────────────
                # Percentile vol gate: skip if ATR below 50th percentile of last 200 bars
                if i >= 200 and not pd.isna(atr):
                    atr_lookback = df["atr_14"].iloc[max(0, i - 200):i].dropna()
                    if len(atr_lookback) > 0 and atr < atr_lookback.quantile(0.50):
                        equity_curve.append(equity)
                        continue

                # Time-of-day filter: only trade 8:30 AM - 1:00 PM CT
                # Audit: pre-open -$3,761, afternoon -$298. Only open/mid-morning/lunch profitable.
                if ct_minutes < 510 or ct_minutes >= 780:
                    equity_curve.append(equity)
                    continue

                # Signal-type circuit breaker: 3 consecutive losses → skip
                sig_type = int(row["signal_type"])
                if signal_type_losses.get(sig_type, 0) >= 3:
                    equity_curve.append(equity)
                    continue

                # Volatility spike gate: skip extreme vol days
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
                            # Minimum confidence floor
                            if win_prob < 0.50:
                                should_take = False

                    if should_take:
                        # Signal-type-specific stop/target sizing
                        if sig_type == SignalType.ORB:
                            sl_mult, rr_ratio = 2.5, 2.0
                        elif sig_type == SignalType.VWAP_REVERSION:
                            sl_mult, rr_ratio = 2.0, 1.5
                        elif sig_type == SignalType.TREND_CONTINUATION:
                            sl_mult, rr_ratio = 2.5, 3.0
                        elif sig_type == SignalType.EMA_PULLBACK:
                            sl_mult, rr_ratio = 1.5, 2.0  # Tight stop near EMA
                        elif sig_type == SignalType.RANGE_BREAKOUT:
                            sl_mult, rr_ratio = 2.0, 2.5
                        elif sig_type == SignalType.MOMENTUM_IGNITION:
                            sl_mult, rr_ratio = 2.0, 2.0  # Quick in/out
                        elif sig_type == SignalType.VOL_CONTRACTION:
                            sl_mult, rr_ratio = 2.0, 3.0  # Squeeze = big moves
                        elif sig_type == SignalType.RSI_REVERSAL:
                            sl_mult, rr_ratio = 1.5, 2.0
                        else:  # PREV_DAY_LEVEL, SESSION_LEVEL, etc.
                            sl_mult, rr_ratio = 2.0, 2.0

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
