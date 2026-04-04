"""AI-enhanced backtester — same as base engine but with ML scoring.

Two modes:
1. Data collection: run without model, collect features + outcomes for training
2. AI-filtered: run with trained model, only take trades the AI approves
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.features import AI_FEATURE_COLS, extract_ai_features
from src.ai.model import TradeScorer
from src.backtest.engine import BacktestResult, Trade, _check_exit, _close_trade
from src.config import BacktestConfig, RiskConfig, StrategyConfig
from src.features.engine import compute_features
from src.risk.engine import RiskEngine
from src.strategy.regime import add_regime
from src.strategy.signals import Signal, generate_signals


def run_backtest_ai(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    scorer: TradeScorer | None = None,
    starting_balance: float = 50_000.0,
    collect_features: bool = True,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Run backtest with AI scoring.

    If scorer is None, takes all signals (data collection mode).
    If scorer is provided, only takes trades the AI approves.

    Returns (BacktestResult, features_df) where features_df has
    one row per signal with AI features + outcome.
    """
    # Compute features and regime
    df = compute_features(df)
    df = add_regime(df)
    df["signal"] = generate_signals(df)

    risk = RiskEngine(risk_cfg, starting_balance)
    trades: list[Trade] = []
    active_trade: Trade | None = None
    equity = starting_balance
    equity_curve = []
    current_date = ""

    # Feature collection for training
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
        current_date = date_str

        # ── Check active trade for exit ───────────────────────────────
        if active_trade is not None:
            exit_price, exit_reason = _check_exit(
                active_trade, row, i, ct_minutes, strategy_cfg, bt_cfg
            )
            if exit_price is not None:
                _close_trade(active_trade, exit_price, i, exit_reason, bt_cfg, risk)
                trades.append(active_trade)
                equity = risk.state.current_balance
                active_trade = None

        # ── Check for new entry ───────────────────────────────────────
        if active_trade is None and row["signal"] != Signal.FLAT:
            if risk.can_trade(ct_minutes, strategy_cfg.max_trades_per_day):
                atr = row.get("atr_14", 0)
                if not pd.isna(atr) and atr > 0:
                    # Extract AI features
                    ai_features = extract_ai_features(df, i) if collect_features or scorer else {}

                    # AI gate: check if model approves this trade
                    should_take = True
                    win_prob = 0.5
                    if scorer and scorer.model is not None:
                        should_take, win_prob = scorer.should_trade(ai_features)

                    if should_take:
                        size = risk.compute_position_size(atr, bt_cfg.tick_size, bt_cfg.tick_value)
                        sl_ticks = risk.compute_stop_ticks(atr, bt_cfg.tick_size)
                        tp_ticks = risk.compute_target_ticks(sl_ticks)

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

                    # Record features for training (even if AI rejected)
                    if collect_features:
                        record = {
                            "entry_bar": i,
                            "signal": int(row["signal"]),
                            "ai_approved": should_take,
                            "win_prob": win_prob,
                            **ai_features,
                        }
                        feature_records.append(record)

        equity_curve.append(equity)

    # Close remaining trade
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

    # Build features DataFrame with trade outcomes
    features_df = pd.DataFrame(feature_records)
    if not features_df.empty and trades:
        # Add P&L outcomes to features (for training)
        trade_pnls = {t.entry_bar: t.pnl - t.fees for t in trades}
        features_df["net_pnl"] = features_df["entry_bar"].map(trade_pnls).fillna(0.0)
        features_df["was_winner"] = (features_df["net_pnl"] > 0).astype(int)

    return result, features_df
