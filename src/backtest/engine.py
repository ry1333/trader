"""Walk-forward backtester with prop rule enforcement.

Simulates trading with:
- Realistic transaction costs and slippage
- Topstep daily loss / max loss / consistency rules
- Forced flatten before session close
- Per-bar position tracking and P&L
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BacktestConfig, RiskConfig, StrategyConfig
from src.features.engine import FEATURE_COLS, compute_features
from src.risk.engine import RiskEngine
from src.strategy.regime import add_regime
from src.strategy.signals import Signal, generate_signals


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    direction: int  # +1 long, -1 short
    size: int
    sl_price: float
    tp_price: float
    exit_bar: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    fees: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    daily_pnl: pd.Series
    risk_summary: dict
    df: pd.DataFrame  # bars with features/signals/regime

    @property
    def net_pnl(self) -> float:
        return sum(t.pnl - t.fees for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if (t.pnl - t.fees) > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl - t.fees for t in self.trades if (t.pnl - t.fees) > 0)
        gross_loss = abs(sum(t.pnl - t.fees for t in self.trades if (t.pnl - t.fees) < 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd = self.equity_curve - peak
        return dd.min()

    @property
    def sharpe(self) -> float:
        if self.daily_pnl.empty or self.daily_pnl.std() == 0:
            return 0.0
        # Annualized: ~252 trading days
        return (self.daily_pnl.mean() / self.daily_pnl.std()) * np.sqrt(252)

    def summary(self) -> dict:
        return {
            "total_trades": len(self.trades),
            "net_pnl": round(self.net_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 3),
            "trading_days": len(self.daily_pnl),
            **self.risk_summary,
        }


def run_backtest(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    starting_balance: float = 50_000.0,
) -> BacktestResult:
    """Run a single backtest pass over the provided bar data."""

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

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]

        # Convert to CT minutes for session checks
        if hasattr(ts, "tz_convert"):
            ct = ts.tz_convert("US/Central")
        else:
            ct = ts
        ct_minutes = ct.hour * 60 + ct.minute
        date_str = str(ct.date())

        # New day: end previous day
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

        equity_curve.append(equity)

    # Close any remaining trade at end of data
    if active_trade is not None:
        _close_trade(active_trade, df.iloc[-1]["close"], len(df) - 1, "end_of_data", bt_cfg, risk)
        trades.append(active_trade)

    # End final day
    if current_date:
        risk.end_day(current_date)

    # Build results
    eq_series = pd.Series(equity_curve, index=df["timestamp"].values)
    daily_pnl = pd.Series(
        [d.pnl for d in risk.state.daily_history],
        index=[d.date for d in risk.state.daily_history],
    )

    return BacktestResult(
        trades=trades,
        equity_curve=eq_series,
        daily_pnl=daily_pnl,
        risk_summary=risk.summary,
        df=df,
    )


def _check_exit(
    trade: Trade,
    row: pd.Series,
    bar_idx: int,
    ct_minutes: int,
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
) -> tuple[float | None, str]:
    """Check if active trade should be exited. Returns (exit_price, reason) or (None, '')."""

    # Forced flatten before session close (3:00 PM CT = 900 minutes)
    if ct_minutes >= 900:
        return row["close"], "session_flatten"

    # Max hold time
    bars_held = bar_idx - trade.entry_bar
    if bars_held >= strategy_cfg.max_hold_bars:
        return row["close"], "max_hold"

    # Stop loss
    if trade.direction == 1:  # Long
        if row["low"] <= trade.sl_price:
            return trade.sl_price + (bt_cfg.slippage_ticks * bt_cfg.tick_size), "stop_loss"
        if row["high"] >= trade.tp_price:
            return trade.tp_price - (bt_cfg.slippage_ticks * bt_cfg.tick_size), "take_profit"
    else:  # Short
        if row["high"] >= trade.sl_price:
            return trade.sl_price - (bt_cfg.slippage_ticks * bt_cfg.tick_size), "stop_loss"
        if row["low"] <= trade.tp_price:
            return trade.tp_price + (bt_cfg.slippage_ticks * bt_cfg.tick_size), "take_profit"

    # Stress regime: exit immediately
    if row.get("regime") == 0:  # Regime.STRESS
        return row["close"], "stress_exit"

    return None, ""


def _close_trade(
    trade: Trade,
    exit_price: float,
    bar_idx: int,
    reason: str,
    bt_cfg: BacktestConfig,
    risk: RiskEngine,
) -> None:
    """Finalize a trade with P&L and fees."""
    trade.exit_bar = bar_idx
    trade.exit_price = exit_price
    trade.exit_reason = reason

    price_diff = (exit_price - trade.entry_price) * trade.direction
    ticks = price_diff / bt_cfg.tick_size
    trade.pnl = ticks * bt_cfg.tick_value * trade.size
    trade.fees = bt_cfg.cost_per_side_per_contract * 2 * trade.size

    risk.record_trade(trade.pnl, trade.fees)
