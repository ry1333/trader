"""Walk-forward validation framework.

Splits data into rolling train/val/test windows to prevent overfitting.
Each window produces an independent backtest result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.backtest.engine import BacktestResult, run_backtest
from src.config import BacktestConfig, RiskConfig, StrategyConfig


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult | None = None


def walk_forward(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    starting_balance: float = 50_000.0,
) -> list[WalkForwardWindow]:
    """Run walk-forward validation across the dataset.

    For each window:
    1. Train period: used for regime calibration (future: ML training)
    2. Test period: out-of-sample backtest with prop rules
    """
    train_days = bt_cfg.train_window_days
    test_days = bt_cfg.test_window_days
    step_days = bt_cfg.walk_forward_step_days

    timestamps = df["timestamp"].sort_values()
    data_start = timestamps.iloc[0]
    data_end = timestamps.iloc[-1]

    windows: list[WalkForwardWindow] = []
    cursor = data_start

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > data_end:
            break

        window = WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # Get test period data
        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)
        test_df = df[test_mask].reset_index(drop=True)

        if len(test_df) < 50:  # Need minimum bars
            cursor += pd.Timedelta(days=step_days)
            continue

        logger.info(
            f"WF window: train {train_start.date()}→{train_end.date()}, "
            f"test {test_start.date()}→{test_end.date()} ({len(test_df)} bars)"
        )

        window.result = run_backtest(
            test_df, strategy_cfg, risk_cfg, bt_cfg, starting_balance
        )
        windows.append(window)
        cursor += pd.Timedelta(days=step_days)

    return windows


def aggregate_results(windows: list[WalkForwardWindow]) -> dict:
    """Aggregate walk-forward results into a summary."""
    results = [w.result for w in windows if w.result is not None]
    if not results:
        return {"error": "No valid windows"}

    all_trades = []
    for r in results:
        all_trades.extend(r.trades)

    total_pnl = sum(r.net_pnl for r in results)
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if (t.pnl - t.fees) > 0)
    max_dd = min(r.max_drawdown for r in results)

    daily_pnls = pd.concat([r.daily_pnl for r in results])
    sharpe = 0.0
    if len(daily_pnls) > 1 and daily_pnls.std() > 0:
        import numpy as np
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * np.sqrt(252)

    # Exit reason breakdown
    exit_reasons: dict[str, int] = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "windows": len(results),
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_window": round(total_pnl / len(results), 2),
        "win_rate": round(wins / total_trades, 4) if total_trades > 0 else 0,
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "exit_reasons": exit_reasons,
    }
