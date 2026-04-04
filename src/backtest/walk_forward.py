"""Walk-forward validation with rolling model retraining.

Each window:
1. Train: run backtest in training_mode → collect features → train fresh AI model
2. Validate: run backtest with model → sweep thresholds → pick best
3. Test: run backtest with validated model → true OOS result
4. Roll forward 1 month, repeat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.model import EnsembleScorer, TradeScorer
from src.ai.trainer import train_and_save
from src.backtest.engine import BacktestResult
from src.backtest.engine_v2 import run_backtest_v2
from src.config import BacktestConfig, RiskConfig, StrategyConfig


@dataclass
class WalkForwardWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    # Results
    train_trades: int = 0
    train_win_rate: float = 0.0
    val_result: BacktestResult | None = None
    test_result: BacktestResult | None = None
    model_metadata: dict = field(default_factory=dict)
    best_threshold: float = 0.50
    degraded: bool = False  # True if OOS much worse than validation


def walk_forward(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    bt_cfg: BacktestConfig,
    starting_balance: float = 50_000.0,
    instrument: str = "MES",
) -> list[WalkForwardWindow]:
    """Run walk-forward validation with rolling model retraining.

    For each window:
    1. Train (train_window_days): collect features + train model
    2. Validate (val_window_days): sweep thresholds
    3. Test (test_window_days): true OOS
    Roll forward by walk_forward_step_days.
    """
    train_days = bt_cfg.train_window_days
    val_days = bt_cfg.val_window_days
    test_days = bt_cfg.test_window_days
    step_days = bt_cfg.walk_forward_step_days

    timestamps = df["timestamp"].sort_values()
    data_start = timestamps.iloc[0]
    data_end = timestamps.iloc[-1]

    windows: list[WalkForwardWindow] = []
    cursor = data_start
    window_id = 0

    model_dir = Path("data/models/walk_forward")
    model_dir.mkdir(parents=True, exist_ok=True)

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        val_start = train_end
        val_end = val_start + pd.Timedelta(days=val_days)
        test_start = val_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > data_end:
            break

        # Slice data
        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        val_mask = (df["timestamp"] >= val_start) & (df["timestamp"] < val_end)
        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)

        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        if len(train_df) < 200 or len(val_df) < 50 or len(test_df) < 50:
            cursor += pd.Timedelta(days=step_days)
            continue

        window = WalkForwardWindow(
            window_id=window_id,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            test_start=test_start, test_end=test_end,
        )

        logger.info(
            f"WF #{window_id}: train {train_start.date()}→{train_end.date()}, "
            f"val {val_start.date()}→{val_end.date()}, "
            f"test {test_start.date()}→{test_end.date()}"
        )

        # ── Phase 1: Train ────────────────────────────────────────────
        train_result, train_features = run_backtest_v2(
            train_df, strategy_cfg, risk_cfg, bt_cfg,
            collect_features=True, training_mode=True,
        )
        window.train_trades = len(train_result.trades)
        window.train_win_rate = train_result.win_rate

        if window.train_trades < 20:
            logger.warning(f"WF #{window_id}: only {window.train_trades} training trades, skipping")
            cursor += pd.Timedelta(days=step_days)
            continue

        # Train AI model
        trades_data = pd.DataFrame([{
            "entry_bar": t.entry_bar,
            "net_pnl": t.pnl - t.fees,
            "pnl": t.pnl,
            "fees": t.fees,
        } for t in train_result.trades])

        model_path = model_dir / f"{instrument}_window_{window_id}.pkl"
        metadata = train_and_save(
            trades_data, train_features, model_path,
            use_ensemble=window.train_trades >= 30,
        )
        window.model_metadata = metadata

        if "error" in metadata:
            logger.warning(f"WF #{window_id}: training failed: {metadata}")
            cursor += pd.Timedelta(days=step_days)
            continue

        # ── Phase 2: Validate — sweep thresholds ──────────────────────
        scorer = EnsembleScorer(model_path)
        best_thresh = 0.50
        best_score = float("-inf")

        for thresh in np.arange(0.40, 0.65, 0.05):
            scorer.threshold = thresh
            val_result, _ = run_backtest_v2(
                val_df, strategy_cfg, risk_cfg, bt_cfg, scorer=scorer,
            )
            pnl = val_result.net_pnl
            killed = val_result.risk_summary.get("is_killed", False)
            # Score: PnL + survival bonus + PF bonus
            pf = val_result.profit_factor
            score = pnl + (300 if not killed else 0) + (pf * 100 if pf > 1.0 else 0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        window.best_threshold = best_thresh
        scorer.threshold = best_thresh

        # Run validation with best threshold for reporting
        val_result, _ = run_backtest_v2(
            val_df, strategy_cfg, risk_cfg, bt_cfg, scorer=scorer,
        )
        window.val_result = val_result

        # ── Phase 3: Test — true OOS ──────────────────────────────────
        test_result, _ = run_backtest_v2(
            test_df, strategy_cfg, risk_cfg, bt_cfg, scorer=scorer,
        )
        window.test_result = test_result

        # Degradation check: flag if OOS much worse than validation
        val_pnl = val_result.net_pnl if val_result else 0
        test_pnl = test_result.net_pnl if test_result else 0
        if val_pnl > 0 and test_pnl < -val_pnl * 0.5:
            window.degraded = True
            logger.warning(
                f"WF #{window_id}: DEGRADED — val ${val_pnl:.0f} → test ${test_pnl:.0f}"
            )

        logger.info(
            f"WF #{window_id}: train={window.train_trades}tr, "
            f"val=${val_pnl:.0f}, test=${test_pnl:.0f}, "
            f"thresh={best_thresh:.2f}, cv={metadata.get('cv_accuracy', 'N/A')}"
        )

        windows.append(window)
        window_id += 1
        cursor += pd.Timedelta(days=step_days)

    return windows


def aggregate_results(windows: list[WalkForwardWindow]) -> dict:
    """Aggregate walk-forward results — only using TRUE OOS (test) data."""
    test_results = [w.test_result for w in windows if w.test_result is not None]
    if not test_results:
        return {"error": "No valid windows"}

    all_trades = []
    for r in test_results:
        all_trades.extend(r.trades)

    total_pnl = sum(r.net_pnl for r in test_results)
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if (t.pnl - t.fees) > 0)
    max_dd = min(r.max_drawdown for r in test_results)

    # Combined daily PnL for Sharpe
    daily_pnls = pd.concat([r.daily_pnl for r in test_results if not r.daily_pnl.empty])
    sharpe = 0.0
    if len(daily_pnls) > 1 and daily_pnls.std() > 0:
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * np.sqrt(252)

    # Per-window breakdown
    per_window = []
    for w in windows:
        if w.test_result:
            per_window.append({
                "window": w.window_id,
                "test_period": f"{w.test_start.date()} → {w.test_end.date()}",
                "train_trades": w.train_trades,
                "threshold": w.best_threshold,
                "cv_accuracy": w.model_metadata.get("cv_accuracy", 0),
                "test_trades": len(w.test_result.trades),
                "test_pnl": round(w.test_result.net_pnl, 2),
                "test_wr": round(w.test_result.win_rate, 3),
                "test_pf": round(w.test_result.profit_factor, 3),
                "test_dd": round(w.test_result.max_drawdown, 2),
                "degraded": w.degraded,
            })

    # Exit reasons
    exit_reasons: dict[str, int] = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    n_degraded = sum(1 for w in windows if w.degraded)

    return {
        "windows": len(test_results),
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_window": round(total_pnl / len(test_results), 2),
        "win_rate": round(wins / total_trades, 4) if total_trades > 0 else 0,
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "exit_reasons": exit_reasons,
        "degraded_windows": n_degraded,
        "per_window": per_window,
    }
