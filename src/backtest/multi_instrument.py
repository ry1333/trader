"""Multi-instrument backtester — runs strategy across ES, NQ, CL simultaneously.

Key improvement: shared risk budget across instruments but independent signals.
This multiplies trade frequency while keeping total risk controlled.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.model import TradeScorer
from src.backtest.engine_v2 import run_backtest_v2
from src.config import BacktestConfig, RiskConfig, StrategyConfig, load_settings


@dataclass
class InstrumentConfig:
    name: str
    symbol: str
    tick_size: float
    tick_value: float
    cost_per_side: float
    max_position: int
    data_file: str


def load_instruments(path: Path | None = None) -> list[InstrumentConfig]:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "instruments.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    instruments = []
    for symbol, cfg in raw.items():
        instruments.append(InstrumentConfig(
            name=cfg["name"], symbol=symbol,
            tick_size=cfg["tick_size"], tick_value=cfg["tick_value"],
            cost_per_side=cfg["cost_per_side"],
            max_position=cfg["max_position"],
            data_file=cfg["data_file"],
        ))
    return instruments


def run_multi_instrument(
    instruments: list[InstrumentConfig],
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    scorer: TradeScorer | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    starting_balance: float = 50_000.0,
    immortal: bool = False,
) -> dict:
    """Run backtest across multiple instruments and aggregate results."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data_cache"

    all_results = {}
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    all_daily_pnls = []
    max_dd = 0.0
    any_killed = False

    for inst in instruments:
        data_path = data_dir / inst.data_file
        if not data_path.exists():
            logger.warning(f"No data for {inst.symbol}: {data_path}")
            continue

        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if start_date:
            df = df[df["timestamp"] >= start_date]
        if end_date:
            df = df[df["timestamp"] < end_date]
        df = df.reset_index(drop=True)

        if len(df) < 100:
            logger.warning(f"{inst.symbol}: only {len(df)} bars, skipping")
            continue

        # Create instrument-specific backtest config
        bt_cfg = BacktestConfig(
            train_window_days=120, val_window_days=30,
            test_window_days=30, walk_forward_step_days=30,
            cost_per_side_per_contract=inst.cost_per_side,
            slippage_ticks=1,
            tick_size=inst.tick_size,
            tick_value=inst.tick_value,
        )

        # Scale risk per instrument (divide budget across instruments)
        n_inst = len(instruments)
        inst_risk = RiskConfig(
            max_daily_loss=risk_cfg.max_daily_loss / n_inst,
            max_total_loss=risk_cfg.max_total_loss,  # Shared total limit
            max_position_size=inst.max_position,
            risk_per_trade_pct=risk_cfg.risk_per_trade_pct,
            flatten_time_ct=risk_cfg.flatten_time_ct,
            session_start_ct=risk_cfg.session_start_ct,
            consistency_target=risk_cfg.consistency_target,
        )

        result, _ = run_backtest_v2(
            df, strategy_cfg, inst_risk, bt_cfg,
            scorer=scorer, starting_balance=starting_balance,
            immortal=immortal,
        )

        s = result.summary()
        wins = sum(1 for t in result.trades if (t.pnl - t.fees) > 0)
        total_pnl += result.net_pnl
        total_trades += len(result.trades)
        total_wins += wins
        if result.max_drawdown < max_dd:
            max_dd = result.max_drawdown
        if s["is_killed"]:
            any_killed = True

        if not result.daily_pnl.empty:
            all_daily_pnls.append(result.daily_pnl)

        all_results[inst.symbol] = {
            "trades": len(result.trades),
            "pnl": round(result.net_pnl, 2),
            "win_rate": round(result.win_rate, 3),
            "profit_factor": round(result.profit_factor, 3),
            "max_drawdown": round(result.max_drawdown, 2),
        }

        logger.info(
            f"{inst.symbol}: {len(result.trades)} trades, "
            f"${result.net_pnl:.2f}, WR={result.win_rate:.0%}, PF={result.profit_factor:.2f}"
        )

    # Combined daily PnL for Sharpe
    combined_sharpe = 0.0
    if all_daily_pnls:
        combined = pd.concat(all_daily_pnls).groupby(level=0).sum()
        if len(combined) > 1 and combined.std() > 0:
            combined_sharpe = (combined.mean() / combined.std()) * np.sqrt(252)

    summary = {
        "instruments": all_results,
        "combined": {
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(total_wins / total_trades, 3) if total_trades > 0 else 0,
            "max_drawdown": round(max_dd, 2),
            "sharpe": round(combined_sharpe, 3),
            "any_killed": any_killed,
        },
    }

    return summary
