"""Full pipeline v3: walk-forward retraining with per-window OOS reporting.

Run:
    PYTHONPATH=. .venv/bin/python scripts/full_pipeline.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.walk_forward import aggregate_results, walk_forward
from src.config import load_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data_cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_data(instrument: str = "MES") -> pd.DataFrame:
    """Load best available data for an instrument."""
    file_map = {
        "MES": ["es_5m_3y_databento.csv", "es_5m_12mo_databento.csv", "es_5m_60d.csv"],
        "MNQ": ["nq_5m_2y_databento.csv"],
        "MCL": ["cl_5m_2y_databento.csv"],
    }
    for fname in file_map.get(instrument, []):
        path = DATA_DIR / fname
        if path.exists():
            logger.info(f"Loading {instrument} data from {fname}")
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    raise FileNotFoundError(f"No data for {instrument}")


def main() -> None:
    settings = load_settings()

    # Run walk-forward on ES (primary instrument)
    instrument = "MES"
    df = load_data(instrument)
    logger.info(f"{instrument}: {len(df)} bars, {df.timestamp.iloc[0].date()} → {df.timestamp.iloc[-1].date()}")

    logger.info("Running walk-forward with rolling retraining...")
    windows = walk_forward(
        df,
        settings.strategy,
        settings.risk,
        settings.backtest,
        instrument=instrument,
    )

    summary = aggregate_results(windows)

    # Print results
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS — TRUE OUT-OF-SAMPLE (per-window model retraining)")
    print("=" * 80)

    if "per_window" in summary:
        print(f"\n{'Win':>4} {'Test Period':<26} {'Tr':>4} {'Thr':>5} {'CV':>5} "
              f"{'Trades':>6} {'PnL':>9} {'WR':>6} {'PF':>6} {'DD':>9} {'Deg':>4}")
        print("-" * 90)
        for w in summary["per_window"]:
            deg = "!!" if w["degraded"] else ""
            print(f"#{w['window']:>3} {w['test_period']:<26} {w['train_trades']:>4} "
                  f"{w['threshold']:>5.2f} {w['cv_accuracy']:>5.3f} "
                  f"{w['test_trades']:>6} ${w['test_pnl']:>8.0f} "
                  f"{w['test_wr']:>5.0%} {w['test_pf']:>6.2f} "
                  f"${w['test_dd']:>8.0f} {deg:>4}")

    print(f"\n--- AGGREGATE (OOS only) ---")
    print(f"Windows: {summary.get('windows', 0)}")
    print(f"Total trades: {summary.get('total_trades', 0)}")
    print(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
    print(f"Avg PnL/window: ${summary.get('avg_pnl_per_window', 0):.2f}")
    print(f"Win rate: {summary.get('win_rate', 0):.1%}")
    print(f"Max drawdown: ${summary.get('max_drawdown', 0):.2f}")
    print(f"Sharpe: {summary.get('sharpe', 0):.3f}")
    print(f"Degraded windows: {summary.get('degraded_windows', 0)}")

    if summary.get("exit_reasons"):
        print(f"\nExit reasons: {summary['exit_reasons']}")

    # Topstep check
    n_windows = summary.get("windows", 0)
    if n_windows > 0:
        monthly_avg = summary["total_pnl"] / n_windows
        dd = abs(summary.get("max_drawdown", 0))
        print(f"\n--- TOPSTEP READINESS ---")
        print(f"Avg monthly PnL: ${monthly_avg:.2f} ({monthly_avg / 50000 * 100:.1f}%)")
        print(f"Max drawdown: ${dd:.2f} (limit $2,000)")
        print(f"Target: $5,000/month (10%)")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"wf_v3_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to results/wf_v3_{ts}.json")


if __name__ == "__main__":
    main()
