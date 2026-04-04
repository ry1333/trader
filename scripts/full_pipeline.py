"""Full pipeline v3: multi-instrument walk-forward with per-instrument models.

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

from src.backtest.multi_instrument import load_instruments
from src.backtest.walk_forward import aggregate_results, walk_forward
from src.config import BacktestConfig, load_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data_cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

INSTRUMENT_DATA = {
    "MES": "es_5m_3y_databento.csv",
    "MNQ": "nq_5m_2y_databento.csv",
    # MCL removed — negative OOS, drags combined results
}


def main() -> None:
    settings = load_settings()
    instruments = load_instruments()
    inst_map = {i.symbol: i for i in instruments}

    all_summaries = {}
    combined_pnl = 0.0
    combined_trades = 0
    combined_windows = 0

    for symbol, data_file in INSTRUMENT_DATA.items():
        path = DATA_DIR / data_file
        if not path.exists():
            logger.warning(f"No data for {symbol}")
            continue

        inst = inst_map.get(symbol)
        if not inst:
            continue

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        logger.info(f"{symbol}: {len(df)} bars, {df.timestamp.iloc[0].date()} → {df.timestamp.iloc[-1].date()}")

        # Instrument-specific backtest config
        bt_cfg = BacktestConfig(
            train_window_days=settings.backtest.train_window_days,
            val_window_days=settings.backtest.val_window_days,
            test_window_days=settings.backtest.test_window_days,
            walk_forward_step_days=settings.backtest.walk_forward_step_days,
            cost_per_side_per_contract=inst.cost_per_side,
            slippage_ticks=1,
            tick_size=inst.tick_size,
            tick_value=inst.tick_value,
        )

        windows = walk_forward(
            df, settings.strategy, settings.risk, bt_cfg,
            instrument=symbol,
        )
        summary = aggregate_results(windows)
        all_summaries[symbol] = summary

        n_w = summary.get("windows", 0)
        pnl = summary.get("total_pnl", 0)
        combined_pnl += pnl
        combined_trades += summary.get("total_trades", 0)
        combined_windows = max(combined_windows, n_w)

        logger.info(f"{symbol}: {summary.get('total_trades', 0)} trades, "
                    f"${pnl:.2f}, WR={summary.get('win_rate', 0):.0%}")

    # Print results
    print("\n" + "=" * 80)
    print("MULTI-INSTRUMENT WALK-FORWARD — TRUE OOS (per-window retraining)")
    print("=" * 80)

    for sym, s in all_summaries.items():
        n = s.get("windows", 0)
        if n == 0:
            continue
        print(f"\n--- {sym} ---")
        print(f"  Windows: {n}, Trades: {s.get('total_trades', 0)}")
        print(f"  Total PnL: ${s.get('total_pnl', 0):.2f} (${s.get('total_pnl', 0) / n:.2f}/window)")
        print(f"  Win rate: {s.get('win_rate', 0):.1%}, Sharpe: {s.get('sharpe', 0):.3f}")
        print(f"  Max DD: ${s.get('max_drawdown', 0):.2f}, Degraded: {s.get('degraded_windows', 0)}")
        if s.get("per_window"):
            profitable = sum(1 for w in s["per_window"] if w["test_pnl"] > 0)
            print(f"  Profitable windows: {profitable}/{n} ({profitable / n:.0%})")

    print(f"\n{'=' * 80}")
    print(f"COMBINED: {combined_trades} trades, ${combined_pnl:.2f}")
    if combined_windows > 0:
        print(f"Monthly avg: ${combined_pnl / combined_windows:.2f} ({combined_pnl / combined_windows / 50000 * 100:.1f}%)")
    print(f"Target: $5,000/month (10%)")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"multi_wf_{ts}.json", "w") as f:
        json.dump({"instruments": {k: v for k, v in all_summaries.items()},
                    "combined_pnl": combined_pnl, "combined_trades": combined_trades},
                   f, indent=2, default=str)


if __name__ == "__main__":
    main()
