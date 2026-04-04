"""Run backtest on historical data.

Usage:
    # With Topstep API data:
    TOPSTEP_USERNAME=you TOPSTEP_API_KEY=xxx python run_backtest.py

    # With local CSV:
    python run_backtest.py --csv data/es_5m.csv

    # With demo API:
    python run_backtest.py --demo
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from src.backtest.engine import run_backtest
from src.backtest.walk_forward import aggregate_results, walk_forward
from src.config import load_settings
from src.data.loader import load_bars, load_bars_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="RSIH Backtest Runner")
    parser.add_argument("--csv", type=str, help="Path to local CSV with OHLCV data")
    parser.add_argument("--contract", type=str, default="CON.F.US.EP.M25",
                        help="Topstep contract ID (default: ES June 2025)")
    parser.add_argument("--start", type=str, default="2024-06-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-03-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--demo", action="store_true",
                        help="Use demo API environment")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation instead of single pass")
    parser.add_argument("--balance", type=float, default=50_000.0,
                        help="Starting balance")
    args = parser.parse_args()

    settings = load_settings()

    # Load data
    if args.csv:
        logger.info(f"Loading bars from CSV: {args.csv}")
        df = load_bars_csv(args.csv)
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        logger.info(f"Fetching bars from API: {args.contract} {args.start} → {args.end}")
        df = load_bars(settings, args.contract, start, end)

    logger.info(f"Loaded {len(df)} bars, {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    if args.walk_forward:
        # Walk-forward validation
        logger.info("Running walk-forward validation...")
        windows = walk_forward(
            df, settings.strategy, settings.risk, settings.backtest, args.balance
        )
        summary = aggregate_results(windows)
        logger.info("Walk-Forward Results:")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
    else:
        # Single pass backtest
        logger.info("Running single-pass backtest...")
        result = run_backtest(
            df, settings.strategy, settings.risk, settings.backtest, args.balance
        )
        summary = result.summary()
        logger.info("Backtest Results:")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")

        # Trade log
        if result.trades:
            logger.info(f"\nSample trades (first 10):")
            for t in result.trades[:10]:
                net = t.pnl - t.fees
                logger.info(
                    f"  {'LONG' if t.direction == 1 else 'SHORT'} "
                    f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                    f"pnl=${net:.2f} reason={t.exit_reason}"
                )

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"backtest_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
