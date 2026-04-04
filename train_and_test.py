"""Full pipeline: tune parameters → collect training data → train AI → test with AI.

Steps:
1. Load real ES data
2. Split: first 70% for training data collection, last 30% for testing
3. Run backtest on training period (collect features + outcomes)
4. Train AI model on collected data
5. Run AI-filtered backtest on test period
6. Compare: base strategy vs AI-filtered strategy
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.ai.model import TradeScorer
from src.ai.trainer import train_and_save
from src.backtest.engine import run_backtest
from src.backtest.engine_ai import run_backtest_ai
from src.config import load_settings


def main() -> None:
    settings = load_settings()

    # Load real ES data
    df = pd.read_csv("data_cache/es_5m_60d.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Loaded {len(df)} bars: {df.timestamp.iloc[0]} → {df.timestamp.iloc[-1]}")

    # ── Split data: 70% train / 30% test ──────────────────────────────
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"Train: {len(train_df)} bars | Test: {len(test_df)} bars")

    # ── Step 1: Run base backtest on training data (collect features) ─
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting training data from base strategy...")
    logger.info("=" * 60)

    train_result, features_df = run_backtest_ai(
        train_df, settings.strategy, settings.risk, settings.backtest,
        scorer=None,  # No AI filter — take all signals
        collect_features=True,
    )

    logger.info(f"Training period: {len(train_result.trades)} trades, "
                f"net P&L: ${train_result.net_pnl:.2f}")

    if features_df.empty or len(train_result.trades) < 10:
        logger.error("Not enough trades to train AI. Adjusting parameters...")
        return

    # Show training data stats
    n_winners = features_df["was_winner"].sum()
    n_total = len(features_df[features_df["net_pnl"] != 0])
    logger.info(f"Training samples: {n_total} trades ({n_winners} winners, "
                f"{n_total - n_winners} losers)")

    # ── Step 2: Train AI model ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Training AI model...")
    logger.info("=" * 60)

    # Prepare DataFrames for trainer
    trades_data = []
    for t in train_result.trades:
        trades_data.append({
            "entry_bar": t.entry_bar,
            "pnl": t.pnl,
            "fees": t.fees,
            "net_pnl": t.pnl - t.fees,
            "direction": t.direction,
            "exit_reason": t.exit_reason,
        })
    trades_df = pd.DataFrame(trades_data)

    model_path = Path("data/models/trade_scorer.pkl")
    metadata = train_and_save(
        trades_df=trades_df,
        features_df=features_df,
        output_path=model_path,
        use_ensemble=len(trades_df) >= 30,
    )

    logger.info(f"Model trained: {metadata}")

    # ── Step 3: Test base strategy on test period ─────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Base strategy on test data (no AI)...")
    logger.info("=" * 60)

    base_result = run_backtest(
        test_df, settings.strategy, settings.risk, settings.backtest,
    )
    base_summary = base_result.summary()
    logger.info("Base results:")
    for k, v in base_summary.items():
        logger.info(f"  {k}: {v}")

    # ── Step 4: AI-filtered strategy on test period ───────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: AI-filtered strategy on test data...")
    logger.info("=" * 60)

    scorer = TradeScorer(model_path)
    ai_result, ai_features = run_backtest_ai(
        test_df, settings.strategy, settings.risk, settings.backtest,
        scorer=scorer,
        collect_features=True,
    )
    ai_summary = ai_result.summary()
    logger.info("AI-filtered results:")
    for k, v in ai_summary.items():
        logger.info(f"  {k}: {v}")

    # ── Comparison ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("COMPARISON: Base vs AI-Filtered")
    logger.info("=" * 60)

    comparison = {
        "metric": ["Trades", "Net P&L", "Win Rate", "Profit Factor",
                    "Max Drawdown", "Sharpe", "Killed"],
        "base": [
            base_summary["total_trades"],
            f"${base_summary['net_pnl']:.2f}",
            f"{base_summary['win_rate']:.1%}",
            f"{base_summary['profit_factor']:.3f}",
            f"${base_summary['max_drawdown']:.2f}",
            f"{base_summary['sharpe']:.3f}",
            base_summary.get("is_killed", False),
        ],
        "ai_filtered": [
            ai_summary["total_trades"],
            f"${ai_summary['net_pnl']:.2f}",
            f"{ai_summary['win_rate']:.1%}",
            f"{ai_summary['profit_factor']:.3f}",
            f"${ai_summary['max_drawdown']:.2f}",
            f"{ai_summary['sharpe']:.3f}",
            ai_summary.get("is_killed", False),
        ],
    }

    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))

    # Show AI rejection stats
    if not ai_features.empty:
        total_signals = len(ai_features)
        approved = ai_features["ai_approved"].sum()
        rejected = total_signals - approved
        logger.info(f"\nAI filter: {approved} approved, {rejected} rejected "
                    f"out of {total_signals} signals ({rejected/total_signals:.0%} filtered)")

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"ai_comparison_{ts}.json", "w") as f:
        json.dump({"base": base_summary, "ai": ai_summary, "model": metadata}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
