"""Full pipeline: load 12mo data → train AI → walk-forward validate → report.

Run after fetching data from Databento:
    python scripts/full_pipeline.py
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
from src.backtest.engine_v2 import run_backtest_v2
from src.config import load_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data_cache"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_data() -> pd.DataFrame:
    """Load best available data."""
    # Prefer Databento 12mo
    databento = DATA_DIR / "es_5m_12mo_databento.csv"
    if databento.exists():
        logger.info(f"Loading Databento 12mo data")
        df = pd.read_csv(databento)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # Fallback to yfinance 60d
    yf_file = DATA_DIR / "es_5m_60d.csv"
    if yf_file.exists():
        logger.info("Loading yfinance 60d data (limited)")
        df = pd.read_csv(yf_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    raise FileNotFoundError("No data found. Run fetch_databento.py first.")


def walk_forward_train_test(
    df: pd.DataFrame,
    settings,
    train_pct: float = 0.60,
    val_pct: float = 0.20,
) -> dict:
    """Walk-forward: train on 60%, validate on 20%, test on 20%."""
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} bars | Val: {len(val_df)} bars | Test: {len(test_df)} bars")

    # ── Phase 1: Collect training data ────────────────────────────────
    logger.info("Phase 1: Collecting training data...")
    train_result, train_features = run_backtest_v2(
        train_df, settings.strategy, settings.risk, settings.backtest,
        collect_features=True,
    )
    logger.info(f"Train: {len(train_result.trades)} trades, ${train_result.net_pnl:.2f}")

    if len(train_result.trades) < 20:
        return {"error": "Not enough training trades", "n_trades": len(train_result.trades)}

    # ── Phase 2: Train AI model ───────────────────────────────────────
    logger.info("Phase 2: Training AI model...")
    trades_data = pd.DataFrame([{
        "entry_bar": t.entry_bar, "net_pnl": t.pnl - t.fees,
        "pnl": t.pnl, "fees": t.fees,
    } for t in train_result.trades])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "trade_scorer_v2.pkl"
    metadata = train_and_save(trades_data, train_features, model_path,
                              use_ensemble=len(trades_data) >= 30)

    # ── Phase 3: Validate (tune threshold if needed) ──────────────────
    logger.info("Phase 3: Validation period...")
    scorer = TradeScorer(model_path)

    val_base, _ = run_backtest_v2(val_df, settings.strategy, settings.risk, settings.backtest)
    val_ai, _ = run_backtest_v2(val_df, settings.strategy, settings.risk, settings.backtest, scorer=scorer)

    logger.info(f"Val base: {val_base.summary()['total_trades']} trades, ${val_base.net_pnl:.2f}")
    logger.info(f"Val AI:   {val_ai.summary()['total_trades']} trades, ${val_ai.net_pnl:.2f}")

    # ── Phase 4: Out-of-sample test ───────────────────────────────────
    logger.info("Phase 4: Out-of-sample test...")
    test_base, _ = run_backtest_v2(test_df, settings.strategy, settings.risk, settings.backtest)
    test_ai, _ = run_backtest_v2(test_df, settings.strategy, settings.risk, settings.backtest, scorer=scorer)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)

    headers = ["Metric", "Val Base", "Val AI", "Test Base", "Test AI"]
    rows = []
    for key, fmt in [
        ("total_trades", lambda v: str(v)),
        ("net_pnl", lambda v: f"${v:.2f}"),
        ("win_rate", lambda v: f"{v:.1%}"),
        ("profit_factor", lambda v: f"{v:.3f}"),
        ("max_drawdown", lambda v: f"${v:.2f}"),
        ("sharpe", lambda v: f"{v:.3f}"),
        ("is_killed", lambda v: str(v)),
    ]:
        vb = val_base.summary()[key]
        va = val_ai.summary()[key]
        tb = test_base.summary()[key]
        ta = test_ai.summary()[key]
        rows.append([key, fmt(vb), fmt(va), fmt(tb), fmt(ta)])

    # Print table
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(5)]
    header_str = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(row, widths)))

    # Topstep check
    test_s = test_ai.summary()
    print(f"\n--- TOPSTEP COMBINE CHECK (Test AI) ---")
    dd = abs(test_s["max_drawdown"])
    print(f"Max drawdown: ${dd:.2f} (limit: $2,000)")
    print(f"Account killed: {test_s['is_killed']}")
    if test_s["total_pnl"] > 0:
        consistency = test_s["best_day_pnl"] / test_s["total_pnl"]
        print(f"Consistency: {consistency:.1%} (must be < 50%)")
    passed = dd < 1600 and not test_s["is_killed"] and test_s["total_pnl"] > 0
    print(f"VERDICT: {'READY FOR COMBINE' if passed else 'NEEDS MORE TUNING'}")

    results = {
        "model": metadata,
        "validation": {"base": val_base.summary(), "ai": val_ai.summary()},
        "test": {"base": test_base.summary(), "ai": test_ai.summary()},
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"walk_forward_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    settings = load_settings()
    df = load_data()
    logger.info(f"Loaded {len(df)} bars: {df.timestamp.iloc[0]} → {df.timestamp.iloc[-1]}")
    walk_forward_train_test(df, settings)


if __name__ == "__main__":
    main()
