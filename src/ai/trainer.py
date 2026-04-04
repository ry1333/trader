"""AI model trainer — trains on backtest results to predict winners.

Same approach as Morgan bot:
1. Run backtest → get trade log with features + P&L
2. Label: profitable trade = 1, losing trade = 0
3. Train GradientBoostingClassifier
4. Optimize threshold for profit (not just accuracy)
5. Save model for live scoring
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.ai.features import AI_FEATURE_COLS
from src.ai.model import TradeScorer


def generate_training_data(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare training data from backtest results.

    Args:
        trades_df: DataFrame with columns [entry_bar, pnl, fees, ...]
        features_df: DataFrame with AI features per trade, indexed by entry_bar

    Returns:
        X (features), y (labels), feature_names
    """
    # Merge trades with features (suffix to avoid column collisions)
    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))

    # Label: net profitable = 1, else 0
    pnl_col = "net_pnl" if "net_pnl" in merged.columns else "net_pnl_feat"
    merged["label"] = (merged[pnl_col] > 0).astype(int)

    feature_cols = [c for c in AI_FEATURE_COLS if c in merged.columns]
    X = merged[feature_cols].values.astype(float)
    y = merged["label"].values

    # Clean NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features, "
                f"{y.sum()} wins ({y.mean():.1%} win rate)")

    return X, y, feature_cols


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    use_ensemble: bool = True,
) -> tuple[object, dict]:
    """Train the ML model.

    Returns (model, metadata_dict).
    """
    if len(X) < 20:
        logger.warning(f"Only {len(X)} samples — model may overfit. Need 50+ ideally.")

    if use_ensemble and len(X) >= 30:
        # Ensemble of 3 models (like Morgan bot's advanced trainer)
        gbt = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42,
        )
        et = ExtraTreesClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42,
        )
        model = VotingClassifier(
            estimators=[("gbt", gbt), ("rf", rf), ("et", et)],
            voting="soft",
        )
    else:
        # Single GBT for small datasets
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42,
        )

    # Cross-validation
    n_splits = min(5, max(2, len(X) // 10))
    if n_splits >= 2 and len(np.unique(y)) > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_accuracy = scores.mean()
        logger.info(f"CV Accuracy: {cv_accuracy:.3f} (+/- {scores.std():.3f})")
    else:
        cv_accuracy = 0.0
        logger.warning("Not enough data for cross-validation")

    # Train on full dataset
    model.fit(X, y)

    # Feature importance (from GBT)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # Ensemble: use GBT's importances
        gbt_model = model.named_estimators_.get("gbt")
        importances = gbt_model.feature_importances_ if gbt_model else np.zeros(len(feature_names))
    else:
        importances = np.zeros(len(feature_names))

    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:10]

    metadata = {
        "cv_accuracy": round(cv_accuracy, 4),
        "n_samples": len(X),
        "n_features": len(feature_names),
        "win_rate": float(y.mean()),
        "top_features": [(name, round(imp, 4)) for name, imp in top_features],
    }

    logger.info(f"Top features: {[f'{n}={v:.3f}' for n, v in top_features[:5]]}")

    return model, metadata


def find_optimal_threshold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    pnls: np.ndarray,
) -> float:
    """Find the profit-maximizing threshold (not just accuracy).

    Same approach as Morgan bot: optimize for total P&L, not just win rate.
    """
    probas = model.predict_proba(X)[:, 1]

    best_threshold = 0.50
    best_score = float("-inf")

    for threshold in np.arange(0.30, 0.80, 0.05):
        mask = probas >= threshold
        if mask.sum() < 5:  # Need minimum trades
            continue

        taken_pnl = pnls[mask].sum()
        avoided_losses = abs(pnls[~mask & (pnls < 0)].sum())
        n_trades = mask.sum()
        win_rate = y[mask].mean() if mask.sum() > 0 else 0

        # Score: profit from taken trades + credit for avoided losses
        score = taken_pnl + avoided_losses * 0.5

        logger.debug(
            f"  threshold={threshold:.2f}: {n_trades} trades, "
            f"pnl=${taken_pnl:.0f}, avoided_loss=${avoided_losses:.0f}, "
            f"win_rate={win_rate:.1%}, score={score:.0f}"
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} (score={best_score:.0f})")
    return best_threshold


def train_and_save(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_path: str | Path = "data/models/trade_scorer.pkl",
    use_ensemble: bool = True,
) -> dict:
    """Full training pipeline: data → model → threshold → save.

    Returns metadata dict with performance metrics.
    """
    X, y, feature_names = generate_training_data(trades_df, features_df)

    if len(X) < 10:
        logger.error(f"Only {len(X)} trades — need at least 10 to train")
        return {"error": "insufficient_data", "n_samples": len(X)}

    model, metadata = train_model(X, y, feature_names, use_ensemble)

    # Optimize threshold using P&L
    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))
    pnl_col = "net_pnl" if "net_pnl" in merged.columns else "net_pnl_feat"
    pnls = merged[pnl_col].values
    threshold = find_optimal_threshold(model, X, y, pnls)

    # Save
    scorer = TradeScorer()
    scorer.model = model
    scorer.threshold = threshold
    scorer.feature_names = feature_names
    scorer.metadata = metadata
    scorer.save(output_path)

    metadata["threshold"] = threshold
    return metadata
