"""AI model trainer — trains ensemble of 3 specialized models.

Momentum model: ret_*, macd, rsi, momentum features
Mean-reversion model: zscore_*, vwap, price_position features
Volatility model: atr_*, rvol_*, vol_ratio features

Each sub-model is a GBT trained on its feature subset.
Threshold optimized for profit on validation data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.ai.features import (
    AI_FEATURE_COLS,
    MEAN_REVERSION_FEATURES,
    MOMENTUM_FEATURES,
    VOLATILITY_FEATURES,
)
from src.ai.model import EnsembleScorer, TradeScorer


def generate_training_data(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare training data from backtest results."""
    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))
    pnl_col = "net_pnl" if "net_pnl" in merged.columns else "net_pnl_feat"
    merged["label"] = (merged[pnl_col] > 0).astype(int)

    feature_cols = [c for c in AI_FEATURE_COLS if c in merged.columns]
    X = merged[feature_cols].values.astype(float)
    y = merged["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features, "
                f"{y.sum()} wins ({y.mean():.1%} win rate)")
    return X, y, feature_cols


def _train_single_gbt(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[GradientBoostingClassifier, float]:
    """Train a single GBT and return (model, cv_accuracy)."""
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )

    n_splits = min(5, max(2, len(X) // 10))
    cv_acc = 0.0
    if n_splits >= 2 and len(np.unique(y)) > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_acc = scores.mean()

    model.fit(X, y)
    return model, cv_acc


def _select_features(
    X_full: np.ndarray,
    y: np.ndarray,
    all_feature_names: list[str],
    target_features: list[str],
    max_features: int = 12,
) -> tuple[np.ndarray, list[str]]:
    """Select relevant features from a target subset that exist in the data."""
    # Find which target features exist in the full feature set
    available = []
    indices = []
    for feat in target_features:
        if feat in all_feature_names:
            available.append(feat)
            indices.append(all_feature_names.index(feat))

    if not available:
        return X_full[:, :max_features], all_feature_names[:max_features]

    X_sub = X_full[:, indices]

    # If more than max_features, use mutual information to rank
    if len(available) > max_features:
        mi = mutual_info_classif(X_sub, y, random_state=42)
        top_idx = np.argsort(mi)[-max_features:]
        X_sub = X_sub[:, top_idx]
        available = [available[i] for i in top_idx]

    return X_sub, available


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    pnls: np.ndarray,
) -> tuple[EnsembleScorer, dict]:
    """Train 3-model ensemble on feature subsets."""

    sub_models = {}
    cv_accs = {}

    for name, target_feats in [
        ("momentum", MOMENTUM_FEATURES),
        ("mean_reversion", MEAN_REVERSION_FEATURES),
        ("volatility", VOLATILITY_FEATURES),
    ]:
        X_sub, feat_names = _select_features(X, y, feature_names, target_feats)
        if X_sub.shape[1] < 3:
            logger.warning(f"Ensemble {name}: only {X_sub.shape[1]} features, using full set")
            X_sub, feat_names = X[:, :12], feature_names[:12]

        model, cv_acc = _train_single_gbt(X_sub, y, feat_names)
        cv_accs[name] = cv_acc

        scorer = TradeScorer()
        scorer.model = model
        scorer.feature_names = feat_names
        sub_models[name] = scorer

        logger.info(f"  {name}: {len(feat_names)} features, CV={cv_acc:.3f}")

    # Build ensemble
    ensemble = EnsembleScorer()
    ensemble.momentum_model = sub_models["momentum"]
    ensemble.mr_model = sub_models["mean_reversion"]
    ensemble.vol_model = sub_models["volatility"]

    avg_cv = np.mean(list(cv_accs.values()))

    # Also train a single model for comparison/fallback
    single_model, single_cv = _train_single_gbt(X, y, feature_names)

    # Feature importance from single model
    importances = single_model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

    metadata = {
        "cv_accuracy": round(avg_cv, 4),
        "single_cv": round(single_cv, 4),
        "n_samples": len(X),
        "n_features": len(feature_names),
        "win_rate": float(y.mean()),
        "top_features": [(n, round(v, 4)) for n, v in top_features],
        "sub_model_cv": {k: round(v, 4) for k, v in cv_accs.items()},
    }

    return ensemble, metadata


def find_optimal_threshold(
    scorer,
    X: np.ndarray,
    y: np.ndarray,
    pnls: np.ndarray,
    feature_names: list[str],
) -> float:
    """Find profit-maximizing threshold."""
    # Build feature dicts for scorer
    probas = []
    for i in range(len(X)):
        features = {name: float(X[i, j]) for j, name in enumerate(feature_names)}
        _, prob = scorer.should_trade(features)
        probas.append(prob)
    probas = np.array(probas)

    best_threshold = 0.50
    best_score = float("-inf")

    for threshold in np.arange(0.35, 0.65, 0.05):
        mask = probas >= threshold
        if mask.sum() < 5:
            continue

        taken_pnl = pnls[mask].sum()
        avoided_losses = abs(pnls[~mask & (pnls < 0)].sum())
        n_trades = mask.sum()
        win_rate = y[mask].mean() if mask.sum() > 0 else 0

        score = taken_pnl + avoided_losses * 0.5

        logger.debug(
            f"  threshold={threshold:.2f}: {n_trades} trades, "
            f"pnl=${taken_pnl:.0f}, avoided=${avoided_losses:.0f}, "
            f"wr={win_rate:.1%}, score={score:.0f}"
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
    """Full training pipeline: data → ensemble → threshold → save."""
    X, y, feature_names = generate_training_data(trades_df, features_df)

    if len(X) < 10:
        logger.error(f"Only {len(X)} trades — need at least 10 to train")
        return {"error": "insufficient_data", "n_samples": len(X)}

    # Get PnL values for threshold optimization
    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))
    pnl_col = "net_pnl" if "net_pnl" in merged.columns else "net_pnl_feat"
    pnls = merged[pnl_col].values

    if use_ensemble and len(X) >= 30:
        scorer, metadata = train_ensemble(X, y, feature_names, pnls)
        threshold = find_optimal_threshold(scorer, X, y, pnls, feature_names)
        scorer.threshold = threshold
        for sub in [scorer.momentum_model, scorer.mr_model, scorer.vol_model]:
            if sub:
                sub.threshold = threshold
        scorer.metadata = metadata
        scorer.save(output_path)
    else:
        # Fallback to single model
        model, cv_acc = _train_single_gbt(X, y, feature_names)
        importances = model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        metadata = {
            "cv_accuracy": round(cv_acc, 4),
            "n_samples": len(X),
            "n_features": len(feature_names),
            "win_rate": float(y.mean()),
            "top_features": [(n, round(v, 4)) for n, v in top_features],
        }
        scorer = TradeScorer()
        scorer.model = model
        scorer.feature_names = feature_names
        scorer.metadata = metadata

        # Simple threshold
        probas = model.predict_proba(X)[:, 1]
        best_thresh = 0.50
        best_score = float("-inf")
        for t in np.arange(0.35, 0.65, 0.05):
            mask = probas >= t
            if mask.sum() < 5:
                continue
            score = pnls[mask].sum() + abs(pnls[~mask & (pnls < 0)].sum()) * 0.5
            if score > best_score:
                best_score = score
                best_thresh = t
        scorer.threshold = best_thresh
        scorer.save(output_path)

    metadata["threshold"] = scorer.threshold if hasattr(scorer, 'threshold') else 0.50
    return metadata
