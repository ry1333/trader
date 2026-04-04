"""Expected Value model — predicts dollar P&L per trade, not just win/loss.

Instead of P(win) → binary classification:
  EV = P(win) * avg_win - P(loss) * avg_loss

Train a regressor on actual P&L outcomes. Only take trades where predicted EV > threshold.
This captures both probability AND magnitude of wins/losses.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.ai.features import AI_FEATURE_COLS


class EVScorer:
    """Predicts expected dollar value per trade."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model: GradientBoostingRegressor | None = None
        self.ev_threshold: float = 0.0  # Min predicted EV to take trade
        self.feature_names: list[str] = AI_FEATURE_COLS
        self.metadata: dict = {}
        # Compatibility with engine_v2 interface
        self.threshold: float = 0.50

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.ev_threshold = data.get("ev_threshold", 0.0)
        self.feature_names = data.get("feature_names", AI_FEATURE_COLS)
        self.metadata = data.get("metadata", {})
        logger.info(f"Loaded EVScorer: ev_threshold=${self.ev_threshold:.2f}, "
                    f"features={len(self.feature_names)}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "ev_threshold": self.ev_threshold,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            }, f)
        logger.info(f"Saved EVScorer → {path}")

    def predict_ev(self, features: dict) -> float:
        """Predict expected dollar P&L for a trade."""
        if self.model is None:
            return 0.0

        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return float(self.model.predict(X)[0])

    def should_trade(self, features: dict) -> tuple[bool, float]:
        """Interface compatible with TradeScorer/EnsembleScorer.

        Returns (should_take, probability_proxy).
        Also stores last_ev for position sizing.
        """
        ev = self.predict_ev(features)
        self.last_ev = ev  # Store for EV-based sizing
        prob_proxy = 1.0 / (1.0 + np.exp(-ev / 50.0))
        return ev > self.ev_threshold, float(prob_proxy)

    def get_size_multiplier(self) -> float:
        """Return position size multiplier based on last predicted EV.

        Top decile EV: 1.5x size
        High EV: 1.2x
        Medium EV: 1.0x
        Low accepted: 0.7x
        """
        ev = getattr(self, "last_ev", 0.0)
        if ev > 150:
            return 1.5
        elif ev > 80:
            return 1.2
        elif ev > 40:
            return 1.0
        else:
            return 0.7


def train_ev_model(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_path: str | Path = "data/models/ev_scorer.pkl",
) -> dict:
    """Train expected value regression model.

    Target: actual net P&L per trade (continuous, not binary).
    """
    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))
    pnl_col = "net_pnl" if "net_pnl" in merged.columns else "net_pnl_feat"

    feature_cols = [c for c in AI_FEATURE_COLS if c in merged.columns]
    X = merged[feature_cols].values.astype(float)
    y = merged[pnl_col].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"EV training: {len(X)} samples, {X.shape[1]} features, "
                f"avg PnL=${y.mean():.2f}, std=${y.std():.2f}")

    # Train GBT regressor
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )

    # Cross-validation (R² score)
    n_splits = min(5, max(2, len(X) // 10))
    cv_r2 = 0.0
    if n_splits >= 2:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        cv_r2 = scores.mean()
        logger.info(f"EV model CV R²: {cv_r2:.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)

    # Feature importance
    importances = model.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:10]

    # Find optimal EV threshold
    predictions = model.predict(X)
    best_threshold = 0.0
    best_profit = float("-inf")

    for thresh in np.arange(-20, 50, 5):
        mask = predictions > thresh
        if mask.sum() < 5:
            continue
        taken_pnl = y[mask].sum()
        n_taken = mask.sum()
        avg_pnl = y[mask].mean()
        avoided = abs(y[~mask & (y < 0)].sum())

        score = taken_pnl + avoided * 0.3
        if score > best_profit:
            best_profit = score
            best_threshold = float(thresh)

    logger.info(f"Optimal EV threshold: ${best_threshold:.2f}")

    # Save
    scorer = EVScorer()
    scorer.model = model
    scorer.ev_threshold = best_threshold
    scorer.feature_names = feature_cols
    scorer.metadata = {
        "cv_r2": round(cv_r2, 4),
        "n_samples": len(X),
        "avg_pnl": round(float(y.mean()), 2),
        "ev_threshold": best_threshold,
        "top_features": [(n, round(v, 4)) for n, v in top_features],
    }
    scorer.save(output_path)

    return scorer.metadata
