"""3-part trade quality + risk + sizing decision system.

Instead of binary win/loss, predicts:
1. Expected dollar PnL (trade quality)
2. Expected adverse excursion / uncertainty (risk)
3. Dynamic position size based on quality + risk + context

Labels trained on:
- net_pnl: dollar outcome
- max_adverse: worst drawdown during trade (from peak_profit tracking)
- r_multiple: PnL / risk (stop distance)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

from src.ai.features import AI_FEATURE_COLS


class QualityRiskScorer:
    """3-part decision system: quality + risk + sizing."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.ev_model: GradientBoostingRegressor | None = None
        self.risk_model: GradientBoostingRegressor | None = None
        self.skip_model: GradientBoostingClassifier | None = None
        self.feature_names: list[str] = AI_FEATURE_COLS
        self.ev_threshold: float = 0.0
        self.skip_threshold: float = 0.50
        self.metadata: dict = {}

        # Compatibility with engine_v2
        self.model = True
        self.threshold: float = 0.50

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ev_model = data.get("ev_model")
        self.risk_model = data.get("risk_model")
        self.skip_model = data.get("skip_model")
        self.feature_names = data.get("feature_names", AI_FEATURE_COLS)
        self.ev_threshold = data.get("ev_threshold", 0.0)
        self.skip_threshold = data.get("skip_threshold", 0.50)
        self.metadata = data.get("metadata", {})
        logger.info(f"Loaded QualityRiskScorer: ev_thresh=${self.ev_threshold:.1f}, "
                    f"skip_thresh={self.skip_threshold:.2f}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "ev_model": self.ev_model,
                "risk_model": self.risk_model,
                "skip_model": self.skip_model,
                "feature_names": self.feature_names,
                "ev_threshold": self.ev_threshold,
                "skip_threshold": self.skip_threshold,
                "metadata": self.metadata,
            }, f)
        logger.info(f"Saved QualityRiskScorer → {path}")

    def _get_X(self, features: dict) -> np.ndarray:
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_ev(self, features: dict) -> float:
        """Predict expected dollar PnL."""
        if self.ev_model is None:
            return 0.0
        return float(self.ev_model.predict(self._get_X(features))[0])

    def predict_risk(self, features: dict) -> float:
        """Predict expected adverse excursion (higher = more risk)."""
        if self.risk_model is None:
            return 50.0
        return float(self.risk_model.predict(self._get_X(features))[0])

    def predict_skip(self, features: dict) -> float:
        """Predict probability trade should be skipped (large loser)."""
        if self.skip_model is None:
            return 0.3
        return float(self.skip_model.predict_proba(self._get_X(features))[0][1])

    def should_trade(self, features: dict) -> tuple[bool, float]:
        """Full 3-part decision. Returns (should_take, quality_score)."""
        ev = self.predict_ev(features)
        risk = self.predict_risk(features)
        skip_prob = self.predict_skip(features)

        # Skip if: high probability of large loser
        if skip_prob > self.skip_threshold:
            return False, 0.0

        # Skip if: EV too low
        if ev < self.ev_threshold:
            return False, 0.0

        # Quality score: EV adjusted by risk (higher risk = lower quality)
        # Normalize: EV can be -100 to +300, risk can be 0 to 200
        risk_penalty = max(0, risk - 50) / 100  # Penalty starts above $50 adverse
        quality = ev * (1.0 - risk_penalty * 0.3)  # Risk reduces quality by up to 30%

        # Convert to 0-1 probability-like score
        prob_proxy = 1.0 / (1.0 + np.exp(-quality / 50.0))
        return True, float(prob_proxy)

    def get_size_multiplier(self) -> float:
        """Dynamic sizing — driven primarily by skip probability (most reliable signal).

        Skip model (71% accuracy) is the strongest predictor.
        EV model (R²=-0.13) is weak, used only for extreme cases.
        """
        skip = getattr(self, "_last_skip", 0.3)
        ev = getattr(self, "_last_ev", 0)
        risk = getattr(self, "_last_risk", 50)

        # Base from skip probability (low skip = high quality)
        if skip < 0.15:
            base = 1.5  # Very unlikely to be a large loser
        elif skip < 0.25:
            base = 1.3
        elif skip < 0.35:
            base = 1.1
        else:
            base = 0.8  # Higher chance of large loss

        # EV boost only for very strong predictions
        if ev > 80:
            base *= 1.15
        elif ev < -30:
            base *= 0.85

        # Risk discount for high adverse excursion
        if risk > 120:
            base *= 0.75

        return max(0.6, min(1.6, base))

    def should_trade(self, features: dict) -> tuple[bool, float]:
        """Full 3-part decision with caching for sizing."""
        ev = self.predict_ev(features)
        risk = self.predict_risk(features)
        skip_prob = self.predict_skip(features)

        # Cache for get_size_multiplier
        self._last_ev = ev
        self._last_risk = risk
        self._last_skip = skip_prob

        if skip_prob > self.skip_threshold:
            return False, 0.0
        if ev < self.ev_threshold:
            return False, 0.0

        risk_penalty = max(0, risk - 50) / 100
        quality = ev * (1.0 - risk_penalty * 0.3)
        prob_proxy = 1.0 / (1.0 + np.exp(-quality / 50.0))
        return True, float(prob_proxy)


def train_quality_risk_model(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_path: str | Path = "data/models/quality_risk.pkl",
) -> dict:
    """Train the 3-part quality + risk + skip model."""

    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))
    pnl_col = "net_pnl" if "net_pnl" in merged.columns else "net_pnl_feat"

    feature_cols = [c for c in AI_FEATURE_COLS if c in merged.columns]
    X = merged[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_pnl = merged[pnl_col].values.astype(float)

    n = len(X)
    logger.info(f"Quality/Risk training: {n} samples, {X.shape[1]} features")

    if n < 15:
        return {"error": "insufficient_data", "n_samples": n}

    # ── Model 1: EV prediction (expected dollar PnL) ──────────────────
    ev_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=5, random_state=42)

    n_splits = min(5, max(2, n // 10))
    ev_r2 = 0.0
    if n_splits >= 2:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        ev_r2 = cross_val_score(ev_model, X, y_pnl, cv=cv, scoring="r2").mean()
    ev_model.fit(X, y_pnl)
    logger.info(f"  EV model: R²={ev_r2:.4f}")

    # ── Model 2: Risk prediction (absolute PnL = adverse excursion proxy)
    # Large absolute losses indicate high-risk setups
    y_risk = np.abs(np.minimum(y_pnl, 0))  # Only negative PnL magnitude

    risk_model = GradientBoostingRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        min_samples_leaf=5, random_state=42)

    risk_r2 = 0.0
    if n_splits >= 2:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        risk_r2 = cross_val_score(risk_model, X, y_risk, cv=cv, scoring="r2").mean()
    risk_model.fit(X, y_risk)
    logger.info(f"  Risk model: R²={risk_r2:.4f}")

    # ── Model 3: Skip classifier (large loser = 1, else = 0) ─────────
    # Define "large loser" as bottom 25th percentile of PnL
    large_loss_threshold = np.percentile(y_pnl, 25)
    y_skip = (y_pnl < large_loss_threshold).astype(int)

    skip_model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        min_samples_leaf=5, random_state=42)

    skip_acc = 0.0
    if n_splits >= 2 and len(np.unique(y_skip)) > 1:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        skip_acc = cross_val_score(skip_model, X, y_skip, cv=cv, scoring="accuracy").mean()
    skip_model.fit(X, y_skip)
    logger.info(f"  Skip model: accuracy={skip_acc:.4f}")

    # ── Optimize thresholds ───────────────────────────────────────────
    ev_preds = ev_model.predict(X)
    skip_preds = skip_model.predict_proba(X)[:, 1]

    best_ev_thresh = 0.0
    best_skip_thresh = 0.50
    best_score = float("-inf")

    for ev_t in [-10, 0, 10, 20, 30]:
        for skip_t in [0.35, 0.45, 0.55, 0.65]:
            mask = (ev_preds > ev_t) & (skip_preds < skip_t)
            if mask.sum() < 5:
                continue
            taken_pnl = y_pnl[mask].sum()
            avoided = abs(y_pnl[~mask & (y_pnl < 0)].sum())
            n_taken = mask.sum()
            score = taken_pnl + avoided * 0.3 + min(n_taken * 3, 100)

            if score > best_score:
                best_score = score
                best_ev_thresh = ev_t
                best_skip_thresh = skip_t

    logger.info(f"  Optimal: EV>${best_ev_thresh:.0f}, skip<{best_skip_thresh:.2f}")

    # Feature importance
    importances = ev_model.feature_importances_
    top_features = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:10]

    metadata = {
        "ev_r2": round(ev_r2, 4),
        "risk_r2": round(risk_r2, 4),
        "skip_accuracy": round(skip_acc, 4),
        "ev_threshold": best_ev_thresh,
        "skip_threshold": best_skip_thresh,
        "n_samples": n,
        "top_features": [(name, round(imp, 4)) for name, imp in top_features],
    }

    scorer = QualityRiskScorer()
    scorer.ev_model = ev_model
    scorer.risk_model = risk_model
    scorer.skip_model = skip_model
    scorer.feature_names = feature_cols
    scorer.ev_threshold = best_ev_thresh
    scorer.skip_threshold = best_skip_thresh
    scorer.metadata = metadata
    scorer.save(output_path)

    return metadata
