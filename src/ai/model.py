"""AI trade scorer — predicts win probability for each setup.

Two modes:
- TradeScorer: single model on all features
- EnsembleScorer: 3 specialized models (momentum, mean-reversion, volatility)
  with 2-of-3 agreement voting
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger

from src.ai.features import (
    AI_FEATURE_COLS,
    MEAN_REVERSION_FEATURES,
    MOMENTUM_FEATURES,
    VOLATILITY_FEATURES,
)


class TradeScorer:
    """Scores potential trades with win probability."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model = None
        self.threshold: float = 0.50
        self.feature_names: list[str] = AI_FEATURE_COLS
        self.metadata: dict = {}

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.threshold = data.get("threshold", 0.50)
        self.feature_names = data.get("feature_names", AI_FEATURE_COLS)
        self.metadata = data.get("metadata", {})
        logger.info(
            f"Loaded TradeScorer: threshold={self.threshold:.2f}, "
            f"features={len(self.feature_names)}, "
            f"cv_accuracy={self.metadata.get('cv_accuracy', 'N/A')}"
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.threshold,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            }, f)
        logger.info(f"Saved TradeScorer → {path}")

    def predict_proba(self, features: dict) -> float:
        """Predict win probability. No post-hoc adjustments."""
        if self.model is None:
            return 0.5

        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self.model.predict_proba(X)[0]
        return float(proba[1])

    def should_trade(self, features: dict) -> tuple[bool, float]:
        prob = self.predict_proba(features)
        return prob >= self.threshold, prob


class EnsembleScorer:
    """3-model ensemble with agreement voting.

    Momentum, mean-reversion, and volatility models each trained
    on their own feature subset. Trade only when 2 of 3 agree.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.momentum_model: TradeScorer | None = None
        self.mr_model: TradeScorer | None = None
        self.vol_model: TradeScorer | None = None
        self.min_agreement: int = 2
        self.threshold: float = 0.50
        self.model = True  # Compatibility flag for engine_v2 model check
        self.metadata: dict = {}

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Load sub-models
        for attr, key in [("momentum_model", "momentum"), ("mr_model", "mean_reversion"), ("vol_model", "volatility")]:
            scorer = TradeScorer()
            scorer.model = data.get(f"{key}_model")
            scorer.feature_names = data.get(f"{key}_features", [])
            scorer.threshold = data.get("threshold", 0.50)
            setattr(self, attr, scorer)
        self.threshold = data.get("threshold", 0.50)
        self.metadata = data.get("metadata", {})
        logger.info(f"Loaded EnsembleScorer: threshold={self.threshold:.2f}, "
                    f"cv={self.metadata.get('cv_accuracy', 'N/A')}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "threshold": self.threshold,
            "metadata": self.metadata,
        }
        for attr, key, feat_list in [
            ("momentum_model", "momentum", MOMENTUM_FEATURES),
            ("mr_model", "mean_reversion", MEAN_REVERSION_FEATURES),
            ("vol_model", "volatility", VOLATILITY_FEATURES),
        ]:
            scorer = getattr(self, attr)
            if scorer:
                data[f"{key}_model"] = scorer.model
                data[f"{key}_features"] = scorer.feature_names
            else:
                data[f"{key}_model"] = None
                data[f"{key}_features"] = feat_list
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved EnsembleScorer → {path}")

    def predict_proba(self, features: dict) -> float:
        """Average probability across sub-models."""
        probs = []
        for scorer in [self.momentum_model, self.mr_model, self.vol_model]:
            if scorer and scorer.model is not None:
                probs.append(scorer.predict_proba(features))
        return np.mean(probs) if probs else 0.5

    def should_trade(self, features: dict) -> tuple[bool, float]:
        """Average probability across models — take if avg exceeds threshold."""
        probs = []
        for scorer in [self.momentum_model, self.mr_model, self.vol_model]:
            if scorer and scorer.model is not None:
                probs.append(scorer.predict_proba(features))

        avg_prob = float(np.mean(probs)) if probs else 0.5
        return avg_prob >= self.threshold, avg_prob
