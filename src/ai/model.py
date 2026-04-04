"""AI trade scorer — predicts win probability for each setup.

Same pattern as Morgan bot's BreakoutScorer:
- GradientBoostingClassifier on tabular features
- Probability output (0.0 - 1.0)
- Threshold-based filtering
- Setup quality bonus adjustment
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger

from src.ai.features import AI_FEATURE_COLS, extract_ai_features


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
        """Predict win probability for a trade setup.

        Returns float 0.0 - 1.0 (probability of profitable trade).
        """
        if self.model is None:
            return 0.5  # No model loaded, neutral

        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        # Replace any NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self.model.predict_proba(X)[0]
        win_prob = float(proba[1])  # [prob_loss, prob_win]

        # Adjust with setup quality bonus (like Morgan's quality bonus)
        quality = features.get("setup_quality", 0.0)
        adjusted = win_prob + (quality / 200.0)  # +/- 10% max adjustment
        return max(0.0, min(1.0, adjusted))

    def should_trade(self, features: dict) -> tuple[bool, float]:
        """Decide whether to take a trade.

        Returns (should_take, win_probability).
        """
        prob = self.predict_proba(features)
        return prob >= self.threshold, prob

    def score_and_rank(self, setups: list[dict]) -> list[tuple[int, float, bool]]:
        """Score multiple setups and rank by win probability.

        Returns list of (index, probability, should_take) sorted by probability desc.
        """
        scored = []
        for i, features in enumerate(setups):
            prob = self.predict_proba(features)
            take = prob >= self.threshold
            scored.append((i, prob, take))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
