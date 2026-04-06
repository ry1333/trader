"""Strategy Model Bank — per-strategy AI models with long/short specialization.

Instead of one generic model scoring all 13 strategies:
- High-frequency strategies get separate LONG and SHORT models
- Medium-frequency strategies get one model with direction as feature
- Low-frequency strategies merge into parent groups

Each model is a GBT trained on its own strategy's historical trades
with triple-barrier (R-multiple) labels.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.ai.features import AI_FEATURE_COLS


# Strategy groupings for model assignment
# High-frequency: separate long/short models
HIGH_FREQ_STRATEGIES = {"TREND_CONTINUATION", "RANGE_BREAKOUT", "EMA_PULLBACK", "ORB"}

# Medium-frequency: single model with direction feature
MED_FREQ_STRATEGIES = {"MOMENTUM_IGNITION", "VWAP_RECLAIM", "VWAP_REVERSION",
                        "FAILED_BREAKOUT", "RSI_REVERSAL"}

# Low-frequency: merge into parent
MERGE_MAP = {
    "ODPC": "TREND_CONTINUATION",
    "VOL_CONTRACTION": "RANGE_BREAKOUT",
    "PREV_DAY_LEVEL": "RANGE_BREAKOUT",
    "SESSION_LEVEL": "RANGE_BREAKOUT",
}

MIN_SAMPLES = 20  # Lowered from 50 — MGC has fewer signals per strategy


class StrategyModel:
    """Single strategy-specific AI model."""

    def __init__(self):
        self.ev_model: GradientBoostingRegressor | None = None
        self.skip_model: GradientBoostingClassifier | None = None
        self.feature_names: list[str] = AI_FEATURE_COLS
        self.ev_threshold: float = 0.0
        self.skip_threshold: float = 0.50
        self.n_samples: int = 0
        self.cv_r2: float = 0.0
        self.cv_skip_acc: float = 0.0

    def should_trade(self, features: dict) -> tuple[bool, float]:
        """Decide whether to take this trade."""
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Skip check
        if self.skip_model is not None:
            skip_prob = float(self.skip_model.predict_proba(X)[0][1])
            if skip_prob > self.skip_threshold:
                return False, 0.0
        else:
            skip_prob = 0.3

        # EV prediction
        if self.ev_model is not None:
            ev = float(self.ev_model.predict(X)[0])
            if ev < self.ev_threshold:
                return False, 0.0
        else:
            ev = 0.0

        # Convert to probability-like score for compatibility
        prob = 1.0 / (1.0 + np.exp(-ev / 50.0))
        self._last_skip = skip_prob
        self._last_ev = ev
        return True, float(prob)

    def get_size_multiplier(self) -> float:
        """Dynamic sizing based on skip probability."""
        skip = getattr(self, "_last_skip", 0.3)
        ev = getattr(self, "_last_ev", 0)

        if skip < 0.15:
            base = 1.5
        elif skip < 0.25:
            base = 1.3
        elif skip < 0.35:
            base = 1.1
        else:
            base = 0.8

        if ev > 80:
            base *= 1.15
        elif ev < -30:
            base *= 0.85

        return max(0.6, min(1.6, base))


class StrategyModelBank:
    """Collection of per-strategy models."""

    def __init__(self):
        self.models: dict[str, StrategyModel] = {}
        self.fallback = StrategyModel()  # Generic fallback
        # Compatibility flags
        self.model = True
        self.threshold = 0.50

    def _get_key(self, strategy_name: str, direction: int) -> str:
        """Get model lookup key for a strategy + direction."""
        # Check if this strategy merges into a parent
        base_strategy = MERGE_MAP.get(strategy_name, strategy_name)

        if base_strategy in HIGH_FREQ_STRATEGIES:
            side = "LONG" if direction == 1 else "SHORT"
            return f"{base_strategy}_{side}"
        else:
            return base_strategy

    def get_model(self, strategy_name: str, direction: int) -> StrategyModel:
        """Look up the correct model for a strategy + direction."""
        key = self._get_key(strategy_name, direction)
        return self.models.get(key, self.fallback)

    def should_trade(self, features: dict, strategy_name: str = "", direction: int = 0) -> tuple[bool, float]:
        """Interface compatible with QualityRiskScorer."""
        if strategy_name and direction:
            model = self.get_model(strategy_name, direction)
        else:
            model = self.fallback
        return model.should_trade(features)

    def get_size_multiplier(self, strategy_name: str = "", direction: int = 0) -> float:
        """Get sizing multiplier from the relevant model."""
        if strategy_name and direction:
            model = self.get_model(strategy_name, direction)
        else:
            model = self.fallback
        return model.get_size_multiplier()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, model in self.models.items():
            data[key] = {
                "ev_model": model.ev_model,
                "skip_model": model.skip_model,
                "feature_names": model.feature_names,
                "ev_threshold": model.ev_threshold,
                "skip_threshold": model.skip_threshold,
                "n_samples": model.n_samples,
                "cv_r2": model.cv_r2,
                "cv_skip_acc": model.cv_skip_acc,
            }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved StrategyModelBank ({len(self.models)} models) → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for key, mdata in data.items():
            model = StrategyModel()
            model.ev_model = mdata["ev_model"]
            model.skip_model = mdata["skip_model"]
            model.feature_names = mdata.get("feature_names", AI_FEATURE_COLS)
            model.ev_threshold = mdata.get("ev_threshold", 0.0)
            model.skip_threshold = mdata.get("skip_threshold", 0.50)
            model.n_samples = mdata.get("n_samples", 0)
            model.cv_r2 = mdata.get("cv_r2", 0.0)
            model.cv_skip_acc = mdata.get("cv_skip_acc", 0.0)
            self.models[key] = model
        logger.info(f"Loaded StrategyModelBank: {len(self.models)} models")


def train_strategy_bank(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_path: str | Path = "data/models/strategy_bank.pkl",
) -> dict:
    """Train all per-strategy models from backtest data.

    trades_df must have: entry_bar, r_multiple, net_pnl, direction, signal_type_name
    features_df must have: entry_bar + AI_FEATURE_COLS
    """
    merged = trades_df.merge(features_df, on="entry_bar", how="inner", suffixes=("", "_feat"))

    feature_cols = [c for c in AI_FEATURE_COLS if c in merged.columns]
    bank = StrategyModelBank()
    metadata = {"models": {}}

    # Group trades by model key
    groups: dict[str, pd.DataFrame] = {}
    for _, row in merged.iterrows():
        strategy = row.get("signal_type_name", "UNKNOWN")
        direction = int(row.get("direction", 0))
        base = MERGE_MAP.get(strategy, strategy)

        if base in HIGH_FREQ_STRATEGIES:
            side = "LONG" if direction == 1 else "SHORT"
            key = f"{base}_{side}"
        else:
            key = base

        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Train each group
    for key, rows in groups.items():
        group_df = pd.DataFrame(rows)
        n = len(group_df)

        if n < MIN_SAMPLES:
            logger.info(f"  {key}: {n} samples (below {MIN_SAMPLES}, skip)")
            continue

        X = group_df[feature_cols].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # R-multiple labels for EV regression
        r_col = "r_multiple" if "r_multiple" in group_df.columns else "net_pnl"
        y_r = group_df[r_col].values.astype(float)

        # Skip labels: bottom 25th percentile = "large loser"
        large_loss_thresh = np.percentile(y_r, 25)
        y_skip = (y_r < large_loss_thresh).astype(int)

        model = StrategyModel()
        model.feature_names = feature_cols
        model.n_samples = n

        # Train EV regressor on R-multiples
        ev_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=max(5, n // 50), random_state=42)
        n_splits = min(5, max(2, n // 20))
        if n_splits >= 2:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            ev_r2 = cross_val_score(ev_model, X, y_r, cv=cv, scoring="r2").mean()
        else:
            ev_r2 = 0.0
        ev_model.fit(X, y_r)
        model.ev_model = ev_model
        model.cv_r2 = round(ev_r2, 4)

        # Train skip classifier
        skip_model = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_samples_leaf=max(5, n // 50), random_state=42)
        if n_splits >= 2 and len(np.unique(y_skip)) > 1:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            skip_acc = cross_val_score(skip_model, X, y_skip, cv=cv, scoring="accuracy").mean()
        else:
            skip_acc = 0.0
        skip_model.fit(X, y_skip)
        model.skip_model = skip_model
        model.cv_skip_acc = round(skip_acc, 4)

        # Optimize thresholds
        ev_preds = ev_model.predict(X)
        skip_preds = skip_model.predict_proba(X)[:, 1]
        pnl_col = "net_pnl" if "net_pnl" in group_df.columns else "net_pnl_feat"
        pnls = group_df[pnl_col].values if pnl_col in group_df.columns else y_r

        best_score = float("-inf")
        best_ev_t, best_skip_t = 0.0, 0.50
        for ev_t in [-10, 0, 10, 20, 30]:
            for skip_t in [0.35, 0.45, 0.55]:
                mask = (ev_preds > ev_t) & (skip_preds < skip_t)
                if mask.sum() < 5:
                    continue
                score = pnls[mask].sum() + abs(pnls[~mask & (pnls < 0)].sum()) * 0.3
                if score > best_score:
                    best_score = score
                    best_ev_t, best_skip_t = ev_t, skip_t

        model.ev_threshold = best_ev_t
        model.skip_threshold = best_skip_t
        bank.models[key] = model

        metadata["models"][key] = {
            "n_samples": n,
            "cv_r2": model.cv_r2,
            "cv_skip_acc": model.cv_skip_acc,
            "ev_threshold": best_ev_t,
            "skip_threshold": best_skip_t,
            "win_rate": round(float((y_r > 0).mean()), 3),
        }

        logger.info(f"  {key}: {n}tr, R²={ev_r2:.3f}, skip={skip_acc:.3f}")

    bank.save(output_path)

    # Train fallback generic model on ALL data
    all_X = merged[feature_cols].values.astype(float)
    all_X = np.nan_to_num(all_X, nan=0.0, posinf=0.0, neginf=0.0)
    r_col = "r_multiple" if "r_multiple" in merged.columns else "net_pnl"
    all_y = merged[r_col].values.astype(float)

    fallback = StrategyModel()
    fallback.feature_names = feature_cols
    fb_ev = GradientBoostingRegressor(n_estimators=80, max_depth=3, min_samples_leaf=5, random_state=42)
    fb_ev.fit(all_X, all_y)
    fallback.ev_model = fb_ev
    bank.fallback = fallback

    metadata["total_models"] = len(bank.models)
    metadata["total_samples"] = len(merged)
    return metadata
