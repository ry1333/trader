"""Hard quantitative trade filter — final gate before execution.

This sits AFTER the AI model scores a trade but BEFORE execution.
It combines all context signals into a single take/reduce/skip decision.

Key additions beyond what we already have:
1. Rolling performance tracking per setup type (last 20 trades)
2. Combined score from all filters (single composite decision)
3. Side-asymmetric rules (different thresholds for longs vs shorts)
4. Drawdown-aware sizing (reduce when in drawdown)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SetupPerformance:
    """Track rolling performance of each setup type."""
    recent_pnls: list[float] = field(default_factory=list)
    max_history: int = 20

    def add(self, pnl: float) -> None:
        self.recent_pnls.append(pnl)
        if len(self.recent_pnls) > self.max_history:
            self.recent_pnls.pop(0)

    @property
    def avg_pnl(self) -> float:
        return np.mean(self.recent_pnls) if self.recent_pnls else 0.0

    @property
    def win_rate(self) -> float:
        if not self.recent_pnls:
            return 0.5
        return sum(1 for p in self.recent_pnls if p > 0) / len(self.recent_pnls)

    @property
    def expectancy(self) -> float:
        """Expected value per trade from recent history."""
        return self.avg_pnl

    @property
    def is_cold(self) -> bool:
        """Losing streak: last 3+ trades all negative."""
        if len(self.recent_pnls) < 3:
            return False
        return all(p <= 0 for p in self.recent_pnls[-3:])

    @property
    def is_hot(self) -> bool:
        """Winning streak: last 3+ trades all positive."""
        if len(self.recent_pnls) < 3:
            return False
        return all(p > 0 for p in self.recent_pnls[-3:])


class TradeFilter:
    """Hard quantitative filter that combines all context signals."""

    def __init__(self):
        # Rolling performance per strategy+side
        self.setup_perf: dict[str, SetupPerformance] = {}
        # Overall account performance
        self.overall_perf = SetupPerformance(max_history=50)

    def _get_perf(self, strategy: str, direction: int) -> SetupPerformance:
        key = f"{strategy}_{'L' if direction == 1 else 'S'}"
        if key not in self.setup_perf:
            self.setup_perf[key] = SetupPerformance()
        return self.setup_perf[key]

    def record_trade(self, strategy: str, direction: int, pnl: float) -> None:
        """Record a completed trade for rolling performance."""
        perf = self._get_perf(strategy, direction)
        perf.add(pnl)
        self.overall_perf.add(pnl)

    def evaluate(
        self,
        strategy: str,
        direction: int,
        ai_confidence: float,
        htf_bearish_count: int,
        htf_bullish_count: int,
        session_grade: int,
        vol_regime: float,
        current_drawdown_pct: float,
    ) -> tuple[str, float]:
        """Final gate: combine all signals into take/reduce/skip.

        Returns (action, size_multiplier):
        - ("take", 1.0-1.5) — green light
        - ("reduce", 0.3-0.7) — caution
        - ("skip", 0.0) — no trade

        This is the HARD FILTER — simple rules, no ML, no overfitting.
        """
        perf = self._get_perf(strategy, direction)

        # ═══════════════════════════════════════════════════════════════
        # SKIP rules (any one triggers skip)
        # ═══════════════════════════════════════════════════════════════

        # 1. Setup type on cold streak (last 3 trades all losses)
        if perf.is_cold and ai_confidence < 0.65:
            return "skip", 0.0

        # 2. Counter-trend long in strong bear (the March fix)
        if direction == 1 and htf_bearish_count >= 3:
            return "skip", 0.0

        # 3. Counter-trend short in strong bull
        if direction == -1 and htf_bullish_count >= 3:
            return "skip", 0.0

        # 4. Extreme volatility (>2x normal)
        if vol_regime > 2.0:
            return "skip", 0.0

        # 5. Overall account on losing streak (last 5 trades all losses)
        if len(self.overall_perf.recent_pnls) >= 5:
            if all(p <= 0 for p in self.overall_perf.recent_pnls[-5:]):
                return "skip", 0.0

        # ═══════════════════════════════════════════════════════════════
        # REDUCE rules (cumulative — multiple reductions stack)
        # ═══════════════════════════════════════════════════════════════
        mult = 1.0

        # 1. In drawdown: scale down proportionally
        if current_drawdown_pct > 0.30:
            mult *= max(0.4, 1.0 - current_drawdown_pct)

        # 2. Counter-trend (2 HTF timeframes against)
        if direction == 1 and htf_bearish_count >= 2:
            mult *= 0.5
        elif direction == -1 and htf_bullish_count >= 2:
            mult *= 0.5

        # 3. High volatility (1.3-2.0x normal)
        if vol_regime > 1.5:
            mult *= 0.6
        elif vol_regime > 1.3:
            mult *= 0.8

        # 4. Setup type has negative recent expectancy
        if perf.expectancy < 0 and len(perf.recent_pnls) >= 5:
            mult *= 0.6

        # 5. Low session quality (C-grade)
        if session_grade <= 2:  # SessionGrade.C or lower
            mult *= 0.5

        # 6. Side-specific: longs need higher bar in ANY bearish context
        if direction == 1 and htf_bearish_count >= 1:
            mult *= 0.8

        # ═══════════════════════════════════════════════════════════════
        # BOOST rules
        # ═══════════════════════════════════════════════════════════════

        # 1. Setup type on hot streak + high AI confidence
        if perf.is_hot and ai_confidence > 0.65:
            mult *= 1.3

        # 2. Perfect HTF alignment
        if direction == 1 and htf_bullish_count >= 3:
            mult *= 1.2
        elif direction == -1 and htf_bearish_count >= 3:
            mult *= 1.2

        # 3. A-grade session
        if session_grade >= 4:  # SessionGrade.A
            mult *= 1.1

        # Clamp
        mult = max(0.3, min(1.5, mult))

        if mult < 0.35:
            return "skip", 0.0

        return "take" if mult >= 0.8 else "reduce", mult
