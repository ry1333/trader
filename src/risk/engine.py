"""Risk engine — enforces prop firm rules and position sizing.

Topstep rules enforced:
- Max daily loss limit
- Max total loss limit (trailing)
- Consistency target (best day < 50% of total profit)
- Forced flatten before 3:10 PM CT
- Max position size
- Per-trade risk budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from src.config import RiskConfig


@dataclass
class DayStats:
    date: str = ""
    pnl: float = 0.0
    trades: int = 0
    peak_pnl: float = 0.0


@dataclass
class RiskState:
    """Tracks running risk metrics for a session."""
    starting_balance: float = 50_000.0
    current_balance: float = 50_000.0
    total_pnl: float = 0.0
    day_pnl: float = 0.0
    day_trades: int = 0
    current_position: int = 0  # +N long, -N short, 0 flat
    best_day_pnl: float = 0.0
    daily_history: list[DayStats] = field(default_factory=list)
    is_killed: bool = False


class RiskEngine:
    """Enforces all prop firm risk constraints."""

    def __init__(self, cfg: RiskConfig, starting_balance: float = 50_000.0) -> None:
        self.cfg = cfg
        self.state = RiskState(
            starting_balance=starting_balance,
            current_balance=starting_balance,
        )

    def can_trade(self, current_time_ct_minutes: int, max_trades_per_day: int) -> bool:
        """Check if a new trade is allowed right now."""
        if self.state.is_killed:
            return False

        # Flatten zone: no new trades after flatten_time
        flatten_h, flatten_m = map(int, self.cfg.flatten_time_ct.split(":"))
        flatten_minutes = flatten_h * 60 + flatten_m
        if current_time_ct_minutes >= flatten_minutes:
            return False

        # Daily trade limit
        if self.state.day_trades >= max_trades_per_day:
            return False

        # Daily loss check
        if self.state.day_pnl <= -self.cfg.max_daily_loss:
            logger.warning(f"Daily loss limit hit: ${self.state.day_pnl:.2f}")
            return False

        # Total loss check
        if self.state.total_pnl <= -self.cfg.max_total_loss:
            logger.warning(f"Max loss limit hit: ${self.state.total_pnl:.2f}")
            self.state.is_killed = True
            return False

        return True

    def compute_position_size(self, atr: float, tick_size: float, tick_value: float) -> int:
        """Compute position size based on ATR and risk budget.

        Risk per trade = risk_per_trade_pct * current_balance
        Size = risk_budget / (atr_in_ticks * tick_value)
        Capped at max_position_size.
        """
        if atr <= 0:
            return 0

        risk_budget = self.cfg.risk_per_trade_pct / 100 * self.state.current_balance
        atr_ticks = atr / tick_size
        risk_per_contract = atr_ticks * tick_value

        if risk_per_contract <= 0:
            return 0

        size = int(risk_budget / risk_per_contract)
        return max(1, min(size, self.cfg.max_position_size))

    def compute_stop_ticks(self, atr: float, tick_size: float, multiplier: float = 2.0) -> int:
        """Compute stop-loss distance in ticks based on ATR."""
        if atr <= 0 or tick_size <= 0:
            return 8  # fallback: 8 ticks = 2 points ES
        return max(4, int((atr * multiplier) / tick_size))

    def compute_target_ticks(self, stop_ticks: int, reward_risk: float = 1.5) -> int:
        """Compute take-profit distance in ticks."""
        return max(4, int(stop_ticks * reward_risk))

    def record_trade(self, pnl: float, fees: float) -> None:
        """Record a completed trade's P&L."""
        net = pnl - fees
        self.state.day_pnl += net
        self.state.total_pnl += net
        self.state.current_balance += net
        self.state.day_trades += 1

    def end_day(self, date_str: str) -> None:
        """Finalize daily stats and reset for next day."""
        day = DayStats(
            date=date_str,
            pnl=self.state.day_pnl,
            trades=self.state.day_trades,
        )
        self.state.daily_history.append(day)

        if self.state.day_pnl > self.state.best_day_pnl:
            self.state.best_day_pnl = self.state.day_pnl

        # Reset daily counters
        self.state.day_pnl = 0.0
        self.state.day_trades = 0

    def check_consistency(self, profit_target: float) -> bool:
        """Check if best day violates consistency target.

        Topstep: best day must be < 50% of total profit.
        """
        if self.state.total_pnl <= 0:
            return True  # Not relevant if not profitable
        return self.state.best_day_pnl < (self.cfg.consistency_target * self.state.total_pnl)

    @property
    def summary(self) -> dict:
        return {
            "total_pnl": self.state.total_pnl,
            "current_balance": self.state.current_balance,
            "best_day_pnl": self.state.best_day_pnl,
            "total_trades": sum(d.trades for d in self.state.daily_history),
            "trading_days": len(self.state.daily_history),
            "is_killed": self.state.is_killed,
        }
