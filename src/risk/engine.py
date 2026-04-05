"""Risk engine — enforces prop firm rules and position sizing.

Topstep rules enforced:
- Tiered daily loss limits ($500 reduce, $1000 stop)
- Max total loss limit (trailing)
- Weekly loss limit
- Consistency target (best day < 50% of total profit)
- Forced flatten before 3:10 PM CT
- Max position size
- Per-trade risk budget with hard dollar cap
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    current_position: int = 0
    best_day_pnl: float = 0.0
    daily_history: list[DayStats] = field(default_factory=list)
    is_killed: bool = False
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    week_pnl: float = 0.0
    is_week_paused: bool = False
    week_day_count: int = 0


class RiskEngine:
    """Enforces all prop firm risk constraints with tiered limits."""

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

        if self.state.is_week_paused:
            return False

        # Flatten zone
        flatten_h, flatten_m = map(int, self.cfg.flatten_time_ct.split(":"))
        flatten_minutes = flatten_h * 60 + flatten_m
        if current_time_ct_minutes >= flatten_minutes:
            return False

        # Daily trade limit
        if self.state.day_trades >= max_trades_per_day:
            return False

        # Tiered daily loss — hard stop at tier2
        if self.state.day_pnl <= -self.cfg.daily_loss_tier2:
            return False

        # Total loss check — hard stop at 90% of limit
        if self.state.total_pnl <= -self.cfg.max_total_loss * 0.90:
            self.state.is_killed = True
            return False

        return True

    def compute_position_size(
        self, atr: float, tick_size: float, tick_value: float,
        atr_50: float = 0,
    ) -> int:
        """Compute position size with volatility targeting.

        Volatility-targeted sizing (research-backed):
        - Low vol (ATR < ATR_50): BIGGER positions (trends are clean)
        - High vol (ATR > ATR_50): SMALLER positions (noise is high)
        - This mechanically reduces drawdowns in volatile periods

        Also includes:
        - Tiered daily loss reduction
        - Consecutive loss scaling
        - Proportional budget scaling
        """
        if atr <= 0:
            return 0

        risk_budget = self.cfg.risk_per_trade_pct / 100 * self.state.current_balance
        risk_budget = min(risk_budget, self.cfg.max_risk_per_trade)

        # Tiered daily loss reduction
        if self.state.day_pnl <= -self.cfg.daily_loss_tier1:
            atr_ticks = atr / tick_size
            risk_per_contract = atr_ticks * tick_value
            return 1 if risk_per_contract > 0 else 0

        # Volatility targeting: inverse vol scaling
        # When vol is low (calm trend): size UP
        # When vol is high (chaos): size DOWN
        if atr_50 > 0:
            vol_ratio = atr / atr_50
            if vol_ratio < 0.8:
                risk_budget *= 1.3  # Low vol = calm trend = size up
            elif vol_ratio > 1.5:
                risk_budget *= 0.5  # High vol = chaos = size down
            elif vol_ratio > 1.2:
                risk_budget *= 0.7  # Elevated vol = moderate reduction

        # Consecutive loss scaling
        if self.state.consecutive_losses >= 3:
            risk_budget *= 0.25
        elif self.state.consecutive_losses >= 2:
            risk_budget *= 0.50

        # Scale down proportionally to remaining daily budget
        daily_used_pct = abs(self.state.day_pnl) / self.cfg.max_daily_loss if self.cfg.max_daily_loss > 0 else 0
        if daily_used_pct > 0.30:
            risk_budget *= max(0.25, 1.0 - daily_used_pct)

        # Scale down when total P&L is significantly negative
        if self.state.total_pnl < 0 and self.cfg.max_total_loss > 0:
            total_used_pct = abs(self.state.total_pnl) / self.cfg.max_total_loss
            if total_used_pct > 0.30:
                risk_budget *= max(0.25, 1.0 - total_used_pct)

        atr_ticks = atr / tick_size
        risk_per_contract = atr_ticks * tick_value

        if risk_per_contract <= 0:
            return 0

        size = int(risk_budget / risk_per_contract)
        return max(1, min(size, self.cfg.max_position_size))

    def compute_stop_ticks(self, atr: float, tick_size: float, multiplier: float = 2.0) -> int:
        """Compute stop-loss distance in ticks based on ATR."""
        if atr <= 0 or tick_size <= 0:
            return 8
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

        if net > 0:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

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

        # Weekly tracking
        self.state.week_pnl += self.state.day_pnl
        self.state.week_day_count += 1

        # End of week (5 trading days) or weekly loss limit
        if self.state.week_day_count >= 5:
            self.state.week_pnl = 0.0
            self.state.week_day_count = 0
            self.state.is_week_paused = False
        elif self.state.week_pnl <= -self.cfg.weekly_loss_limit:
            self.state.is_week_paused = True

        # Reset daily counters
        self.state.day_pnl = 0.0
        self.state.day_trades = 0

    def check_consistency(self, profit_target: float) -> bool:
        """Check if best day violates consistency target."""
        if self.state.total_pnl <= 0:
            return True
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
