"""Central configuration loaded from settings.toml + environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _ROOT / "config" / "settings.toml"


@dataclass(frozen=True)
class TopstepConfig:
    rest_url: str
    market_hub: str
    user_hub: str
    username: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class StrategyConfig:
    instrument: str
    bar_interval_minutes: int
    max_trades_per_day: int
    max_hold_bars: int


@dataclass(frozen=True)
class RiskConfig:
    max_daily_loss: float
    max_total_loss: float
    max_position_size: int
    risk_per_trade_pct: float
    flatten_time_ct: str
    session_start_ct: str
    consistency_target: float
    max_risk_per_trade: float = 100.0
    daily_loss_tier1: float = 500.0
    daily_loss_tier2: float = 1000.0
    weekly_loss_limit: float = 1500.0
    max_simultaneous_positions: int = 2


@dataclass(frozen=True)
class BacktestConfig:
    train_window_days: int
    val_window_days: int
    test_window_days: int
    walk_forward_step_days: int
    cost_per_side_per_contract: float
    slippage_ticks: int
    tick_size: float
    tick_value: float

    @property
    def cost_per_round_trip(self) -> float:
        return (self.cost_per_side_per_contract * 2) + (self.slippage_ticks * self.tick_value)


@dataclass(frozen=True)
class Settings:
    topstep: TopstepConfig
    strategy: StrategyConfig
    risk: RiskConfig
    backtest: BacktestConfig


def load_settings(path: Path = _CFG_PATH) -> Settings:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    ts = raw["topstep"]
    return Settings(
        topstep=TopstepConfig(
            rest_url=ts["rest_url"],
            market_hub=ts["market_hub"],
            user_hub=ts["user_hub"],
            username=os.environ.get("TOPSTEP_USERNAME", ""),
            api_key=os.environ.get("TOPSTEP_API_KEY", ""),
        ),
        strategy=StrategyConfig(**raw["strategy"]),
        risk=RiskConfig(**raw["risk"]),
        backtest=BacktestConfig(**raw["backtest"]),
    )
