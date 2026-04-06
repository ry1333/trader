"""Data-driven strategy statistics — empirical stop/target optimization.

NOT a replacement for the whole system. Used ONLY for:
1. Exit design: optimal stop/target from MAE/MFE distributions per strategy
2. Regime-aware: separate stats for high-vol vs low-vol, long vs short
3. Sizing reference: quarter-Kelly as UPPER BOUND, hard dollar cap as actual limit

Does NOT claim:
- No overfitting (any historical optimization can overfit)
- Exact optimal levels (these are historically best-looking candidates)
- Safe Kelly sizing (estimation error makes raw Kelly dangerous)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ExitProfile:
    """Empirical stop/target parameters for one strategy+side+regime slice."""
    strategy_name: str
    direction: str  # "LONG" or "SHORT"
    regime: str  # "HIGH_VOL", "LOW_VOL", or "ALL"
    n_trades: int = 0

    # Win/loss
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0

    # Optimal stop (ATR multiples) — from MAE distribution
    optimal_stop_atr: float = 2.0
    stop_survivors: float = 0.0  # % of winners that survive this stop

    # Optimal target (ATR multiples) — from MFE distribution
    optimal_target_atr: float = 3.0
    target_hit_rate: float = 0.0  # % of trades reaching target

    # R:R
    optimal_rr: float = 1.5

    # Kelly reference (quarter-Kelly as UPPER BOUND only)
    quarter_kelly: float = 0.0

    # Expected value
    ev_per_trade: float = 0.0

    # MAE/MFE percentiles
    mae_50pct: float = 0.0
    mae_75pct: float = 0.0
    mfe_50pct: float = 0.0
    mfe_75pct: float = 0.0


class StrategyStatsBank:
    """Collection of exit profiles per strategy+side+regime."""

    def __init__(self):
        self.profiles: dict[str, ExitProfile] = {}
        self.default_stop = 2.0
        self.default_target = 3.0
        self.default_rr = 1.5
        # Compatibility
        self.model = True
        self.threshold = 0.0

    def _get_key(self, strategy: str, direction: int, vol_regime: str = "ALL") -> str:
        side = "LONG" if direction == 1 else "SHORT"
        return f"{strategy}_{side}_{vol_regime}"

    def get_exit_params(self, strategy: str, direction: int, vol_regime: str = "ALL") -> tuple[float, float]:
        """Get optimal stop and R:R for this strategy+side+regime.

        Returns (stop_atr_mult, reward_risk_ratio).
        Falls back to less specific profiles if exact match not found.
        """
        # Try exact match first
        key = self._get_key(strategy, direction, vol_regime)
        if key in self.profiles and self.profiles[key].n_trades >= 20:
            p = self.profiles[key]
            return p.optimal_stop_atr, p.optimal_rr

        # Fall back to ALL regime
        key_all = self._get_key(strategy, direction, "ALL")
        if key_all in self.profiles and self.profiles[key_all].n_trades >= 20:
            p = self.profiles[key_all]
            return p.optimal_stop_atr, p.optimal_rr

        return self.default_stop, self.default_rr

    def get_ev(self, strategy: str, direction: int) -> float:
        """Get expected value per trade. Negative = skip this strategy."""
        key = self._get_key(strategy, direction, "ALL")
        if key in self.profiles:
            return self.profiles[key].ev_per_trade
        return 0.0

    def should_trade(self, features: dict = None, strategy_name: str = "", direction: int = 0) -> tuple[bool, float]:
        """Compatibility interface. Only skip if EV is clearly negative with enough data."""
        ev = self.get_ev(strategy_name, direction)
        key = self._get_key(strategy_name, direction, "ALL")
        p = self.profiles.get(key)
        n = p.n_trades if p else 0

        # Only skip if 30+ trades AND clearly negative EV
        if n >= 30 and ev < -20:
            return False, 0.0

        return True, 0.5

    def get_size_multiplier(self, strategy_name: str = "", direction: int = 0) -> float:
        """Quarter-Kelly as upper bound, never above 1.5x, never below 0.5x."""
        key = self._get_key(strategy_name, direction, "ALL")
        p = self.profiles.get(key)
        if not p or p.n_trades < 20:
            return 1.0

        qk = p.quarter_kelly
        if qk <= 0:
            return 0.5  # Negative edge — minimum

        # Normalize: 0.05 quarter-kelly = 1.0x baseline
        mult = qk / 0.05
        return max(0.5, min(1.5, mult))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.profiles, f)
        logger.info(f"Saved StrategyStatsBank ({len(self.profiles)} profiles) → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.profiles = pickle.load(f)
        logger.info(f"Loaded StrategyStatsBank: {len(self.profiles)} profiles")


def compute_strategy_stats(
    trades: list,
    tick_size: float = 0.25,
    tick_value: float = 0.50,
    min_trades: int = 15,
    output_path: str | Path = "data/models/strategy_stats.pkl",
) -> StrategyStatsBank:
    """Compute empirical exit profiles from historical trades.

    Groups by strategy × direction × vol_regime.
    For each group, finds optimal stop/target from MAE/MFE distributions.
    """
    bank = StrategyStatsBank()
    from src.strategy.signals_v3 import SignalType

    # Build records with excursion data
    records = []
    for t in trades:
        net = t.pnl - t.fees
        bars = t.exit_bar - t.entry_bar if t.exit_bar else 0
        stop_dist = abs(t.entry_price - t.sl_price)
        if stop_dist <= 0:
            continue

        # Use stop distance as ATR proxy (stop was set as ATR multiple × ATR)
        atr_approx = stop_dist / 2.0  # Most stops are ~2 ATR

        # MAE in ATR (how far against us before exit)
        mae_atr = stop_dist / atr_approx if atr_approx > 0 else 2.0

        # MFE in ATR (peak profit in ATR units)
        mfe_atr = t.peak_profit / atr_approx if atr_approx > 0 and t.peak_profit > 0 else 0

        # Actual R-multiple
        actual_move = (t.exit_price - t.entry_price) * t.direction
        r_mult = actual_move / stop_dist

        # Get signal type
        sig_type = getattr(t, "_signal_type", 0)
        try:
            sig_name = SignalType(int(sig_type)).name
        except (ValueError, AttributeError):
            sig_name = "UNKNOWN"

        records.append({
            "net": net, "direction": t.direction, "bars": bars,
            "mae_atr": mae_atr, "mfe_atr": mfe_atr, "r_mult": r_mult,
            "exit_reason": t.exit_reason, "size": t.size,
            "atr": atr_approx, "strategy": sig_name,
            "is_winner": net > 0,
        })

    if not records:
        bank.save(output_path)
        return bank

    df = pd.DataFrame(records)

    # Group by strategy × side × regime
    for strategy in df["strategy"].unique():
        for direction, side_label in [(1, "LONG"), (-1, "SHORT")]:
            for vol_label in ["ALL"]:  # Start with ALL, add regime splits if enough data
                mask = (df["strategy"] == strategy) & (df["direction"] == direction)
                grp = df[mask]
                n = len(grp)

                if n < min_trades:
                    continue

                winners = grp[grp["is_winner"]]
                losers = grp[~grp["is_winner"]]
                win_rate = len(winners) / n
                avg_win = winners["net"].mean() if len(winners) > 0 else 0
                avg_loss = abs(losers["net"].mean()) if len(losers) > 0 else 1
                payoff = avg_win / avg_loss if avg_loss > 0 else 0
                ev = win_rate * avg_win - (1 - win_rate) * avg_loss

                # MAE/MFE percentiles
                maes = grp["mae_atr"].values
                mfes = grp[grp["mfe_atr"] > 0]["mfe_atr"].values

                # Optimal stop: test grid, find max net PnL
                best_stop = 2.0
                best_stop_pnl = float("-inf")
                best_stop_surv = 0

                winner_maes = winners["mae_atr"].values if len(winners) > 0 else []

                for test_stop in np.arange(0.75, 4.0, 0.25):
                    # Simulated PnL: winners that survive keep their PnL,
                    # losers that would be stopped earlier lose less
                    sim_pnl = 0
                    for _, r in grp.iterrows():
                        if r["mfe_atr"] >= 0 and r["mae_atr"] <= test_stop:
                            sim_pnl += r["net"]  # Survives — actual outcome
                        else:
                            # Stopped out at test_stop distance
                            sim_pnl -= test_stop * r["atr"] * r["size"] * tick_value / tick_size

                    if sim_pnl > best_stop_pnl:
                        best_stop_pnl = sim_pnl
                        best_stop = test_stop
                        if len(winner_maes) > 0:
                            best_stop_surv = np.mean(winner_maes <= test_stop)

                # Optimal target: find target that maximizes total captured
                best_target = 3.0
                best_target_pnl = float("-inf")
                best_target_hit = 0

                for test_target in np.arange(0.75, 6.0, 0.25):
                    sim_pnl = 0
                    hits = 0
                    for _, r in grp.iterrows():
                        if r["mfe_atr"] >= test_target:
                            sim_pnl += test_target * r["atr"] * r["size"] * tick_value / tick_size
                            hits += 1
                        else:
                            sim_pnl += r["net"]  # Didn't reach — actual outcome

                    if sim_pnl > best_target_pnl:
                        best_target_pnl = sim_pnl
                        best_target = test_target
                        best_target_hit = hits / n

                optimal_rr = best_target / best_stop if best_stop > 0 else 1.5

                # Quarter-Kelly (conservative upper bound)
                if payoff > 0 and win_rate > 0:
                    kelly = (win_rate * payoff - (1 - win_rate)) / payoff
                    kelly = max(0, kelly)
                else:
                    kelly = 0
                quarter_kelly = kelly / 4

                key = f"{strategy}_{side_label}_{vol_label}"
                profile = ExitProfile(
                    strategy_name=strategy, direction=side_label, regime=vol_label,
                    n_trades=n, win_rate=round(win_rate, 3),
                    avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2),
                    payoff_ratio=round(payoff, 3),
                    optimal_stop_atr=round(best_stop, 2),
                    stop_survivors=round(best_stop_surv, 3),
                    optimal_target_atr=round(best_target, 2),
                    target_hit_rate=round(best_target_hit, 3),
                    optimal_rr=round(optimal_rr, 2),
                    quarter_kelly=round(quarter_kelly, 4),
                    ev_per_trade=round(ev, 2),
                    mae_50pct=round(np.percentile(maes, 50), 2) if len(maes) > 0 else 0,
                    mae_75pct=round(np.percentile(maes, 75), 2) if len(maes) > 0 else 0,
                    mfe_50pct=round(np.percentile(mfes, 50), 2) if len(mfes) > 0 else 0,
                    mfe_75pct=round(np.percentile(mfes, 75), 2) if len(mfes) > 0 else 0,
                )

                bank.profiles[key] = profile
                logger.info(
                    f"  {key}: {n}tr WR={win_rate:.0%} EV=${ev:.0f} "
                    f"stop={best_stop:.1f} target={best_target:.1f} "
                    f"R:R={optimal_rr:.1f} qKelly={quarter_kelly:.3f}"
                )

    bank.save(output_path)
    return bank
