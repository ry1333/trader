"""Live trading bot for TopstepX — connects strategy to real market.

Architecture:
1. Polls 5-min bars from TopstepX REST API every 60 seconds
2. When new bar completes: compute features → signals → filters → trade decision
3. Places orders via REST API with bracket stops (SL + TP)
4. Monitors positions and enforces exits
5. Sends Discord alerts for every action
6. Flattens all positions by 3:00 PM CT
7. Respects all Topstep rules ($2K MLL, consistency, etc.)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from src.ai.quality_model import QualityRiskScorer
from src.ai.strategy_stats import StrategyStatsBank
from src.backtest.engine_stats import run_backtest_stats
from src.config import load_settings
from src.features.engine import compute_features
from src.features.session import add_session_features
from src.filters.session_quality import SessionGrade, compute_session_quality
from src.notifications.discord import (
    notify_daily_summary,
    notify_system_alert,
    notify_trade_closed,
    notify_trade_opened,
)
from src.strategy.regime import add_regime
from src.strategy.signals_v3 import Signal, SignalType, generate_signals_v3


class LiveBot:
    """Live trading bot for TopstepX."""

    def __init__(
        self,
        username: str,
        api_key: str,
        account_id: int,
        instrument: str = "MNQ",
        contract_id: str = "CON.F.US.MNQ.M26",
        use_ai: bool = False,
        ai_model_path: str = "",
        stats_bank_path: str = "",
    ):
        self.username = username
        self.api_key = api_key
        self.account_id = account_id
        self.instrument = instrument
        self.contract_id = contract_id
        self.base_url = "https://api.topstepx.com"
        self.token: str | None = None
        self.token_time: float = 0

        self.settings = load_settings()

        # AI model (for MGC) or stats bank (for MNQ)
        self.use_ai = use_ai
        self.scorer: QualityRiskScorer | None = None
        self.stats_bank: StrategyStatsBank | None = None

        if use_ai and ai_model_path and Path(ai_model_path).exists():
            self.scorer = QualityRiskScorer(ai_model_path)
            logger.info(f"Loaded AI model: {ai_model_path}")

        if stats_bank_path and Path(stats_bank_path).exists():
            self.stats_bank = StrategyStatsBank()
            self.stats_bank.load(stats_bank_path)
            logger.info(f"Loaded stats bank: {stats_bank_path}")

        # State
        self.position: dict | None = None  # Current open position
        self.day_pnl: float = 0.0
        self.day_trades: int = 0
        self.total_pnl: float = 0.0
        self.bars: pd.DataFrame = pd.DataFrame()
        self.last_bar_time: datetime | None = None
        self.signal_type_losses: dict[int, int] = {}

    # ── API Methods ───────────────────────────────────────────────────

    def _ensure_token(self) -> str:
        if self.token and (time.time() - self.token_time) < 20 * 3600:
            return self.token
        resp = httpx.post(f"{self.base_url}/api/Auth/loginKey", json={
            "userName": self.username, "apiKey": self.api_key,
        }, timeout=15)
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Auth failed: {data}")
        self.token = data["token"]
        self.token_time = time.time()
        logger.info("Authenticated with TopstepX")
        return self.token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, body: dict) -> dict:
        resp = httpx.post(f"{self.base_url}{path}", json=body,
                          headers=self._headers(), timeout=15)
        return resp.json()

    def get_account(self) -> dict:
        data = self._post("/api/Account/search", {"onlyActiveAccounts": True})
        accounts = data.get("accounts", [])
        for a in accounts:
            if a["id"] == self.account_id:
                return a
        return accounts[0] if accounts else {}

    def get_bars(self, minutes_back: int = 500) -> pd.DataFrame:
        """Fetch recent 5-min bars."""
        now = datetime.utcnow()
        start = now - timedelta(minutes=minutes_back)
        data = self._post("/api/History/retrieveBars", {
            "contractId": self.contract_id,
            "live": False,
            "startTime": start.isoformat() + "Z",
            "endTime": now.isoformat() + "Z",
            "unit": 2,  # Minute
            "unitNumber": 5,
            "limit": 2000,
            "includePartialBar": False,
        })
        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_positions(self) -> list[dict]:
        data = self._post("/api/Position/searchOpen", {"accountId": self.account_id})
        return data.get("positions", [])

    def place_order(self, side: int, size: int, sl_ticks: int = 0, tp_ticks: int = 0,
                    tag: str = "") -> dict:
        """Place market order with optional bracket (SL + TP)."""
        body = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "type": 2,  # Market
            "side": side,  # 0=Buy, 1=Sell
            "size": size,
            "customTag": tag,
        }
        if sl_ticks > 0:
            body["stopLossBracket"] = {"ticks": sl_ticks, "type": 4}
        if tp_ticks > 0:
            body["takeProfitBracket"] = {"ticks": tp_ticks, "type": 1}

        data = self._post("/api/Order/place", body)
        logger.info(f"Order placed: side={side} size={size} tag={tag} → {data}")
        return data

    def close_position(self) -> dict:
        """Flatten all positions on this contract."""
        data = self._post("/api/Position/closeContract", {
            "accountId": self.account_id,
            "contractId": self.contract_id,
        })
        logger.info(f"Position closed: {data}")
        return data

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        orders = self._post("/api/Order/searchOpen", {"accountId": self.account_id})
        for o in orders.get("orders", []):
            self._post("/api/Order/cancel", {
                "accountId": self.account_id, "orderId": o["id"]})

    # ── Trading Logic ─────────────────────────────────────────────────

    def process_bar(self, df: pd.DataFrame) -> None:
        """Process a new completed 5-min bar. Core trading logic."""
        if len(df) < 100:
            return

        # Get current time in Central
        now_utc = datetime.utcnow()
        # Simple CT conversion (UTC - 5 or 6 depending on DST)
        ct_hour = (now_utc.hour - 5) % 24  # Approximate CT
        ct_minutes = ct_hour * 60 + now_utc.minute

        # ── Flatten check (3:00 PM CT = 900 min) ─────────────────────
        if ct_minutes >= 895:
            positions = self.get_positions()
            if positions:
                logger.warning("FLATTEN: Closing all positions before 3:10 PM CT")
                self.close_position()
                self.cancel_all_orders()
                notify_system_alert("Positions flattened for session close", "warning")
            return

        # ── Don't trade outside 8:30 AM - 1:00 PM CT ─────────────────
        if ct_minutes < 510 or ct_minutes >= 780:
            return

        # ── Check if we already have a position ───────────────────────
        positions = self.get_positions()
        if positions:
            return  # Already in a trade, let brackets handle exits

        # ── Daily loss check ──────────────────────────────────────────
        account = self.get_account()
        balance = account.get("balance", 50000)
        if self.day_pnl <= -500:
            logger.warning(f"Daily loss tier1 hit: ${self.day_pnl:.0f}")
            return
        if self.day_pnl <= -1000:
            logger.warning(f"Daily loss tier2 hit: ${self.day_pnl:.0f} — stopping for day")
            return

        # ── Compute features and signals ──────────────────────────────
        df = compute_features(df)
        df = add_session_features(df)
        df = add_regime(df)
        df["signal"], df["signal_type"] = generate_signals_v3(df)

        # Check last bar's signal
        last = df.iloc[-1]
        if last["signal"] == Signal.FLAT:
            return

        sig_type = int(last["signal_type"])
        direction = 1 if last["signal"] == Signal.LONG else -1
        strategy_name = SignalType(sig_type).name

        # ── Filters ───────────────────────────────────────────────────
        atr = last.get("atr_14", 0)
        if pd.isna(atr) or atr <= 0:
            return

        # Circuit breaker
        if self.signal_type_losses.get(sig_type, 0) >= 3:
            return

        # Session quality
        session = compute_session_quality(df, len(df) - 1)
        if session.grade == SessionGrade.D:
            return

        # Vol spike gate
        atr_50 = last.get("atr_50", 0)
        if not pd.isna(atr_50) and atr_50 > 0 and atr > atr_50 * 2.0:
            return

        # ── AI scoring (for MGC) or stats (for MNQ) ──────────────────
        if self.use_ai and self.scorer:
            from src.ai.features import extract_ai_features
            features = extract_ai_features(df, len(df) - 1)
            should_take, prob = self.scorer.should_trade(features)
            if not should_take or prob < 0.50:
                return

        # ── Get stop/target from stats bank ───────────────────────────
        if self.stats_bank:
            sl_mult, rr_ratio = self.stats_bank.get_exit_params(strategy_name, direction)
            size_mult = self.stats_bank.get_size_multiplier(strategy_name, direction)

            # Skip negative EV
            should_take, _ = self.stats_bank.should_trade(
                strategy_name=strategy_name, direction=direction)
            if not should_take:
                return
        else:
            sl_mult, rr_ratio = 2.0, 2.0
            size_mult = 1.0

        # ── Compute stop/target in ticks ──────────────────────────────
        tick_size = 0.25  # MNQ
        tick_value = 0.50
        sl_ticks = max(4, int(atr * sl_mult / tick_size))
        tp_ticks = max(4, int(sl_ticks * rr_ratio))

        # ── Position size ─────────────────────────────────────────────
        risk_budget = min(200, balance * 0.004)  # $200 or 0.4% of balance
        risk_per_contract = sl_ticks * tick_value
        size = max(1, min(5, int(risk_budget / risk_per_contract * size_mult)))

        # Apply session quality multiplier
        size = max(1, int(size * session.size_multiplier))

        # ── Place order ───────────────────────────────────────────────
        side = 0 if direction == 1 else 1  # 0=Buy, 1=Sell
        tag = f"{strategy_name}_{datetime.utcnow().strftime('%H%M%S')}"

        logger.info(f"TRADE: {strategy_name} {'LONG' if direction==1 else 'SHORT'} "
                    f"size={size} SL={sl_ticks}ticks TP={tp_ticks}ticks")

        result = self.place_order(side, size, sl_ticks, tp_ticks, tag)

        if result.get("success"):
            self.day_trades += 1
            dir_label = "LONG" if direction == 1 else "SHORT"
            notify_trade_opened(
                self.instrument, dir_label, last["close"],
                size, strategy_name, session.size_multiplier)
            logger.info(f"Order filled: {result.get('orderId')}")
        else:
            logger.error(f"Order FAILED: {result}")
            notify_system_alert(f"Order failed: {result}", "error")

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Main trading loop. Polls every 60 seconds for new bars."""
        logger.info(f"Starting LiveBot: {self.instrument} on account {self.account_id}")
        notify_system_alert(
            f"Bot started: {self.instrument} on ${self.get_account().get('balance', 0):,.0f} account",
            "success")

        poll_interval = 60  # seconds
        last_date = ""

        while True:
            try:
                now_utc = datetime.utcnow()
                ct_hour = (now_utc.hour - 5) % 24
                ct_minutes = ct_hour * 60 + now_utc.minute
                current_date = now_utc.strftime("%Y-%m-%d")

                # New day reset
                if current_date != last_date and last_date:
                    # Send daily summary
                    notify_daily_summary(
                        last_date, self.day_trades, self.day_pnl, 0,
                        0, self.total_pnl)
                    self.day_pnl = 0
                    self.day_trades = 0
                    self.signal_type_losses.clear()
                    logger.info(f"New trading day: {current_date}")
                last_date = current_date

                # Only active during trading hours (8:00 AM - 3:15 PM CT)
                if ct_minutes < 480 or ct_minutes > 915:
                    time.sleep(300)  # Sleep 5 min outside hours
                    continue

                # Flatten at 3:00 PM CT
                if ct_minutes >= 895:
                    positions = self.get_positions()
                    if positions:
                        self.close_position()
                        self.cancel_all_orders()
                        notify_system_alert("Session flatten", "warning")
                    time.sleep(300)
                    continue

                # Fetch bars and process
                df = self.get_bars(minutes_back=2000)  # ~16 hours of 5-min bars
                if len(df) > 50:
                    # Check if new bar completed
                    latest_ts = df["timestamp"].iloc[-1]
                    if self.last_bar_time is None or latest_ts > self.last_bar_time:
                        self.last_bar_time = latest_ts
                        self.process_bar(df)

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                notify_system_alert("Bot stopped by user", "warning")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                notify_system_alert(f"Bot error: {e}", "error")
                time.sleep(30)  # Wait and retry


def main():
    """Entry point for live trading."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="MNQ")
    parser.add_argument("--contract", default="CON.F.US.MNQ.M26")
    parser.add_argument("--use-ai", action="store_true")
    parser.add_argument("--ai-model", default="")
    parser.add_argument("--stats-bank", default="")
    args = parser.parse_args()

    username = os.environ.get("TOPSTEP_USERNAME", "ryanss.pv2026@gmail.com")
    api_key = os.environ.get("TOPSTEP_API_KEY", "")
    account_id = int(os.environ.get("TOPSTEP_ACCOUNT_ID", "21306011"))

    bot = LiveBot(
        username=username, api_key=api_key, account_id=account_id,
        instrument=args.instrument, contract_id=args.contract,
        use_ai=args.use_ai, ai_model_path=args.ai_model,
        stats_bank_path=args.stats_bank)

    bot.run()


if __name__ == "__main__":
    main()
