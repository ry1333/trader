"""Live BTC trading bot for TopstepX — Multi-TF Trend Pullback + US ORB.

Architecture:
1. Polls 5-min bars from TopstepX API every 60 seconds
2. Resamples to 15-min bars internally
3. Computes 4h bias + 1h tactical + 15m signals
4. Places orders with bracket stops (SL + TP)
5. Monitors positions for Chandelier trail exits (3 ATR after 1R)
6. Trading windows: London open (2:30-5 AM CT) + US morning (7:30-10:30 AM CT)
7. Flattens before CME maintenance (2:55 PM CT)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from src.notifications.discord import (
    notify_system_alert,
    notify_trade_opened,
)
from src.strategy.btc_signals import (
    BTCSignal,
    BTCSignalType,
    compute_btc_features,
    generate_btc_signals,
)

CT = ZoneInfo("US/Central")
ET = ZoneInfo("US/Eastern")

TICK_SIZE = 5.0
TICK_VALUE = 0.50
MAX_POSITION = 3


class LiveBTCBot:
    """Live BTC trading bot — 15-min multi-TF trend pullback + US ORB."""

    def __init__(
        self,
        username: str,
        api_key: str,
        account_id: int,
        contract_id: str = "CON.F.US.MBT.M26",
    ):
        self.username = username
        self.api_key = api_key
        self.account_id = account_id
        self.contract_id = contract_id
        self.base_url = "https://api.topstepx.com"
        self.token: str | None = None
        self.token_time: float = 0

        # State
        self.day_pnl: float = 0.0
        self.day_trades: int = 0
        self.session_start_balance: float = 0.0
        self.last_bar_time: datetime | None = None
        self.consecutive_losses: int = 0

        # Active trade tracking (for Chandelier trail)
        self.active_entry_price: float = 0.0
        self.active_direction: int = 0
        self.active_peak_profit: float = 0.0
        self.active_sl_price: float = 0.0
        self.active_entry_time: datetime | None = None
        self.active_signal_type: int = 0

    # ── Time Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _now_ct() -> datetime:
        return datetime.now(timezone.utc).astimezone(CT)

    @staticmethod
    def _now_et() -> datetime:
        return datetime.now(timezone.utc).astimezone(ET)

    # ── API Methods ───────────────────────────────────────────────────

    def _ensure_token(self) -> str:
        if self.token and (time.time() - self.token_time) < 12 * 3600:
            return self.token
        try:
            resp = httpx.post(f"{self.base_url}/api/Auth/loginKey", json={
                "userName": self.username, "apiKey": self.api_key,
            }, timeout=15)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise RuntimeError(f"Auth failed: {e}")
        if not data.get("success"):
            raise RuntimeError(f"Auth failed: {data}")
        self.token = data["token"]
        self.token_time = time.time()
        logger.info("BTC: Authenticated with TopstepX")
        return self.token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, body: dict) -> dict:
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=body,
                              headers=self._headers(), timeout=15)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"BTC API HTTP {e.response.status_code}: {path}")
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except (httpx.HTTPError, ValueError) as e:
            logger.error(f"BTC API error: {path} → {e}")
            return {"success": False, "error": str(e)}

    def get_account(self) -> dict:
        data = self._post("/api/Account/search", {"onlyActiveAccounts": True})
        accounts = data.get("accounts", [])
        for a in accounts:
            if a["id"] == self.account_id:
                return a
        return accounts[0] if accounts else {}

    def get_bars_15m(self, minutes_back: int = 50000) -> pd.DataFrame:
        """Fetch 15-min bars directly from API. ~35 days at 5000 bar limit."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=minutes_back)
        data = self._post("/api/History/retrieveBars", {
            "contractId": self.contract_id,
            "live": False,
            "startTime": start.isoformat(),
            "endTime": now.isoformat(),
            "unit": 2,  # Minute
            "unitNumber": 15,
            "limit": 5000,
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

    def get_positions_for_contract(self) -> list[dict]:
        data = self._post("/api/Position/searchOpen", {"accountId": self.account_id})
        all_pos = data.get("positions", [])
        return [p for p in all_pos if p.get("contractId") == self.contract_id]

    def get_all_positions(self) -> list[dict]:
        data = self._post("/api/Position/searchOpen", {"accountId": self.account_id})
        return data.get("positions", [])

    def place_order(self, side: int, size: int, sl_ticks: int, tp_ticks: int,
                    tag: str = "") -> dict:
        body = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "type": 2, "side": side, "size": size,
            "customTag": tag,
        }
        if sl_ticks > 0:
            body["stopLossBracket"] = {"ticks": sl_ticks, "type": 4}
        if tp_ticks > 0:
            body["takeProfitBracket"] = {"ticks": tp_ticks, "type": 1}
        data = self._post("/api/Order/place", body)
        logger.info(f"BTC order: side={side} size={size} SL={sl_ticks} TP={tp_ticks} "
                    f"tag={tag} → {data}")
        return data

    def close_position(self) -> dict:
        data = self._post("/api/Position/closeContract", {
            "accountId": self.account_id, "contractId": self.contract_id})
        logger.info(f"BTC position closed: {data}")
        self.active_direction = 0
        self.active_peak_profit = 0
        return data

    def cancel_all_orders(self) -> None:
        orders = self._post("/api/Order/searchOpen", {"accountId": self.account_id})
        for o in orders.get("orders", []):
            self._post("/api/Order/cancel", {
                "accountId": self.account_id, "orderId": o["id"]})

    def _update_day_pnl(self) -> float:
        account = self.get_account()
        balance = account.get("balance", 0)
        if self.session_start_balance > 0:
            self.day_pnl = balance - self.session_start_balance
        return self.day_pnl

    # ── Chandelier Trail Management ───────────────────────────────────

    def _check_chandelier_exit(self, current_price: float, atr: float) -> bool:
        """Check if we should exit via Chandelier trail. Returns True if exit needed."""
        if self.active_direction == 0 or atr <= 0:
            return False

        current_pnl = (current_price - self.active_entry_price) * self.active_direction
        stop_dist = abs(self.active_entry_price - self.active_sl_price)

        # Track peak profit
        if current_pnl > self.active_peak_profit:
            self.active_peak_profit = current_pnl

        r_mult = self.active_peak_profit / stop_dist if stop_dist > 0 else 0

        # Only trail after 1R profit
        if r_mult >= 1.0:
            # Breakeven: pull back to entry
            if current_pnl <= 0:
                logger.info(f"BTC: Breakeven exit (was up {r_mult:.1f}R, now flat)")
                return True

            # Chandelier: 3 ATR from highest high
            if self.active_peak_profit >= atr:
                highest = self.active_entry_price + self.active_peak_profit * self.active_direction
                chandelier = highest - atr * 3.0 * self.active_direction
                if (self.active_direction == 1 and current_price < chandelier) or \
                   (self.active_direction == -1 and current_price > chandelier):
                    logger.info(f"BTC: Chandelier exit at {current_price:.0f} "
                               f"(peak {self.active_peak_profit:.0f}, trail {chandelier:.0f})")
                    return True

        # Time decay: 2 hours (~8 bars of 15-min) with no profit
        if self.active_entry_time:
            elapsed = (datetime.now(timezone.utc) - self.active_entry_time).total_seconds() / 3600
            if elapsed >= 2.0 and current_pnl <= 0:
                logger.info(f"BTC: Time decay exit ({elapsed:.1f}h, pnl={current_pnl:.0f})")
                return True

        return False

    # ── Trading Logic ─────────────────────────────────────────────────

    def process_bars(self, df_15m: pd.DataFrame) -> None:
        """Process 15-min bars. Core BTC trading logic."""
        if len(df_15m) < 60:
            return

        et = self._now_et()
        et_min = et.hour * 60 + et.minute
        ct = self._now_ct()
        ct_min = ct.hour * 60 + ct.minute

        # ── Flatten before CME maintenance (2:55 PM CT = 3:55 PM ET) ──
        if et_min >= 955:
            positions = self.get_positions_for_contract()
            if positions:
                logger.warning("BTC: Flatten before CME maintenance")
                self.close_position()
                self.cancel_all_orders()
                notify_system_alert("BTC positions flattened (maintenance)", "warning")
            return

        # ── Check active position for Chandelier exit ─────────────────
        positions = self.get_positions_for_contract()
        if positions and self.active_direction != 0:
            # Get current price from latest bar
            current_price = df_15m.iloc[-1]["close"]
            atr = df_15m.iloc[-1].get("btc_atr14", 0) if "btc_atr14" in df_15m.columns else 0

            if not pd.isna(atr) and atr > 0:
                if self._check_chandelier_exit(current_price, atr):
                    self.close_position()
                    self.cancel_all_orders()
                    notify_system_alert("BTC Chandelier trail exit", "warning")
                    return
            return  # Still in a trade, don't enter new one

        if positions and self.active_direction == 0:
            return  # Position from bracket, let it run

        # ── Daily loss check ──────────────────────────────────────────
        self._update_day_pnl()
        if self.day_pnl <= -1000:
            return
        if self.consecutive_losses >= 3:
            return

        # ── Portfolio heat: max 2 across all instruments ──────────────
        all_pos = self.get_all_positions()
        if len(all_pos) >= 2:
            return

        # ── Compute features + signals on 15-min bars ─────────────────
        df = compute_btc_features(df_15m)
        df["signal"], df["signal_type"] = generate_btc_signals(df)

        last = df.iloc[-1]
        if last["signal"] == BTCSignal.FLAT:
            return

        direction = 1 if last["signal"] == BTCSignal.LONG else -1
        sig_type = int(last["signal_type"])
        strategy_name = BTCSignalType(sig_type).name

        # ── ATR check ─────────────────────────────────────────────────
        atr = last.get("btc_atr14", 0)
        if pd.isna(atr) or atr <= 0:
            return

        # ── Compute stops (ATR-based, Variant D) ─────────────────────
        sl_mult = 2.0
        rr = 4.0  # Wide target — Chandelier handles exits
        entry_price = last["close"]
        sl_ticks = max(4, int(atr * sl_mult / TICK_SIZE))
        tp_ticks = max(4, int(sl_ticks * rr))

        # ── Position size ─────────────────────────────────────────────
        account = self.get_account()
        balance = account.get("balance", 50000)
        risk_budget = min(200, balance * 0.004)
        risk_per_contract = sl_ticks * TICK_VALUE
        size = max(1, min(MAX_POSITION, int(risk_budget / risk_per_contract)))

        if self.day_pnl <= -500:
            size = 1  # Tier1 loss: min size

        # Caution mode
        htf_caution = last.get("htf_caution", 0)
        if htf_caution and size > 1:
            size = max(1, size // 2)

        # ── Place order ───────────────────────────────────────────────
        side = 0 if direction == 1 else 1
        tag = f"BTC_{strategy_name}_{ct.strftime('%H%M%S')}"

        logger.info(f"BTC TRADE: {strategy_name} {'LONG' if direction==1 else 'SHORT'} "
                    f"size={size} SL={sl_ticks}ticks TP={tp_ticks}ticks "
                    f"entry~{entry_price:.0f}")

        result = self.place_order(side, size, sl_ticks, tp_ticks, tag)

        if result.get("success"):
            self.day_trades += 1
            self.active_entry_price = entry_price
            self.active_direction = direction
            self.active_peak_profit = 0.0
            self.active_sl_price = entry_price - sl_ticks * TICK_SIZE * direction
            self.active_entry_time = datetime.now(timezone.utc)
            self.active_signal_type = sig_type

            dir_label = "LONG" if direction == 1 else "SHORT"
            notify_trade_opened("MBT", dir_label, entry_price, size, strategy_name, 1.0)
            logger.info(f"BTC order filled: {result.get('orderId')}")
        else:
            logger.error(f"BTC order FAILED: {result}")
            notify_system_alert(f"BTC order failed: {result}", "error")

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop. Polls every 60 seconds."""
        logger.info("Starting BTC LiveBot")

        account = self.get_account()
        self.session_start_balance = account.get("balance", 50000)
        logger.info(f"BTC: Session start balance ${self.session_start_balance:,.0f}")
        notify_system_alert(
            f"BTC bot started on ${self.session_start_balance:,.0f} account", "success")

        poll_interval = 60
        last_date = ""

        while True:
            try:
                ct = self._now_ct()
                et = self._now_et()
                ct_min = ct.hour * 60 + ct.minute
                et_min = et.hour * 60 + et.minute
                current_date = ct.strftime("%Y-%m-%d")

                # New day reset
                if current_date != last_date and last_date:
                    self._update_day_pnl()
                    self.day_pnl = 0
                    self.day_trades = 0
                    self.consecutive_losses = 0
                    account = self.get_account()
                    self.session_start_balance = account.get("balance",
                                                             self.session_start_balance)
                    logger.info(f"BTC: New day {current_date}, "
                               f"balance ${self.session_start_balance:,.0f}")
                last_date = current_date

                # BTC trading windows (ET):
                # London: 3:00-6:00 AM ET (180-360)
                # US morning: 8:00-11:30 AM ET (480-690)
                # Also check positions during all active hours for Chandelier exits
                in_trading_window = (180 <= et_min < 360) or (480 <= et_min < 690)
                in_active_hours = (180 <= et_min < 960)  # 3 AM - 4 PM ET

                # CME maintenance flatten
                if 955 <= et_min < 1080:
                    positions = self.get_positions_for_contract()
                    if positions:
                        self.close_position()
                        self.cancel_all_orders()
                        notify_system_alert("BTC maintenance flatten", "warning")
                    time.sleep(300)
                    continue

                # Sleep outside active hours
                if not in_active_hours:
                    time.sleep(300)
                    continue

                # Fetch and process bars
                df_15m = self.get_bars_15m()  # 15-min bars directly, ~35 days
                if len(df_15m) > 30:
                    latest_ts = df_15m["timestamp"].iloc[-1]
                    if self.last_bar_time is None or latest_ts > self.last_bar_time:
                        self.last_bar_time = latest_ts
                        logger.info(f"BTC: New 15m bar {latest_ts}, "
                                   f"{len(df_15m)} bars, "
                                   f"close={df_15m.iloc[-1]['close']:.0f}, "
                                   f"window={'YES' if in_trading_window else 'NO'}")

                        # Always check Chandelier exits on active positions
                        # Only enter new trades during trading windows
                        if in_trading_window or self.active_direction != 0:
                            self.process_bars(df_15m)
                elif len(df_15m) == 0:
                    logger.warning("BTC: No bars returned from API")

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("BTC bot stopped by user")
                notify_system_alert("BTC bot stopped", "warning")
                break
            except Exception as e:
                logger.error(f"BTC main loop error: {e}")
                notify_system_alert(f"BTC error: {e}", "error")
                time.sleep(30)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="CON.F.US.MBT.M26")
    args = parser.parse_args()

    username = os.environ.get("TOPSTEP_USERNAME", "ryanss.pv2026@gmail.com")
    api_key = os.environ.get("TOPSTEP_API_KEY", "")
    account_id = int(os.environ.get("TOPSTEP_ACCOUNT_ID", "21306011"))

    bot = LiveBTCBot(
        username=username, api_key=api_key,
        account_id=account_id, contract_id=args.contract)
    bot.run()


if __name__ == "__main__":
    main()
