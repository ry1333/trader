"""TopstepX / ProjectX REST API client."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from loguru import logger

from src.config import TopstepConfig


class TopstepClient:
    """Thin wrapper around the ProjectX Gateway REST API."""

    def __init__(self, cfg: TopstepConfig) -> None:
        self.cfg = cfg
        self._base = cfg.rest_url.rstrip("/")
        self._token: str | None = None
        self._token_ts: float = 0
        self._http = httpx.Client(timeout=30)

    # ── Auth ──────────────────────────────────────────────────────────

    def login(self) -> None:
        resp = self._post("/api/Auth/loginKey", {
            "userName": self.cfg.username,
            "apiKey": self.cfg.api_key,
        })
        self._token = resp["token"]
        self._token_ts = time.time()
        logger.info("Authenticated with TopstepX")

    def _ensure_token(self) -> str:
        if not self._token:
            self.login()
        # Refresh if older than 20 hours (token lasts 24h)
        if time.time() - self._token_ts > 20 * 3600:
            resp = self._post("/api/Auth/validate", {})
            self._token = resp["newToken"]
            self._token_ts = time.time()
            logger.info("Token refreshed")
        return self._token  # type: ignore[return-value]

    # ── Account ───────────────────────────────────────────────────────

    def get_accounts(self, active_only: bool = True) -> list[dict]:
        return self._post("/api/Account/search", {
            "onlyActiveAccounts": active_only,
        })["accounts"]

    # ── Contracts ─────────────────────────────────────────────────────

    def search_contracts(self, text: str, live: bool = True) -> list[dict]:
        return self._post("/api/Contract/search", {
            "searchText": text,
            "live": live,
        })["contracts"]

    # ── Historical Data ───────────────────────────────────────────────

    def get_bars(
        self,
        contract_id: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
        live: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars. Max 20k bars per request."""
        resp = self._post("/api/History/retrieveBars", {
            "contractId": contract_id,
            "live": live,
            "startTime": start.isoformat(),
            "endTime": end.isoformat(),
            "unit": 2,  # Minute
            "unitNumber": interval_minutes,
            "limit": 20000,
            "includePartialBar": False,
        })
        bars = resp.get("bars", [])
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(bars)
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                 "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_bars_bulk(
        self,
        contract_id: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 5,
        live: bool = True,
    ) -> pd.DataFrame:
        """Fetch bars in chunks to handle >20k bar ranges."""
        all_frames: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            df = self.get_bars(contract_id, cursor, end, interval_minutes, live)
            if df.empty:
                break
            all_frames.append(df)
            cursor = df["timestamp"].iloc[-1].to_pydatetime() + pd.Timedelta(minutes=interval_minutes)
            # Respect rate limit: 50 requests per 30 seconds
            time.sleep(0.7)

        if not all_frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.concat(all_frames, ignore_index=True).drop_duplicates("timestamp")

    # ── Orders ────────────────────────────────────────────────────────

    def place_order(
        self,
        account_id: int,
        contract_id: str,
        side: int,  # 0=Buy, 1=Sell
        size: int,
        order_type: int = 2,  # 2=Market
        limit_price: float | None = None,
        stop_price: float | None = None,
        sl_ticks: int | None = None,
        tp_ticks: int | None = None,
        tag: str = "",
    ) -> dict:
        body: dict = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": order_type,
            "side": side,
            "size": size,
            "customTag": tag,
        }
        if limit_price is not None:
            body["limitPrice"] = limit_price
        if stop_price is not None:
            body["stopPrice"] = stop_price
        if sl_ticks is not None:
            body["stopLossBracket"] = {"ticks": sl_ticks, "type": 4}
        if tp_ticks is not None:
            body["takeProfitBracket"] = {"ticks": tp_ticks, "type": 1}
        return self._post("/api/Order/place", body)

    def cancel_order(self, account_id: int, order_id: int) -> dict:
        return self._post("/api/Order/cancel", {
            "accountId": account_id,
            "orderId": order_id,
        })

    def get_open_orders(self, account_id: int) -> list[dict]:
        return self._post("/api/Order/searchOpen", {
            "accountId": account_id,
        }).get("orders", [])

    # ── Positions ─────────────────────────────────────────────────────

    def get_open_positions(self, account_id: int) -> list[dict]:
        return self._post("/api/Position/searchOpen", {
            "accountId": account_id,
        }).get("positions", [])

    def close_position(self, account_id: int, contract_id: str) -> dict:
        return self._post("/api/Position/closeContract", {
            "accountId": account_id,
            "contractId": contract_id,
        })

    # ── HTTP plumbing ─────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._token and "/Auth/" not in path:
            headers["Authorization"] = f"Bearer {self._ensure_token()}"
        resp = self._http.post(f"{self._base}{path}", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
