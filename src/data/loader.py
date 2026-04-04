"""Load and cache historical bar data as Parquet files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import Settings
from src.data.client import TopstepClient

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


def load_bars(
    settings: Settings,
    contract_id: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load 5-minute bars, pulling from cache or API."""
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{contract_id}_{start:%Y%m%d}_{end:%Y%m%d}_5m.parquet"

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached bars from {cache_file.name}")
        return pd.read_parquet(cache_file)

    client = TopstepClient(settings.topstep)
    client.login()
    df = client.get_bars_bulk(contract_id, start, end, settings.strategy.bar_interval_minutes)

    if not df.empty and use_cache:
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached {len(df)} bars → {cache_file.name}")

    return df


def load_bars_csv(path: str | Path) -> pd.DataFrame:
    """Load bars from a local CSV file (for external data sources)."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    return df.sort_values("timestamp").reset_index(drop=True)
