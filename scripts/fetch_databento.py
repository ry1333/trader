"""Fetch historical ES futures data from Databento.

Usage:
    DATABENTO_API_KEY=your_key python scripts/fetch_databento.py

Sign up at https://databento.com for $50 free credit.
Get your API key from https://databento.com/portal/keys
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd
from loguru import logger

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def fetch_es_bars(
    api_key: str,
    start_date: str = "2025-04-01",
    end_date: str = "2026-04-01",
    symbol: str = "ES.FUT",  # Continuous front-month ES
    interval: str = "5min",
) -> pd.DataFrame:
    """Fetch 5-minute bars for ES futures from Databento."""

    client = db.Historical(api_key)

    logger.info(f"Fetching {symbol} {interval} bars: {start_date} → {end_date}")
    logger.info("This may take a minute for large date ranges...")

    # Get cost estimate first
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",  # CME Globex
        symbols=[symbol],
        schema="ohlcv-1m",  # 1-min bars (we'll resample to 5min)
        start=start_date,
        end=end_date,
    )
    logger.info(f"Estimated cost: ${cost:.2f}")

    # Fetch 1-minute bars (more granular, we resample)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        schema="ohlcv-1m",
        start=start_date,
        end=end_date,
    )

    df = data.to_df()
    logger.info(f"Downloaded {len(df)} 1-minute bars")

    # Resample to 5-minute
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Resample to 5-min
    df = df.set_index("timestamp")
    df_5m = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    df_5m = df_5m.reset_index()

    logger.info(f"Resampled to {len(df_5m)} 5-minute bars")
    return df_5m


def main() -> None:
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        logger.error("Set DATABENTO_API_KEY environment variable")
        logger.info("Sign up at https://databento.com ($50 free credit)")
        logger.info("Get key at https://databento.com/portal/keys")
        return

    CACHE_DIR.mkdir(exist_ok=True)

    # Fetch 12 months of ES data
    df = fetch_es_bars(
        api_key=api_key,
        start_date="2025-04-01",
        end_date="2026-04-01",
        symbol="ES.FUT",
    )

    # Save
    out_path = CACHE_DIR / "es_5m_12mo_databento.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} bars → {out_path}")
    logger.info(f"Date range: {df.timestamp.iloc[0]} → {df.timestamp.iloc[-1]}")
    logger.info(f"Price range: ${df.close.min():.2f} - ${df.close.max():.2f}")


if __name__ == "__main__":
    main()
