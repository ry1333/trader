"""Economic news calendar — fetches real event dates for trade filtering.

Two modes:
1. Live: fetches this/next week from ForexFactory API
2. Backtest: uses known recurring US macro schedule with historical dates

Filters trades around high-impact USD events:
- FOMC decisions (2:00 PM ET, 8 times/year)
- CPI (8:30 AM ET, monthly)
- NFP / Employment (8:30 AM ET, first Friday)
- PPI (8:30 AM ET, monthly)
- Retail Sales (8:30 AM ET, monthly)
- GDP (8:30 AM ET, quarterly)
- PCE (8:30 AM ET, monthly)
- ISM Manufacturing (10:00 AM ET, monthly)
- Fed Chair speaks (variable)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# Cache directory for downloaded calendars
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache" / "news"


def fetch_live_calendar() -> list[dict]:
    """Fetch this week's economic calendar from ForexFactory feed."""
    import httpx

    events = []
    for url in [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
    ]:
        try:
            resp = httpx.get(url, timeout=15)
            if resp.status_code == 200:
                for e in resp.json():
                    if e.get("country") == "USD" and e.get("impact") in ("High", "Medium"):
                        events.append({
                            "datetime": e["date"],
                            "title": e["title"],
                            "impact": e["impact"],
                        })
        except Exception as ex:
            logger.warning(f"Failed to fetch calendar: {ex}")

    logger.info(f"Fetched {len(events)} USD High/Medium impact events")
    return events


def build_historical_calendar(start_year: int = 2023, end_year: int = 2026) -> pd.DataFrame:
    """Build historical US macro event calendar from known schedules.

    This uses the known patterns of recurring US economic releases.
    Not perfectly exact dates but within 1-2 days of actual release.
    """
    events = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 4:
                break

            # ── NFP: First Friday of month, 8:30 AM ET ────────────────
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            nfp_date = first_day + timedelta(days=days_until_friday)
            events.append({
                "date": nfp_date.strftime("%Y-%m-%d"),
                "time_et": "08:30",
                "event": "NFP",
                "impact": 3,
                "buffer_minutes": 30,
            })

            # ── CPI: ~10th-13th of month, 8:30 AM ET ─────────────────
            cpi_day = 12 if month % 2 == 0 else 13
            try:
                cpi_date = datetime(year, month, min(cpi_day, 28))
                # Skip weekends
                while cpi_date.weekday() >= 5:
                    cpi_date += timedelta(days=1)
                events.append({
                    "date": cpi_date.strftime("%Y-%m-%d"),
                    "time_et": "08:30",
                    "event": "CPI",
                    "impact": 3,
                    "buffer_minutes": 30,
                })
            except ValueError:
                pass

            # ── PPI: ~14th-16th of month, 8:30 AM ET ─────────────────
            ppi_day = 14
            try:
                ppi_date = datetime(year, month, min(ppi_day, 28))
                while ppi_date.weekday() >= 5:
                    ppi_date += timedelta(days=1)
                events.append({
                    "date": ppi_date.strftime("%Y-%m-%d"),
                    "time_et": "08:30",
                    "event": "PPI",
                    "impact": 2,
                    "buffer_minutes": 20,
                })
            except ValueError:
                pass

            # ── Retail Sales: ~15th-17th, 8:30 AM ET ──────────────────
            try:
                retail_date = datetime(year, month, 16)
                while retail_date.weekday() >= 5:
                    retail_date += timedelta(days=1)
                events.append({
                    "date": retail_date.strftime("%Y-%m-%d"),
                    "time_et": "08:30",
                    "event": "Retail Sales",
                    "impact": 2,
                    "buffer_minutes": 20,
                })
            except ValueError:
                pass

            # ── ISM Manufacturing: First business day, 10:00 AM ET ────
            ism_date = datetime(year, month, 1)
            while ism_date.weekday() >= 5:
                ism_date += timedelta(days=1)
            events.append({
                "date": ism_date.strftime("%Y-%m-%d"),
                "time_et": "10:00",
                "event": "ISM Manufacturing",
                "impact": 2,
                "buffer_minutes": 15,
            })

            # ── PCE: Last Friday of month, 8:30 AM ET ─────────────────
            if month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                next_month = month + 1 if month < 12 else 1
                next_year = year if month < 12 else year + 1
                try:
                    last_day = datetime(next_year, next_month, 1) - timedelta(days=1)
                    while last_day.weekday() != 4:  # Friday
                        last_day -= timedelta(days=1)
                    events.append({
                        "date": last_day.strftime("%Y-%m-%d"),
                        "time_et": "08:30",
                        "event": "PCE",
                        "impact": 2,
                        "buffer_minutes": 20,
                    })
                except ValueError:
                    pass

        # ── FOMC: 8 meetings per year, 2:00 PM ET ────────────────────
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        for fm in fomc_months:
            # FOMC typically on Wednesday in 3rd or 4th week
            fomc_date = datetime(year, fm, 15)
            while fomc_date.weekday() != 2:  # Wednesday
                fomc_date += timedelta(days=1)
            events.append({
                "date": fomc_date.strftime("%Y-%m-%d"),
                "time_et": "14:00",
                "event": "FOMC",
                "impact": 3,
                "buffer_minutes": 45,
            })

        # ── GDP: Quarterly, ~last week of Jan/Apr/Jul/Oct, 8:30 AM ET
        for gdp_month in [1, 4, 7, 10]:
            try:
                gdp_date = datetime(year, gdp_month, 26)
                while gdp_date.weekday() >= 5:
                    gdp_date += timedelta(days=1)
                events.append({
                    "date": gdp_date.strftime("%Y-%m-%d"),
                    "time_et": "08:30",
                    "event": "GDP",
                    "impact": 2,
                    "buffer_minutes": 20,
                })
            except ValueError:
                pass

    df = pd.DataFrame(events)
    df["datetime_et"] = pd.to_datetime(df["date"] + " " + df["time_et"])
    df = df.sort_values("datetime_et").reset_index(drop=True)
    logger.info(f"Built historical calendar: {len(df)} events ({start_year}-{end_year})")
    return df


class NewsFilter:
    """Checks whether a given timestamp is within a news no-trade window."""

    def __init__(self, calendar_df: pd.DataFrame | None = None):
        if calendar_df is None:
            calendar_df = build_historical_calendar()
        self.calendar = calendar_df
        self._build_windows()

    def _build_windows(self) -> None:
        """Pre-compute no-trade windows as (start, end) tuples."""
        self.windows: list[tuple[pd.Timestamp, pd.Timestamp, str, int]] = []

        for _, row in self.calendar.iterrows():
            event_time = pd.Timestamp(row["datetime_et"], tz="US/Eastern")
            buffer = int(row["buffer_minutes"])
            impact = int(row["impact"])

            # No-trade window: buffer before and after event
            start = event_time - pd.Timedelta(minutes=buffer)
            end = event_time + pd.Timedelta(minutes=buffer)
            self.windows.append((start, end, row["event"], impact))

        logger.info(f"NewsFilter: {len(self.windows)} no-trade windows")

    def is_blocked(self, timestamp: pd.Timestamp) -> tuple[bool, str]:
        """Check if trading is blocked at this timestamp.

        Returns (is_blocked, event_name).
        """
        # Convert to Eastern for comparison
        if timestamp.tzinfo is not None:
            ts_et = timestamp.tz_convert("US/Eastern")
        else:
            ts_et = timestamp

        for start, end, event, impact in self.windows:
            if start <= ts_et <= end:
                return True, event

        return False, ""

    def get_impact_at(self, timestamp: pd.Timestamp) -> int:
        """Get the impact level of any active news event. 0 = no news."""
        if timestamp.tzinfo is not None:
            ts_et = timestamp.tz_convert("US/Eastern")
        else:
            ts_et = timestamp

        for start, end, event, impact in self.windows:
            if start <= ts_et <= end:
                return impact
        return 0
