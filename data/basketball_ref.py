"""Source 1: Basketball Reference scraper for historical game logs."""

import logging
import time
from io import StringIO
from pathlib import Path

import cloudscraper
import pandas as pd

from config import (
    CURRENT_SEASON, MAX_RETRIES, RAW_DIR,
    REQUEST_DELAY, REQUEST_TIMEOUT, SEASONS_BACK, TEAMS,
)

logger = logging.getLogger(__name__)


class BasketballRefScraper:
    """Scrapes team game logs from basketball-reference.com."""

    GAMELOG_URL = "https://www.basketball-reference.com/teams/{team}/{year}/gamelog/"
    SCHEDULE_URL = (
        "https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
    )

    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
        self.output_path = RAW_DIR / "bref_gamelogs.csv"

    def fetch_page(self, url: str) -> str | None:
        """Fetch a web page with retry logic and rate-limit handling."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.scraper.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 404:
                    logger.warning("Page not found: %s", url)
                    return None
                if resp.status_code == 429:
                    logger.warning("Rate limited, sleeping 120s")
                    time.sleep(120)
                    continue
                logger.error("HTTP %d for %s", resp.status_code, url)
            except Exception as exc:
                logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(REQUEST_DELAY * (attempt + 2))
        return None

    def parse_gamelog(self, html: str, team: str, season: int) -> pd.DataFrame | None:
        """Parse a team game log HTML table into a DataFrame."""
        try:
            tables = pd.read_html(StringIO(html), header=[0, 1], match="Date")
            if not tables:
                return None
            df = tables[0].copy()
            flat_cols = []
            for i, col in enumerate(df.columns):
                flat_cols.append("HomeAway" if i == 3 else col[1])
            df.columns = self._deduplicate_columns(flat_cols)
            df = df[df["Date"] != "Date"].copy()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df["Team"] = team
            df["Season"] = season
            return df.reset_index(drop=True)
        except Exception as exc:
            logger.error("Parse error %s-%d: %s", team, season, exc)
            return None

    def scrape_gamelogs(self) -> pd.DataFrame:
        """Scrape game logs for all teams across configured seasons."""
        completed = self._load_completed()
        start_year = CURRENT_SEASON - SEASONS_BACK
        tasks = [
            (t, y)
            for y in range(start_year, CURRENT_SEASON + 1)
            for t in TEAMS
            if (t, y) not in completed
        ]
        if not tasks:
            logger.info("All game log data is up to date")
            return self._load_existing()

        logger.info("Fetching %d team-season game logs", len(tasks))
        for i, (team, year) in enumerate(tasks):
            logger.info("[%d/%d] %s %d", i + 1, len(tasks), team, year)
            html = self.fetch_page(self.GAMELOG_URL.format(team=team, year=year))
            if html:
                df = self.parse_gamelog(html, team, year)
                if df is not None and not df.empty:
                    header = not self.output_path.exists()
                    df.to_csv(self.output_path, mode="a", index=False, header=header)
                    logger.info("Saved %d rows for %s-%d", len(df), team, year)
            time.sleep(REQUEST_DELAY)

        return self._load_existing()

    def scrape_schedule(self) -> pd.DataFrame:
        """Scrape upcoming NBA schedule from basketball-reference.com."""
        months = [
            "october", "november", "december", "january",
            "february", "march", "april", "may", "june",
        ]
        all_games = []
        for month in months:
            url = self.SCHEDULE_URL.format(year=CURRENT_SEASON, month=month)
            html = self.fetch_page(url)
            if not html:
                continue
            try:
                tables = pd.read_html(StringIO(html))
                df = next((t for t in tables if "Date" in t.columns), None)
                if df is None:
                    continue
                df = df.rename(columns={"Visitor/Neutral": "Away", "Home/Neutral": "Home"})
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])
                future = df[df["Date"] >= pd.Timestamp.today().normalize()]
                if not future.empty:
                    all_games.append(future[["Date", "Home", "Away"]])
            except Exception as exc:
                logger.warning("Schedule parse error for %s: %s", month, exc)
            time.sleep(REQUEST_DELAY)

        if all_games:
            result = pd.concat(all_games).drop_duplicates().reset_index(drop=True)
            result.to_csv(RAW_DIR / "schedule.csv", index=False)
            return result
        return pd.DataFrame(columns=["Date", "Home", "Away"])

    def _load_completed(self) -> set:
        if not self.output_path.exists():
            return set()
        try:
            df = pd.read_csv(self.output_path, usecols=["Team", "Season"])
            return set(zip(df["Team"].astype(str), df["Season"].astype(int)))
        except Exception:
            return set()

    def _load_existing(self) -> pd.DataFrame:
        if self.output_path.exists():
            return pd.read_csv(self.output_path)
        return pd.DataFrame()

    @staticmethod
    def _deduplicate_columns(cols: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        result = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                result.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                result.append(c)
        return result
