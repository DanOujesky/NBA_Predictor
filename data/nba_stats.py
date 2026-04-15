import logging
import time

import pandas as pd
from nba_api.stats.endpoints import (
    CommonTeamRoster,
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    LeagueGameLog,
    TeamGameLog,
)

from config import CURRENT_SEASON, NBA_TEAM_IDS, RAW_DIR, REQUEST_DELAY, TEAM_ABBREVIATION_FIXES

logger = logging.getLogger(__name__)


class NBAStatsClient:

    def fetch_team_gamelogs(self, season: str | None = None) -> pd.DataFrame:
        season = season or self._season_string(CURRENT_SEASON)
        frames = []
        for abbr, team_id in NBA_TEAM_IDS.items():
            try:
                log = TeamGameLog(
                    team_id=team_id,
                    season=season,
                    season_type_all_star="Regular Season",
                )
                df = log.get_data_frames()[0]
                df["Team"] = abbr
                frames.append(df)
                time.sleep(REQUEST_DELAY / 3)
            except Exception as exc:
                logger.warning("Failed to fetch game log for %s: %s", abbr, exc)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.to_csv(RAW_DIR / "nba_api_gamelogs.csv", index=False)
        logger.info("Fetched %d game log rows from NBA API", len(result))
        return result

    def fetch_incremental_gamelogs(self, since_date: str) -> pd.DataFrame:
        """Fetch all team games since since_date in a single API call.

        since_date: MM/DD/YYYY format string.
        Returns rows that can be merged into the existing gamelogs CSV.
        """
        season = self._season_string(CURRENT_SEASON)
        try:
            time.sleep(REQUEST_DELAY)
            log = LeagueGameLog(
                league_id="00",
                season=season,
                season_type_all_star="Regular Season",
                date_from_nullable=since_date,
            )
            df = log.get_data_frames()[0]
            if df.empty:
                logger.info("No new games found since %s", since_date)
                return pd.DataFrame()

            df["Team"] = df["TEAM_ABBREVIATION"].map(
                lambda x: TEAM_ABBREVIATION_FIXES.get(str(x).strip(), str(x).strip())
            )
            logger.info("Incremental fetch: %d game rows since %s", len(df), since_date)
            return df
        except Exception as exc:
            logger.error("Incremental gamelog fetch failed: %s", exc)
            return pd.DataFrame()

    def fetch_team_advanced_stats(self, season: str | None = None) -> pd.DataFrame:
        season = season or self._season_string(CURRENT_SEASON)
        try:
            stats = LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            )
            df = stats.get_data_frames()[0]
            df.to_csv(RAW_DIR / "team_advanced.csv", index=False)
            logger.info("Fetched advanced stats for %d teams", len(df))
            return df
        except Exception as exc:
            logger.error("Failed to fetch advanced team stats: %s", exc)
            return pd.DataFrame()

    def fetch_player_stats(self, season: str | None = None) -> pd.DataFrame:
        season = season or self._season_string(CURRENT_SEASON)
        try:
            stats = LeagueDashPlayerStats(
                season=season, per_mode_detailed="PerGame"
            )
            df = stats.get_data_frames()[0]
            df.to_csv(RAW_DIR / "player_stats.csv", index=False)
            logger.info("Fetched stats for %d players", len(df))
            return df
        except Exception as exc:
            logger.error("Failed to fetch player stats: %s", exc)
            return pd.DataFrame()

    def fetch_team_roster(self, team_abbr: str, season: str | None = None) -> pd.DataFrame:
        season = season or self._season_string(CURRENT_SEASON)
        team_id = NBA_TEAM_IDS.get(team_abbr)
        if not team_id:
            logger.error("Unknown team abbreviation: %s", team_abbr)
            return pd.DataFrame()
        try:
            roster = CommonTeamRoster(team_id=team_id, season=season)
            return roster.get_data_frames()[0]
        except Exception as exc:
            logger.error("Failed to fetch roster for %s: %s", team_abbr, exc)
            return pd.DataFrame()

    @staticmethod
    def _season_string(year: int) -> str:
        return f"{year - 1}-{str(year)[-2:]}"
