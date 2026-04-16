"""Source 2: NBA Stats API klient pro stahování herních logů a statistik hráčů.

Komunikuje s oficiálním NBA Stats API (stats.nba.com) přes knihovnu nba_api.
Stahuje:
- TeamGameLog: herní statistiky každého týmu za aktuální sezónu
- LeagueGameLog: inkrementální update pro nové zápasy (od posledního spuštění)
- LeagueDashPlayerStats: per-game statistiky hráčů (PTS, REB, AST, STL, BLK, TOV)
- ScheduleLeagueV2: nadcházející zápasy jako záloha k Basketball Reference
"""

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import (
    CommonTeamRoster,
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    LeagueGameLog,
    ScheduleLeagueV2,
    TeamGameLog,
)

from config import CURRENT_SEASON, NBA_TEAM_IDS, RAW_DIR, REQUEST_DELAY, TEAM_ABBREVIATION_FIXES

logger = logging.getLogger(__name__)


class NBAStatsClient:
    """Klient pro NBA Stats API.

    Všechny metody jsou odolné vůči chybám — při selhání API vrátí prázdný
    DataFrame místo vyhození výjimky, aby pipeline mohla pokračovat.
    """

    def fetch_team_gamelogs(self, season: str | None = None) -> pd.DataFrame:
        """Stáhne herní log každého ze 30 týmů pro danou sezónu.

        Provede 30 separátních volání (jedno na tým) a výsledky sloučí.
        Výsledek uloží do RAW_DIR/nba_api_gamelogs.csv.

        Args:
            season: Řetězec ve formátu "2025-26". Pokud None, použije CURRENT_SEASON.

        Returns:
            DataFrame s jedním řádkem na zápas na tým (tj. každý zápas 2×).
        """
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
        """Stáhne pokročilé týmové statistiky (eFG%, TS%, pace…) z LeagueDashTeamStats.

        Uloží do RAW_DIR/team_advanced.csv. Tato data v aktuální verzi
        modelu nejsou použita jako příznaky, ale jsou k dispozici pro rozšíření.
        """
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
        """Stáhne per-game statistiky všech hráčů z LeagueDashPlayerStats.

        Výsledek uloží do RAW_DIR/player_stats.csv a slouží jako vstup
        pro výpočet hodnoty hráče (player_value) a dostupnosti sestavy.
        """
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
        """Stáhne soupisku týmu z CommonTeamRoster.

        Args:
            team_abbr: Kanonická 3-písmenná zkratka týmu (např. "BOS").
        """
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

    def fetch_upcoming_schedule(self, days_ahead: int = 14) -> pd.DataFrame:
        """Return upcoming games from the NBA API schedule endpoint.

        Returns a DataFrame with columns: Date, Home, Away.
        """
        season = self._season_string(CURRENT_SEASON)
        try:
            time.sleep(REQUEST_DELAY)
            sched = ScheduleLeagueV2(league_id="00", season=season, timeout=30)
            df = sched.get_data_frames()[0]
        except Exception as exc:
            logger.error("ScheduleLeagueV2 fetch failed: %s", exc)
            return pd.DataFrame(columns=["Date", "Home", "Away"])

        df["Date"] = pd.to_datetime(df["gameDate"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        cutoff = today + pd.Timedelta(days=days_ahead)
        upcoming = df[(df["Date"] >= today) & (df["Date"] <= cutoff)].copy()

        if upcoming.empty:
            return pd.DataFrame(columns=["Date", "Home", "Away"])

        upcoming["Home"] = (
            upcoming["homeTeam_teamCity"].fillna("") + " " + upcoming["homeTeam_teamName"].fillna("")
        ).str.strip()
        upcoming["Away"] = (
            upcoming["awayTeam_teamCity"].fillna("") + " " + upcoming["awayTeam_teamName"].fillna("")
        ).str.strip()

        result = upcoming[["Date", "Home", "Away"]]
        result = result[result["Home"].str.strip().ne("") & result["Away"].str.strip().ne("")]
        result = result.drop_duplicates().reset_index(drop=True)
        logger.info("NBA API schedule: %d upcoming games (next %d days)", len(result), days_ahead)
        return result

    @staticmethod
    def _season_string(year: int) -> str:
        """Převede rok konce sezóny na formát NBA API (např. 2026 → '2025-26')."""
        return f"{year - 1}-{str(year)[-2:]}"
