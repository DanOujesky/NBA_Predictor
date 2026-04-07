"""Cleans raw data and prepares a unified dataset for feature engineering."""

import logging

import numpy as np
import pandas as pd

from config import NBA_TEAM_IDS, PROCESSED_DIR, RAW_DIR, TEAM_NAME_TO_ABBR

logger = logging.getLogger(__name__)


class DataProcessor:
    """Merges, cleans, and normalizes data from both NBA data sources."""

    NUMERIC_COLS = [
        "PTS", "OPP_PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS",
    ]

    def process_nba_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process game logs from the NBA stats API into a standard format."""
        if df.empty:
            return df
        df = df.copy()
        df["Date"] = pd.to_datetime(df["GAME_DATE"])
        df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)
        df["Opponent"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(.+)$")[0].str.strip()
        df["WL"] = (df["WL"] == "W").astype(int)
        rename = {
            "WL": "Win", "PTS": "PTS", "FG_PCT": "FG_PCT", "FG3_PCT": "FG3_PCT",
            "FT_PCT": "FT_PCT", "REB": "REB", "AST": "AST", "STL": "STL",
            "BLK": "BLK", "TOV": "TOV", "PLUS_MINUS": "PLUS_MINUS",
        }
        df = df.rename(columns=rename)
        keep = ["Team", "Opponent", "Date", "is_home", "Win"] + list(rename.values())
        keep = [c for c in dict.fromkeys(keep) if c in df.columns]
        return df[keep].sort_values(["Team", "Date"]).reset_index(drop=True)

    def process_bref_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process game logs from Basketball Reference into a standard format."""
        if df.empty:
            return df
        df = df.copy()
        col_map = {
            "Tm": "PTS", "Opp_1": "OPP_PTS", "FG%": "FG_PCT",
            "AST": "AST", "TOV": "TOV", "Opp": "Opponent",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["is_home"] = np.where(df.get("HomeAway", "") == "@", 0, 1)
        for col in ["PTS", "OPP_PTS", "FG_PCT", "AST", "TOV"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "PTS" in df.columns and "OPP_PTS" in df.columns:
            df["Win"] = (df["PTS"] > df["OPP_PTS"]).astype(int)
        df = df.dropna(subset=["Date", "Team"])
        return df.sort_values(["Team", "Date"]).reset_index(drop=True)

    def merge_sources(self, nba_api_df: pd.DataFrame, bref_df: pd.DataFrame) -> pd.DataFrame:
        """Merge data from both sources, preferring NBA API data when available."""
        if nba_api_df.empty and bref_df.empty:
            raise ValueError("No data available from either source")
        if nba_api_df.empty:
            logger.info("Using Basketball Reference data only")
            return bref_df
        if bref_df.empty:
            logger.info("Using NBA API data only")
            return nba_api_df
        nba_dates = set(nba_api_df["Date"].dt.date)
        bref_extra = bref_df[~bref_df["Date"].dt.date.isin(nba_dates)]
        combined = pd.concat([nba_api_df, bref_extra], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Team", "Date"]).sort_values(["Team", "Date"])
        logger.info("Merged dataset: %d rows", len(combined))
        return combined.reset_index(drop=True)

    def save(self, df: pd.DataFrame, filename: str = "games_clean.csv") -> None:
        """Save processed data to the processed directory."""
        path = PROCESSED_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Saved processed data to %s (%d rows)", path, len(df))
