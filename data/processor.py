"""Cleans raw data and prepares a unified dataset for feature engineering."""

import logging

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR,
    RAW_DIR,
    TEAM_ABBREVIATION_FIXES,
    TEAM_NAME_TO_ABBR,
)

logger = logging.getLogger(__name__)


def _normalize_abbr(value: str) -> str:
    """Map any team abbreviation or full name to our canonical 3-letter form."""
    v = str(value).strip()
    fixed = TEAM_ABBREVIATION_FIXES.get(v)
    if fixed:
        return fixed
    from_name = TEAM_NAME_TO_ABBR.get(v)
    if from_name:
        return from_name
    return v


class DataProcessor:
    """Merges, cleans, and normalises data from both NBA data sources."""

    # Columns coerced to numeric in both sources
    NUMERIC_COLS = [
        "PTS", "OPP_PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS",
    ]

    def process_nba_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process game logs from the NBA Stats API into a standard format.

        NOTE: The NBA API MATCHUP field uses abbreviations that differ from
        our canonical set (e.g. BKN vs BRK). Both Team and Opponent are
        normalised here. OPP_PTS is NOT available from the API; it is derived
        later in merge_sources via a self-join on (Date, Opponent).
        """
        if df.empty:
            return df
        df = df.copy()
        df["Date"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)
        df["Opponent"] = (
            df["MATCHUP"]
            .str.extract(r"(?:vs\.|@)\s*(.+)$")[0]
            .str.strip()
            .map(_normalize_abbr)
        )
        df["WL"] = (df["WL"] == "W").astype(int)
        rename = {
            "WL": "Win", "PTS": "PTS", "FG_PCT": "FG_PCT", "FG3_PCT": "FG3_PCT",
            "FT_PCT": "FT_PCT", "REB": "REB", "AST": "AST", "STL": "STL",
            "BLK": "BLK", "TOV": "TOV", "PLUS_MINUS": "PLUS_MINUS",
        }
        df = df.rename(columns=rename)
        df["Team"] = df["Team"].map(_normalize_abbr)

        keep = ["Team", "Opponent", "Date", "is_home", "Win"] + [
            v for v in rename.values() if v not in ("Win",)
        ]
        keep = list(dict.fromkeys(c for c in keep if c in df.columns))
        return df[keep].sort_values(["Team", "Date"]).reset_index(drop=True)

    def process_bref_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process game logs from Basketball Reference into a standard format.

        Basketball Reference game logs contain columns at level-1 of the
        multi-level header. Key mappings:
          Tm      → PTS   (team points)
          Opp_1   → OPP_PTS (opponent points, second "Opp" after dedup)
          TRB     → REB   (total rebounds — critical for rolling features)
          FG%     → FG_PCT
          3P%     → FG3_PCT
          FT%     → FT_PCT
          Opp     → Opponent (opponent team abbreviation)
        """
        if df.empty:
            return df
        df = df.copy()
        col_map = {
            "Tm": "PTS",
            "Opp_1": "OPP_PTS",
            "FG%": "FG_PCT",
            "3P%": "FG3_PCT",
            "FT%": "FT_PCT",
            "TRB": "REB",
            "AST": "AST",
            "STL": "STL",
            "BLK": "BLK",
            "TOV": "TOV",
            "Opp": "Opponent",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["is_home"] = np.where(df.get("HomeAway", "") == "@", 0, 1)

        for col in ["PTS", "OPP_PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "STL", "BLK", "TOV"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "PTS" in df.columns and "OPP_PTS" in df.columns:
            df["Win"] = (df["PTS"] > df["OPP_PTS"]).astype(int)

        if "Team" in df.columns:
            df["Team"] = df["Team"].map(_normalize_abbr)
        if "Opponent" in df.columns:
            df["Opponent"] = df["Opponent"].map(_normalize_abbr)

        df = df.dropna(subset=["Date", "Team"])
        return df.sort_values(["Team", "Date"]).reset_index(drop=True)

    def merge_sources(self, nba_api_df: pd.DataFrame, bref_df: pd.DataFrame) -> pd.DataFrame:
        """Merge both sources, preferring NBA API data when dates overlap.

        After combining, OPP_PTS is derived for rows where it is missing by
        joining each team's PTS to its opponent's row on the same date. This
        ensures roll_OPP_PTS is computable for all seasons regardless of source.
        """
        if nba_api_df.empty and bref_df.empty:
            raise ValueError("No data available from either source")
        if nba_api_df.empty:
            logger.info("Using Basketball Reference data only")
            combined = bref_df
        elif bref_df.empty:
            logger.info("Using NBA API data only")
            combined = nba_api_df
        else:
            nba_dates = set(nba_api_df["Date"].dt.date)
            bref_extra = bref_df[~bref_df["Date"].dt.date.isin(nba_dates)]
            combined = pd.concat([nba_api_df, bref_extra], ignore_index=True)

        # Deduplicate before the OPP_PTS self-join to guarantee unique (Team, Date) keys
        combined = (
            combined
            .drop_duplicates(subset=["Team", "Date"])
            .sort_values(["Team", "Date"])
            .reset_index(drop=True)
        )

        # Derive OPP_PTS: for a game where Team=A played Opponent=B,
        # the opponent's score = B's PTS on the same date.
        if "OPP_PTS" not in combined.columns:
            combined["OPP_PTS"] = float("nan")

        pts_ref = (
            combined[["Team", "Date", "PTS"]]
            .dropna(subset=["PTS"])
            .rename(columns={"Team": "Opponent", "PTS": "_derived_opp_pts"})
        )
        combined = combined.merge(pts_ref, on=["Date", "Opponent"], how="left")
        combined["OPP_PTS"] = combined["OPP_PTS"].fillna(combined["_derived_opp_pts"])
        combined = combined.drop(columns=["_derived_opp_pts"])

        # Derive Win from PTS/OPP_PTS for rows that still lack it
        if "Win" not in combined.columns:
            combined["Win"] = np.nan
        mask = combined["Win"].isna() & combined["PTS"].notna() & combined["OPP_PTS"].notna()
        combined.loc[mask, "Win"] = (combined.loc[mask, "PTS"] > combined.loc[mask, "OPP_PTS"]).astype(int)

        logger.info("Merged dataset: %d rows", len(combined))
        return combined.reset_index(drop=True)

    def save(self, df: pd.DataFrame, filename: str = "games_clean.csv") -> None:
        """Save processed data to the processed directory."""
        path = PROCESSED_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Saved processed data to %s (%d rows)", path, len(df))
