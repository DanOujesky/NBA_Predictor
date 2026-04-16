"""Player availability and roster strength features.

Two input formats are supported:
  1. Raw player_stats DataFrame from the NBA API (columns: PLAYER_NAME,
     TEAM_ABBREVIATION, GP, MIN, PTS, REB, AST, STL, BLK, TOV) — used
     during the historical training pipeline.
  2. Processed injury_df from data.injuries (columns: player_name, team_abbr,
     availability_factor, is_available, player_value) — used at prediction
     time when real-time injury data is available.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PLAYER_VALUE_WEIGHTS = {
    "PTS": 1.0, "REB": 0.7, "AST": 1.0,
    "STL": 1.5, "BLK": 1.2, "TOV": -1.0,
}


def compute_player_value(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Add a composite player_value score to a raw player stats DataFrame.

    The weights approximate advanced metrics like PER without requiring
    possession-based data.
    """
    if player_stats.empty:
        return player_stats
    df = player_stats.copy()
    df["player_value"] = sum(
        df[col].fillna(0) * w
        for col, w in PLAYER_VALUE_WEIGHTS.items()
        if col in df.columns
    )
    return df


def compute_team_roster_strength(data: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level roster strength from either data format.

    Accepts both raw NBA API player_stats (with TEAM_ABBREVIATION, GP) and
    pre-processed injury DataFrames (with team_abbr, availability_factor).

    Returns a DataFrame with columns:
        Team, roster_strength, available_strength, availability_ratio.

    availability_ratio ∈ [0, 1]: weighted fraction of the full roster that
    is currently healthy enough to contribute.
    """
    if data.empty:
        logger.warning("No player data for roster strength computation")
        return pd.DataFrame(
            columns=["Team", "roster_strength", "available_strength", "availability_ratio"]
        )

    df = data.copy()

    if "team_abbr" in df.columns:
        team_col = "team_abbr"
        if "player_value" not in df.columns:
            df = compute_player_value(df)
        if "availability_factor" not in df.columns:
            df["availability_factor"] = df["is_available"].astype(float) if "is_available" in df.columns else 1.0
    else:
        team_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else "Team"
        df = compute_player_value(df)
        max_gp = df["GP"].max() if "GP" in df.columns else 1
        df["availability_factor"] = (df.get("GP", max_gp) / max_gp).clip(0, 1)
        df["is_available"] = df["availability_factor"] > 0.7

    df["weighted_value"] = df["player_value"] * df["availability_factor"]

    total = df.groupby(team_col)["player_value"].sum().rename("roster_strength")
    available = (
        df[df["is_available"]]
        .groupby(team_col)["weighted_value"]
        .sum()
        .rename("available_strength")
    )

    result = pd.DataFrame({"roster_strength": total, "available_strength": available}).fillna(0)
    safe_total = result["roster_strength"].replace(0, 1)
    result["availability_ratio"] = (result["available_strength"] / safe_total).clip(0, 1)
    result = result.reset_index().rename(columns={team_col: "Team"})

    logger.info("Roster strength computed for %d teams", len(result))
    return result
