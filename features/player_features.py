"""Player availability and injury impact features."""

import logging

import numpy as np
import pandas as pd

from config import RAW_DIR

logger = logging.getLogger(__name__)


def compute_player_value(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite value score for each player.

    Uses a weighted combination of per-game stats to estimate overall impact.
    Weights are calibrated to approximate advanced metrics like PER.
    """
    if player_stats.empty:
        return player_stats
    df = player_stats.copy()
    weights = {"PTS": 1.0, "REB": 0.7, "AST": 1.0, "STL": 1.5, "BLK": 1.2, "TOV": -1.0}
    df["player_value"] = sum(
        df[col].fillna(0) * w for col, w in weights.items() if col in df.columns
    )
    return df


def compute_team_roster_strength(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute total roster strength per team from individual player values.

    Returns a DataFrame with columns: Team, roster_strength, available_strength,
    availability_ratio. The availability_ratio captures how much of the team's
    full-strength roster is currently active and healthy.
    """
    if player_stats.empty:
        logger.warning("No player data available for roster strength")
        return pd.DataFrame(columns=["Team", "roster_strength", "available_strength", "availability_ratio"])

    df = compute_player_value(player_stats)
    team_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else "Team"

    if "availability" not in df.columns:
        max_gp = df["GP"].max() if "GP" in df.columns else 1
        df["availability"] = (df.get("GP", max_gp) / max_gp).clip(0, 1)
        df["is_available"] = df["availability"] > 0.7

    total = df.groupby(team_col)["player_value"].sum().rename("roster_strength")
    available = (
        df[df["is_available"]]
        .groupby(team_col)["player_value"]
        .sum()
        .rename("available_strength")
    )
    result = pd.DataFrame({"roster_strength": total, "available_strength": available}).fillna(0)
    result["availability_ratio"] = (result["available_strength"] / result["roster_strength"]).clip(0, 1)
    result = result.reset_index().rename(columns={team_col: "Team"})
    logger.info("Computed roster strength for %d teams", len(result))
    return result
