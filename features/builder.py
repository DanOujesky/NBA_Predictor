"""Orchestrates feature engineering and builds the final training dataset."""

import logging

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from features.team_features import build_team_features
from features.player_features import compute_team_roster_strength

logger = logging.getLogger(__name__)

DIFF_FEATURES = [
    "diff_pts", "diff_opp_pts", "diff_fg_pct", "diff_reb",
    "diff_ast", "diff_tov", "diff_form", "diff_streak",
    "diff_rest", "diff_roster",
]

ALL_FEATURES = DIFF_FEATURES + ["is_home", "is_b2b"]

# Individual rolling columns kept in features.csv so they can be used at prediction time
ROLL_COLS = ["roll_PTS", "roll_OPP_PTS", "roll_FG_PCT", "roll_REB", "roll_AST", "roll_TOV"]
STATE_COLS = ["form", "streak", "rest_days"]


def build_matchup_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head feature differentials between teams.

    For each game, joins the opponent's rolling stats and computes the
    difference. Positive diff values favour the row's team.
    """
    roll_cols = [
        c for c in df.columns
        if c.startswith("roll_") or c in ("form", "streak", "rest_days")
    ]
    opp = df[["Date", "Team"] + roll_cols].copy()
    opp.columns = ["Date", "Opponent"] + [f"opp_{c}" for c in roll_cols]

    merged = df.merge(opp, on=["Date", "Opponent"], how="inner")
    if merged.empty:
        logger.error(
            "Matchup merge produced 0 rows — verify team abbreviation consistency "
            "between Team and Opponent columns"
        )
        raise ValueError("Team name mismatch: Opponent values don't match any Team value")

    col_pairs = [
        ("roll_PTS",     "opp_roll_PTS",     "diff_pts"),
        ("roll_OPP_PTS", "opp_roll_OPP_PTS", "diff_opp_pts"),
        ("roll_FG_PCT",  "opp_roll_FG_PCT",  "diff_fg_pct"),
        ("roll_REB",     "opp_roll_REB",     "diff_reb"),
        ("roll_AST",     "opp_roll_AST",     "diff_ast"),
        ("roll_TOV",     "opp_roll_TOV",     "diff_tov"),
        ("form",         "opp_form",         "diff_form"),
        ("streak",       "opp_streak",       "diff_streak"),
        ("rest_days",    "opp_rest_days",     "diff_rest"),
    ]
    for team_col, opp_col, diff_col in col_pairs:
        if team_col in merged.columns and opp_col in merged.columns:
            merged[diff_col] = merged[team_col] - merged[opp_col]

    return merged


def add_roster_features(df: pd.DataFrame, player_data: pd.DataFrame | None) -> pd.DataFrame:
    """Attach roster availability differential to the matchup dataset.

    player_data can be either:
      - Raw NBA API player_stats (GP-based availability) — used during training.
      - Pre-processed injury DataFrame from data.injuries — used for prediction.
    """
    if player_data is None or player_data.empty:
        df["diff_roster"] = 0.0
        return df

    roster = compute_team_roster_strength(player_data)

    # Rename team column to "Team" if it's "team_abbr"
    if "team_abbr" in roster.columns and "Team" not in roster.columns:
        roster = roster.rename(columns={"team_abbr": "Team"})

    df = df.merge(
        roster[["Team", "availability_ratio"]],
        on="Team",
        how="left",
    ).rename(columns={"availability_ratio": "team_avail"})

    df = df.merge(
        roster[["Team", "availability_ratio"]].rename(
            columns={"Team": "Opponent", "availability_ratio": "opp_avail"}
        ),
        on="Opponent",
        how="left",
    )

    df["team_avail"] = df["team_avail"].fillna(1.0)
    df["opp_avail"] = df["opp_avail"].fillna(1.0)
    df["diff_roster"] = df["team_avail"] - df["opp_avail"]
    return df


def build_dataset(games: pd.DataFrame, player_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full feature-building pipeline: team stats → matchup diffs → roster impact.

    Returns a DataFrame with all model features, rolling state columns
    (for prediction reuse), and metadata columns. Target column is 'Win'.
    """
    logger.info("Building feature dataset from %d games", len(games))
    df = build_team_features(games)
    df = df.dropna(subset=["roll_PTS", "form"])
    df = build_matchup_differentials(df)
    df = add_roster_features(df, player_data)

    available_features = [c for c in ALL_FEATURES if c in df.columns]
    meta = ["Team", "Opponent", "Date", "Win"]
    # Keep individual rolling/state cols so prediction can reuse them
    extra_state = [c for c in ROLL_COLS + STATE_COLS if c in df.columns]

    final = df[available_features + meta + extra_state].dropna(subset=available_features)

    final.to_csv(PROCESSED_DIR / "features.csv", index=False)
    logger.info(
        "Feature dataset built: %d rows, %d features", len(final), len(available_features)
    )
    return final
