"""Orchestrates feature engineering and builds the final training dataset."""

import logging

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from features.team_features import build_team_features
from features.player_features import compute_team_roster_strength

logger = logging.getLogger(__name__)

DIFF_FEATURES = [
    "diff_pts", "diff_opp_pts", "diff_fg_pct", "diff_fg3_pct",
    "diff_reb", "diff_ast", "diff_tov", "diff_stl", "diff_blk",
    "diff_form", "diff_streak", "diff_rest", "diff_roster", "diff_elo",
]

ALL_FEATURES = DIFF_FEATURES + ["is_home", "is_b2b"]

ROLL_COLS = [
    "roll_PTS", "roll_OPP_PTS", "roll_FG_PCT", "roll_FG3_PCT",
    "roll_REB", "roll_AST", "roll_TOV", "roll_STL", "roll_BLK",
]
STATE_COLS = ["form", "streak", "rest_days", "elo"]


def build_matchup_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head feature differentials between teams.

    For each game, joins the opponent's rolling stats and computes the
    difference. Positive diff values favour the row's team.
    """
    roll_cols = [
        c for c in df.columns
        if c.startswith("roll_") or c in ("form", "streak", "rest_days", "elo")
    ]

    team_values = set(df["Team"].unique())
    opp_values = set(df["Opponent"].unique())
    unmatched = opp_values - team_values
    if unmatched:
        logger.warning(
            "Dropping %d unresolved opponent abbreviations before matchup merge: %s",
            len(unmatched), sorted(unmatched)[:10],
        )
        df = df[df["Opponent"].isin(team_values)].copy()

    opp = df[["Date", "Team"] + roll_cols].copy()
    opp.columns = ["Date", "Opponent"] + [f"opp_{c}" for c in roll_cols]

    merged = df.merge(opp, on=["Date", "Opponent"], how="inner")
    if merged.empty:
        raise ValueError(
            "Matchup merge produced 0 rows — all Opponent values lack a matching Team row. "
            "Check TEAM_ABBREVIATION_FIXES in config.py."
        )

    col_pairs = [
        ("roll_PTS",        "opp_roll_PTS",        "diff_pts"),
        ("roll_OPP_PTS",    "opp_roll_OPP_PTS",    "diff_opp_pts"),
        ("roll_FG_PCT",     "opp_roll_FG_PCT",     "diff_fg_pct"),
        ("roll_FG3_PCT",    "opp_roll_FG3_PCT",    "diff_fg3_pct"),
        ("roll_REB",        "opp_roll_REB",        "diff_reb"),
        ("roll_AST",        "opp_roll_AST",        "diff_ast"),
        ("roll_TOV",        "opp_roll_TOV",        "diff_tov"),
        ("roll_STL",        "opp_roll_STL",        "diff_stl"),
        ("roll_BLK",        "opp_roll_BLK",        "diff_blk"),
        ("form",            "opp_form",            "diff_form"),
        ("streak",          "opp_streak",          "diff_streak"),
        ("rest_days",       "opp_rest_days",       "diff_rest"),
        ("elo",             "opp_elo",             "diff_elo"),
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
    extra_state = [c for c in ROLL_COLS + STATE_COLS if c in df.columns]

    optional = {"diff_fg3_pct", "diff_stl", "diff_blk", "diff_roster", "diff_elo"}
    for col in optional:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    core_required = [c for c in available_features if c not in optional]
    final = df[available_features + meta + extra_state].dropna(subset=core_required)

    final.to_csv(PROCESSED_DIR / "features.csv", index=False)
    logger.info(
        "Feature dataset built: %d rows, %d features", len(final), len(available_features)
    )
    return final
