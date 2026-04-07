"""Team-level rolling statistics and performance features."""

import logging

import numpy as np
import pandas as pd

from config import FORM_WINDOW, ROLLING_WINDOW

logger = logging.getLogger(__name__)


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages for key team statistics.

    All rolling values are shifted by 1 to prevent data leakage, meaning
    each row's features reflect only information available before that game.
    """
    df = df.copy()
    stat_cols = [c for c in ["PTS", "OPP_PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "STL", "BLK"] if c in df.columns]

    for col in stat_cols:
        df[f"roll_{col}"] = (
            df.groupby("Team")[col]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=3).mean().shift(1))
        )
    return df


def compute_form(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recent win rate over the last FORM_WINDOW games."""
    df = df.copy()
    df["form"] = (
        df.groupby("Team")["Win"]
        .transform(lambda x: x.rolling(FORM_WINDOW, min_periods=2).mean().shift(1))
    )
    return df


def compute_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Compute current win/loss streak for each team.

    Positive values indicate consecutive wins, negative for losses.
    """
    df = df.copy()

    def _streak(series: pd.Series) -> pd.Series:
        streak = 0
        result = []
        for val in series:
            result.append(streak)
            streak = (streak + 1 if streak >= 0 else 1) if val == 1 else (streak - 1 if streak <= 0 else -1)
        return pd.Series(result, index=series.index)

    df["streak"] = df.groupby("Team")["Win"].transform(_streak)
    return df


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest between games and back-to-back flag."""
    df = df.copy()
    df["rest_days"] = (
        df.groupby("Team")["Date"]
        .transform(lambda x: x.diff().dt.days.fillna(3))
    )
    df["is_b2b"] = (df["rest_days"] <= 1).astype(int)
    return df


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all team-level feature transformations in sequence."""
    logger.info("Computing team-level features")
    df = compute_rolling_stats(df)
    df = compute_form(df)
    df = compute_streak(df)
    df = compute_rest_days(df)
    return df
