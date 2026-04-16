"""Výpočet příznaků na úrovni týmu pro ML model.

Každá funkce přijímá DataFrame se seřazenými herními logy a přidává nové sloupce.
Vždy se používá shift(1), aby příznak zachycoval stav PŘED zápasem (žádný data leakage).

Příznaky:
- roll_*     : klouzavý průměr statistiky za posledních ROLLING_WINDOW zápasů
- form       : podíl výher za posledních FORM_WINDOW zápasů
- streak     : délka aktuální série výher (+n) nebo proher (−n)
- rest_days  : počet dní od posledního zápasu
- is_b2b     : 1 pokud rest_days ≤ 1
- elo        : ELO rating týmu před zápasem
"""

import logging

import numpy as np
import pandas as pd

from config import FORM_WINDOW, ROLLING_WINDOW

logger = logging.getLogger(__name__)

ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_INITIAL = 1500.0


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá klouzavé průměry statistik za posledních ROLLING_WINDOW zápasů.

    Pro každou dostupnou stat (PTS, OPP_PTS, FG_PCT, …) přidá sloupec roll_<stat>.
    shift(1) zajistí, že průměr neobsahuje aktuální zápas — zabraňuje data leakage.
    """
    df = df.copy()
    stat_cols = [
        c for c in ["PTS", "OPP_PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "STL", "BLK"]
        if c in df.columns
    ]
    for col in stat_cols:
        df[f"roll_{col}"] = (
            df.groupby("Team")[col]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=3).mean().shift(1))
        )
    return df


def compute_form(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá sloupec 'form': podíl výher za posledních FORM_WINDOW zápasů (před aktuálním)."""
    df = df.copy()
    df["form"] = (
        df.groupby("Team")["Win"]
        .transform(lambda x: x.rolling(FORM_WINDOW, min_periods=2).mean().shift(1))
    )
    return df


def compute_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá sloupec 'streak': délka aktuální série před zápasem.

    Kladná hodnota = série výher, záporná = série proher.
    Série se resetuje po první výsledku opačného typu.
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
    """Přidá sloupce 'rest_days' a 'is_b2b'.

    rest_days: počet dní od předchozího zápasu (první zápas dostane výchozí hodnotu 3).
    is_b2b: 1 pokud rest_days ≤ 1 (back-to-back zápas).
    """
    df = df.copy()
    df["rest_days"] = (
        df.groupby("Team")["Date"]
        .transform(lambda x: x.diff().dt.days.fillna(3))
    )
    df["is_b2b"] = (df["rest_days"] <= 1).astype(int)
    return df


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-game ELO ratings for every team-game row.

    Processes games chronologically across all teams. Each team starts at
    ELO_INITIAL (1500). Home teams receive a ELO_HOME_ADVANTAGE bonus during
    expected-score calculation. The pre-game ELO is stored so no future
    information leaks into the feature.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    elo_col: list[float] = [0.0] * len(df)

    for idx, row in df.iterrows():
        team = row["Team"]
        opp = row.get("Opponent", "")
        is_home = int(row.get("is_home", 1))

        r_team = ratings.get(team, ELO_INITIAL)
        r_opp = ratings.get(opp, ELO_INITIAL)

        elo_col[idx] = r_team

        r_team_eff = r_team + (ELO_HOME_ADVANTAGE if is_home else 0.0)
        r_opp_eff = r_opp + (ELO_HOME_ADVANTAGE if not is_home else 0.0)

        expected = 1.0 / (1.0 + 10.0 ** ((r_opp_eff - r_team_eff) / 400.0))
        actual = float(row.get("Win", 0.5))

        ratings[team] = r_team + ELO_K * (actual - expected)

    df["elo"] = elo_col
    return df


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zkompletuje všechny příznaky na úrovni týmu voláním dílčích funkcí.

    Pořadí: rolling stats → form → streak → rest_days → ELO.
    Vstupem je čistý DataFrame z DataProcessor, výstupem je rozšířený
    DataFrame připravený pro build_matchup_differentials().
    """
    logger.info("Computing team-level features")
    df = compute_rolling_stats(df)
    df = compute_form(df)
    df = compute_streak(df)
    df = compute_rest_days(df)
    df = compute_elo(df)
    return df
