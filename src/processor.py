import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessorConfig:
    raw_path: Path = Path("data/raw/nba_raw.csv")
    processed_path: Path = Path("data/processed/nba_processed.csv")
    log_path: Path = Path("logs/processor.log")
    rolling_window: int = 10
    form_window: int = 5


class NBADataProcessor:

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_logging(self):
        os.makedirs(self.config.log_path.parent, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )

    def load_raw_data(self) -> pd.DataFrame:
        if not self.config.raw_path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {self.config.raw_path}")

        df = pd.read_csv(self.config.raw_path)

        self.logger.info(f"Loaded raw dataset: {len(df)} rows")
        return df

    def clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning & normalizing data...")

        df = df.copy().drop_duplicates()

        rename_map = {
            "Tm": "Team_Points",
            "Opp.1": "Opponent_Points",
            "FG%": "Team_FG_pct",
            "AST": "Team_AST",
            "TOV": "Team_TOV",
            "Opp": "Opponent_Code"
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        required = ["Date", "Team", "Opponent_Code", "Team_Points", "Opponent_Points"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        numeric_cols = ["Team_Points", "Opponent_Points", "Team_FG_pct", "Team_AST", "Team_TOV"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        ha_col = "Unnamed: 3_level_1"
        if ha_col in df.columns:
            df["Is_Home"] = (df[ha_col] != "@").fillna(True).astype(int)
        else:
            df["Is_Home"] = 1
            self.logger.warning("Home/Away column missing — defaulting to home")

        before = len(df)
        df = df.dropna(subset=["Team_Points", "Opponent_Points", "Date"])
        self.logger.info(f"Dropped {before - len(df)} invalid rows")

        df = df.sort_values(["Team", "Date"]).reset_index(drop=True)

        return df

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Target"] = (df["Team_Points"] > df["Opponent_Points"]).astype(int)
        return df

    def rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating rolling stats...")

        stats = ["Team_Points", "Team_FG_pct", "Team_AST", "Team_TOV"]

        rolling = (
            df.groupby("Team")[stats]
            .rolling(self.config.rolling_window, min_periods=3)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        rolling.columns = [f"Roll_{c}" for c in rolling.columns]

        before = len(df)
        df = pd.concat([df, rolling], axis=1)
        df = df.dropna(subset=rolling.columns)

        self.logger.info(f"Rolling stats reduced rows: {before} → {len(df)}")

        return df

    def form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating form & streak...")

        df["Form"] = (
            df.groupby("Team")["Target"]
            .rolling(self.config.form_window, min_periods=3)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        def streak(series):
            result = []
            count = 0
            last = None

            for val in series:
                if val == last:
                    count += 1
                else:
                    count = 1
                result.append(count if val == 1 else -count)
                last = val

            return pd.Series(result, index=series.index)

        df["Streak"] = df.groupby("Team")["Target"].transform(streak).shift(1)

        before = len(df)
        df = df.dropna(subset=["Form", "Streak"])
        self.logger.info(f"Form features reduced rows: {before} → {len(df)}")

        return df

    def merge_opponent(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Merging opponent features...")

        cols = [
            "Date", "Team",
            "Roll_Team_Points", "Roll_Team_FG_pct",
            "Roll_Team_AST", "Roll_Team_TOV",
            "Form", "Streak"
        ]

        opp = df[cols].copy()
        opp.columns = ["Date", "Opponent_Code"] + [f"Opp_{c}" for c in cols[2:]]

        merged = df.merge(
            opp,
            on=["Date", "Opponent_Code"],
            how="inner",
            validate="many_to_one"
        )

        self.logger.info(f"Merged dataset size: {len(merged)}")

        return merged

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Creating differential features...")

        df["Diff_Points"] = df["Roll_Team_Points"] - df["Opp_Roll_Team_Points"]
        df["Diff_FG_pct"] = df["Roll_Team_FG_pct"] - df["Opp_Roll_Team_FG_pct"]
        df["Diff_AST"] = df["Roll_Team_AST"] - df["Opp_Roll_Team_AST"]
        df["Diff_TOV"] = df["Roll_Team_TOV"] - df["Opp_Roll_Team_TOV"]

        df["Diff_Form"] = df["Form"] - df["Opp_Form"]
        df["Diff_Streak"] = df["Streak"] - df["Opp_Streak"]

        dup_cols = [col for col in df.columns if col.endswith(".1")]
        df = df.drop(columns=dup_cols, errors="ignore")

        df = df.drop(columns=["Unnamed: 3_level_1"], errors="ignore")

        return df

    def validate_dataset(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Final dataset is empty")

        if df.isna().sum().sum() > 0:
            self.logger.warning("Dataset still contains NaNs")

        self.logger.info("Dataset validation passed")

    def run_pipeline(self) -> pd.DataFrame:
        self.logger.info("Pipeline started")

        df = self.load_raw_data()
        df = self.clean_and_normalize(df)
        df = self.add_target(df)
        df = self.rolling_stats(df)
        df = self.form_features(df)
        df = self.merge_opponent(df)
        df = self.create_features(df)

        before = len(df)

        important_cols = [
            "Roll_Team_Points",
            "Opp_Roll_Team_Points",
            "Form",
            "Opp_Form"
        ]

        df = df.dropna(subset=important_cols)

        self.logger.info(f"Final clean dataset: {before} → {len(df)} rows")

        df = df.replace([np.inf, -np.inf], np.nan)

        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Dropping NaNs: {nan_count} values")

        model_cols = [
            "Diff_Points",
            "Diff_FG_pct",
            "Diff_AST",
            "Diff_TOV",
            "Is_Home",
            "Diff_Form",
            "Diff_Streak",
            "Target"
        ]

        before = len(df)

        df = df.dropna(subset=model_cols).reset_index(drop=True)

        self.logger.info(f"Final model dataset: {before} → {len(df)} rows")

        if df.empty:
            raise ValueError("Final dataset is empty after cleaning")

        self.logger.info("Dataset validation passed")

        os.makedirs(self.config.processed_path.parent, exist_ok=True)
        df.to_csv(self.config.processed_path, index=False)

        self.logger.info(f"Pipeline finished: {len(df)} rows saved")

        return df

if __name__ == "__main__":
    NBADataProcessor().run_pipeline()