
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessorConfig:
    """
    Configuration for the NBA ETL pipeline.
    """

    raw_path: Path = Path("data/raw/nba_raw.csv")
    output_path: Path = Path("data/processed/nba_processed.csv")
    log_path: Path = Path("processor.log")


class NBADataProcessor:
    """
    Production-grade ETL pipeline for NBA game statistics.

    Responsibilities:
    - Load raw dataset
    - Clean malformed rows
    - Normalize column schema
    - Create analytical features
    - Validate dataset integrity
    - Export processed dataset
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):

        self.config = config or ProcessorConfig()
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_logging(self) -> None:

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

            raise FileNotFoundError(
                f"Raw dataset not found at {self.config.raw_path}"
            )

        df = pd.read_csv(self.config.raw_path)

        self.logger.info(f"Loaded dataset with {len(df)} rows")

        return df



    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Starting data cleaning stage")

        df = df.copy()

      
        before = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {before - len(df)} duplicate rows")

       
        rename_map = {
            "Tm": "Team_Points",
            "Opp.1": "Opponent_Points",

            "FG": "Team_FG",
            "FGA": "Team_FGA",
            "FG%": "Team_FG_pct",
            "3P": "Team_3P",
            "3PA": "Team_3PA",
            "3P%": "Team_3P_pct",
            "FT": "Team_FT",
            "FTA": "Team_FTA",
            "FT%": "Team_FT_pct",
            "ORB": "Team_ORB",
            "DRB": "Team_DRB",
            "TRB": "Team_TRB",
            "AST": "Team_AST",
            "STL": "Team_STL",
            "BLK": "Team_BLK",
            "TOV": "Team_TOV",
            "PF": "Team_PF",

            "FG.1": "Opp_FG",
            "FGA.1": "Opp_FGA",
            "FG%.1": "Opp_FG_pct",
            "3P.1": "Opp_3P",
            "3PA.1": "Opp_3PA",
            "3P%.1": "Opp_3P_pct",
            "FT.1": "Opp_FT",
            "FTA.1": "Opp_FTA",
            "FT%.1": "Opp_FT_pct",
            "ORB.1": "Opp_ORB",
            "DRB.1": "Opp_DRB",
            "TRB.1": "Opp_TRB",
            "AST.1": "Opp_AST",
            "STL.1": "Opp_STL",
            "BLK.1": "Opp_BLK",
            "TOV.1": "Opp_TOV",
            "PF.1": "Opp_PF",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        self.logger.info("Column normalization completed")

    
        numeric_columns = [
            "Team_Points",
            "Opponent_Points",
            "Team_FG",
            "Team_FGA",
            "Team_3P",
            "Team_3PA",
            "Team_FT",
            "Team_FTA",
            "Team_ORB",
            "Team_DRB",
            "Team_TRB",
            "Team_AST",
            "Team_STL",
            "Team_BLK",
            "Team_TOV",
            "Team_PF",
        ]

        for col in numeric_columns:

            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.logger.info("Numeric conversion complete")

        
        before = len(df)
        df = df.dropna(subset=["Team_Points", "Opponent_Points"])
        removed = before - len(df)

        self.logger.info(f"Removed {removed} rows missing score data")

       
        if "Unnamed: 3_level_1" in df.columns:

            df["HomeGame"] = (df["Unnamed: 3_level_1"] != "@").astype(int)

        else:

            df["HomeGame"] = np.nan
            self.logger.warning("Home/Away column missing")

        if df.empty:

            raise ValueError("Cleaning resulted in empty dataset")

        self.logger.info(f"Cleaning finished. Remaining rows: {len(df)}")

        return df

  

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Starting feature engineering")

        df = df.copy()

       
        df["ScoreDiff"] = df["Team_Points"] - df["Opponent_Points"]

      
        df["Win"] = (df["ScoreDiff"] > 0).astype(int)

        if "Team_FG" in df.columns and "Team_FGA" in df.columns:

            df["FG_pct"] = (df["Team_FG"] / df["Team_FGA"]).replace(
                [np.inf, -np.inf], np.nan
            )

        if "Team_3P" in df.columns and "Team_3PA" in df.columns:

            df["ThreeP_pct"] = (df["Team_3P"] / df["Team_3PA"]).replace(
                [np.inf, -np.inf], np.nan
            )

        if "Team_FT" in df.columns and "Team_FTA" in df.columns:

            df["FT_pct"] = (df["Team_FT"] / df["Team_FTA"]).replace(
                [np.inf, -np.inf], np.nan
            )

      
        if "Team_ORB" in df.columns and "Team_DRB" in df.columns:

            df["Total_Rebounds"] = df["Team_ORB"] + df["Team_DRB"]

        if "Team_AST" in df.columns and "Team_TOV" in df.columns:

            df["AST_TOV_Ratio"] = df["Team_AST"] / (df["Team_TOV"] + 1)

        self.logger.info("Feature engineering complete")

        return df


    def validate_dataset(self, df: pd.DataFrame) -> None:

        if df.empty:

            raise ValueError("Processed dataset is empty")

        required_columns = ["Win", "ScoreDiff"]

        missing = [c for c in required_columns if c not in df.columns]

        if missing:

            raise ValueError(f"Dataset validation failed. Missing: {missing}")

        self.logger.info("Dataset validation passed")


    def save_processed_data(self, df: pd.DataFrame) -> None:

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.config.output_path, index=False)

        self.logger.info(f"Saved processed dataset to {self.config.output_path}")


    def run_pipeline(self) -> None:

        try:

            self.logger.info("Pipeline execution started")

            raw_df = self.load_raw_data()

            cleaned_df = self.clean_data(raw_df)

            enriched_df = self.create_features(cleaned_df)

            self.validate_dataset(enriched_df)

            self.save_processed_data(enriched_df)

            self.logger.info("Pipeline execution finished successfully")

        except Exception as e:

            self.logger.critical("Pipeline failed", exc_info=True)

            raise


if __name__ == "__main__":

    processor = NBADataProcessor()

    processor.run_pipeline()

