import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessorConfig:
    """
    Configuration for NBA Data Processing pipeline.
    """
    raw_path: Path = Path("data/raw/nba_raw.csv")
    output_path: Path = Path("data/processed/nba_processed.csv")
    log_path: Path = Path("processor.log")
    numeric_columns: tuple = (
        "Tm", "Opp", "FG", "FGA", "3P", "3PA",
        "FT", "FTA", "ORB", "DRB", "TRB",
        "AST", "STL", "BLK", "TOV", "PF"
    )


class NBADataProcessor:
    """
    Professional-grade ETL pipeline for NBA game statistics.

    This class handles the end-to-end lifecycle of game log data, 
    ensuring type consistency, schema validation, and feature derivation.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initializes the processor with configuration and logging.
        """
        self.config = config or ProcessorConfig()
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_logging(self) -> None:
        """
        Configures the logging strategy for the application.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )

    def load_raw_data(self) -> pd.DataFrame:
        """
        Reads the raw CSV dataset from the configured path.

        Raises:
            FileNotFoundError: If the source file does not exist.
        """
        if not self.config.raw_path.exists():
            self.logger.error(f"Source file missing: {self.config.raw_path}")
            raise FileNotFoundError(f"Raw dataset not found at {self.config.raw_path}")

        df = pd.read_csv(self.config.raw_path)
        self.logger.info(f"Successfully loaded {len(df)} records.")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data cleaning including deduplication and type normalization.
        """
        df = df.copy().drop_duplicates()
        
        subset_cols = [c for c in ["Tm", "Opp"] if c in df.columns]
        df = df.dropna(subset=subset_cols)

        for col in self.config.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=subset_cols)
        
        self.logger.info(f"Cleaning complete. Remaining rows: {len(df)}")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives analytical features and performance metrics from raw data.
        """
        df = df.copy()

        df["ScoreDiff"] = df["Tm"] - df["Opp"]
        df["Win"] = (df["ScoreDiff"] > 0).astype(int)

        metrics = {
            "FG_pct": ("FG", "FGA"),
            "3P_pct": ("3P", "3PA"),
            "FT_pct": ("FT", "FTA")
        }

        for feature, (num, den) in metrics.items():
            if num in df.columns and den in df.columns:
                df[feature] = (df[num] / df[den]).replace([np.inf, -np.inf], np.nan)

        if "ORB" in df.columns and "DRB" in df.columns:
            df["Total_Rebounds"] = df["ORB"] + df["DRB"]

        if "AST" in df.columns and "TOV" in df.columns:
            df["AST_TOV_Ratio"] = df["AST"] / (df["TOV"] + 1)

        self.logger.info("Feature engineering transformation applied.")
        return df

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Executes structural and content validation on the processed DataFrame.

        Raises:
            ValueError: If the dataset is empty or critical columns are missing.
        """
        if df.empty:
            raise ValueError("Processing resulted in an empty DataFrame.")

        required_features = ["Win", "ScoreDiff"]
        missing = [col for col in required_features if col not in df.columns]
        
        if missing:
            raise ValueError(f"Validation failed. Missing columns: {missing}")

        self.logger.info("Dataset integrity validation passed.")

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Persists the final dataset to the output path.
        """
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.config.output_path, index=False)
        self.logger.info(f"Processed dataset exported to: {self.config.output_path}")

    def run_pipeline(self) -> None:
        """
        Orchestrates the full execution flow of the data processing pipeline.
        """
        try:
            self.logger.info("Pipeline execution initiated.")
            
            raw_df = self.load_raw_data()
            cleaned_df = self.clean_data(raw_df)
            enriched_df = self.create_features(cleaned_df)
            
            self.validate_dataset(enriched_df)
            self.save_processed_data(enriched_df)
            
            self.logger.info("Pipeline execution completed successfully.")
        except Exception as e:
            self.logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    processor = NBADataProcessor()
    processor.run_pipeline()