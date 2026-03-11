import pandas as pd
import numpy as np
import os
import logging


class NBADataProcessor:
    """
    Data processor for NBA game logs.

    Responsibilities:
    - Load raw dataset
    - Clean corrupted rows
    - Normalize data types
    - Feature engineering
    - Save processed dataset
    """

    RAW_PATH = "data/raw/nba_raw.csv"
    OUTPUT_PATH = "data/processed/nba_processed.csv"

    def __init__(self):

        os.makedirs("data/processed", exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            filename="processor.log"
        )

        self.logger = logging.getLogger("NBA_PROCESSOR")

    def load_raw_data(self):
        """Load raw dataset from scraper."""

        if not os.path.exists(self.RAW_PATH):
            raise FileNotFoundError("Raw dataset not found.")

        df = pd.read_csv(self.RAW_PATH)

        self.logger.info(f"Raw dataset loaded: {len(df)} rows")

        return df

    def clean_data(self, df):
        """
        Clean dataset:
        - remove duplicated rows
        - remove invalid rows
        - convert data types
        """

        df = df.copy()

        before = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {before - len(df)} duplicate rows")

        if "Tm" in df.columns and "Opp" in df.columns:
            df = df.dropna(subset=["Tm", "Opp"])

        numeric_columns = [
            "Tm", "Opp", "FG", "FGA", "3P", "3PA",
            "FT", "FTA", "ORB", "DRB", "TRB",
            "AST", "STL", "BLK", "TOV", "PF"
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Tm", "Opp"])

        self.logger.info(f"Rows after cleaning: {len(df)}")

        return df

    def create_features(self, df):
        """
        Create analytical features useful for ML models.
        """

        df = df.copy()

        df["ScoreDiff"] = df["Tm"] - df["Opp"]

        df["Win"] = (df["ScoreDiff"] > 0).astype(int)

        if "FG" in df.columns and "FGA" in df.columns:
            df["FG_pct"] = df["FG"] / df["FGA"]

        if "3P" in df.columns and "3PA" in df.columns:
            df["3P_pct"] = df["3P"] / df["3PA"]

        if "FT" in df.columns and "FTA" in df.columns:
            df["FT_pct"] = df["FT"] / df["FTA"]

        if "ORB" in df.columns and "DRB" in df.columns:
            df["Total_Rebounds"] = df["ORB"] + df["DRB"]

        if "AST" in df.columns and "TOV" in df.columns:
            df["AST_TOV_Ratio"] = df["AST"] / (df["TOV"] + 1)

        self.logger.info("Feature engineering complete")

        return df

    def validate_dataset(self, df):
        """
        Basic sanity checks.
        """

        if df.empty:
            raise ValueError("Dataset is empty after processing.")

        if "Win" not in df.columns:
            raise ValueError("Win label missing.")

        self.logger.info("Dataset validation successful")

    def save_processed_data(self, df):

        df.to_csv(self.OUTPUT_PATH, index=False)

        self.logger.info(f"Processed dataset saved: {self.OUTPUT_PATH}")
        self.logger.info(f"Total rows: {len(df)}")

    def run_pipeline(self):
        """
        Execute full processing pipeline.
        """

        self.logger.info("Starting data processing pipeline")

        df = self.load_raw_data()

        df = self.clean_data(df)

        df = self.create_features(df)

        self.validate_dataset(df)

        self.save_processed_data(df)

        self.logger.info("Processing pipeline finished successfully")


if __name__ == "__main__":

    processor = NBADataProcessor()
    processor.run_pipeline()