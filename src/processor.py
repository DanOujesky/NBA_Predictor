import logging
import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class ProcessorConfig:
    raw_path: Path = Path("data/raw/nba_raw.csv")
    processed_path: Path = Path("data/processed/nba_processed.csv")
    scaler_path: Path = Path("models/scaler.pkl")
    log_path: Path = Path("logs/processor.log")
    
    rolling_window: int = 10
    form_window: int = 5
    feature_cols: List[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'feature_cols', [
            "Diff_Points", "Diff_FG_pct", "Diff_AST",
            "Diff_TOV", "Diff_Form", "Diff_Streak"
        ])

class NBADataProcessor:
    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = StandardScaler()

    def _setup_logging(self):
        os.makedirs(self.config.log_path.parent, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )

    def load_data(self) -> pd.DataFrame:
        if not self.config.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.config.raw_path}")
        df = pd.read_csv(self.config.raw_path)
        self.logger.info(f"Loaded {len(df)} rows from raw data.")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase 1: Data cleaning and normalization...")
        df = df.copy().drop_duplicates()

        rename_map = {
            "Tm": "Team_Points",
            "Opp_1": "Opponent_Points",
            "FG%": "Team_FG_pct",
            "AST": "Team_AST",
            "TOV": "Team_TOV",
            "Opp": "Opponent_Code"
        }
        df = df.rename(columns=rename_map)

        df["Team"] = df["Team"].astype(str).str.strip()
        df["Opponent_Code"] = df["Opponent_Code"].astype(str).str.strip()
        df["Date"] = pd.to_datetime(df["Date"])

        numeric_cols = ["Team_Points", "Opponent_Points", "Team_FG_pct", "Team_AST", "Team_TOV"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "HomeAway" in df.columns:
            df["Is_Home"] = np.where(df["HomeAway"] == "@", 0, 1)
        else:
            df["Is_Home"] = 1

        df = df.dropna(subset=["Team_Points", "Date", "Team", "Opponent_Code"])
        return df.sort_values(["Team", "Date"])

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase 2: Feature engineering (rolling statistics)...")
        
        df["Target"] = (df["Team_Points"] > df["Opponent_Points"]).astype(float)
        stats = ["Team_Points", "Team_FG_pct", "Team_AST", "Team_TOV"]
        
        for col in stats:
            df[f"Roll_{col}"] = df.groupby("Team")[col].transform(
                lambda x: x.rolling(window=self.config.rolling_window, min_periods=1).mean().shift(1)
            )

        df["Form"] = df.groupby("Team")["Target"].transform(
            lambda x: x.rolling(window=self.config.form_window, min_periods=1).mean().shift(1)
        )

        def calc_streak(series):
            streak = 0
            results = []
            for val in series:
                results.append(streak)
                if val == 1:
                    streak = streak + 1 if streak >= 0 else 1
                else:
                    streak = streak - 1 if streak <= 0 else -1
            return pd.Series(results, index=series.index)

        df["Streak"] = df.groupby("Team")["Target"].transform(calc_streak)
        return df.dropna(subset=["Roll_Team_Points", "Form"])

    def merge_and_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase 3: Computing feature differentials...")

        cols_to_compare = [
            "Roll_Team_Points", "Roll_Team_FG_pct",
            "Roll_Team_AST", "Roll_Team_TOV", "Form", "Streak"
        ]

        opp_df = df[["Date", "Team"] + cols_to_compare].copy()
        opp_df.columns = ["Date", "Opponent_Code"] + [f"Opp_{c}" for c in cols_to_compare]

        merged = df.merge(opp_df, on=["Date", "Opponent_Code"], how="inner")

        if merged.empty:
            self.logger.error("MERGE RESULTED IN 0 ROWS! Problém v názvech týmů.")
            raise ValueError("Data mismatch: Team names and Opponent codes do not align.")

        merged["Diff_Points"] = merged["Roll_Team_Points"] - merged["Opp_Roll_Team_Points"]
        merged["Diff_FG_pct"] = merged["Roll_Team_FG_pct"] - merged["Opp_Roll_Team_FG_pct"]
        merged["Diff_AST"] = merged["Roll_Team_AST"] - merged["Opp_Roll_Team_AST"]
        merged["Diff_TOV"] = merged["Roll_Team_TOV"] - merged["Opp_Roll_Team_TOV"]
        merged["Diff_Form"] = merged["Form"] - merged["Opp_Form"]
        merged["Diff_Streak"] = merged["Streak"] - merged["Opp_Streak"]

        return merged

    def apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Phase 4: Feature scaling...")
        
        scaler_data = df[self.config.feature_cols]
        df[self.config.feature_cols] = self.scaler.fit_transform(scaler_data)

        os.makedirs(self.config.scaler_path.parent, exist_ok=True)
        joblib.dump(self.scaler, self.config.scaler_path)
        return df

    def run_pipeline(self):
        try:
            df = self.load_data()
            df = self.clean_data(df)
            df = self.engineer_features(df)
            df = self.merge_and_diff(df)
            df = self.apply_scaling(df)

            ui_cols = [
                "Roll_Team_Points", "Roll_Team_FG_pct", "Roll_Team_AST", 
                "Roll_Team_TOV", "Form", "Streak"
            ]
            
            meta_cols = ["Is_Home", "Target", "Team", "Opponent_Code", "Date"]
            
            final_selection = self.config.feature_cols + meta_cols + ui_cols
            
            final_selection = list(dict.fromkeys(final_selection))
            
            df_final = df[final_selection]

            os.makedirs(self.config.processed_path.parent, exist_ok=True)
            df_final.to_csv(self.config.processed_path, index=False)
            self.logger.info(f"Pipeline finished! Saved {len(df_final)} rows.")
            return df_final

        except Exception as e:
            self.logger.error(f"Critical failure: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    NBADataProcessor().run_pipeline()