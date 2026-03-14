
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


@dataclass(frozen=True)
class PredictorConfig:
    """
    Configuration for the NBA prediction model.
    """

    dataset_path: Path = Path("data/processed/nba_processed.csv")
    model_path: Path = Path("models/nba_win_predictor.pkl")
    log_path: Path = Path("predictor.log")

    test_size: float = 0.2
    random_state: int = 42


class NBAPredictor:
    """
    Machine learning pipeline for predicting NBA game outcomes.

    Responsibilities:
    - Load processed dataset
    - Train ML model
    - Evaluate model performance
    - Save trained model
    - Predict win probability
    """

    def __init__(self, config: Optional[PredictorConfig] = None):

        self.config = config or PredictorConfig()
        self._setup_logging()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[XGBClassifier] = None


    def _setup_logging(self) -> None:

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )


    def load_dataset(self) -> pd.DataFrame:

        if not self.config.dataset_path.exists():

            raise FileNotFoundError(
                f"Processed dataset not found at {self.config.dataset_path}"
            )

        df = pd.read_csv(self.config.dataset_path)

        self.logger.info(f"Loaded dataset with {len(df)} rows")

        return df


    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        target = "Win"

        feature_columns = [
            "Team_Points",
            "Opponent_Points",
            "Team_FG",
            "Team_FGA",
            "Team_3P",
            "Team_3PA",
            "Team_FT",
            "Team_FTA",
            "Team_TRB",
            "Team_AST",
            "Team_STL",
            "Team_BLK",
            "Team_TOV",
            "Team_PF",
            "HomeGame",
            "ScoreDiff",
            "FG_pct",
            "ThreeP_pct",
            "FT_pct",
            "AST_TOV_Ratio",
            "Total_Rebounds"
        ]

        feature_columns = [c for c in feature_columns if c in df.columns]

        X = df[feature_columns].fillna(0)
        y = df[target]

        self.logger.info(f"Using {len(feature_columns)} features")

        return X, y


    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:

        self.logger.info("Starting model training")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.config.random_state,
            eval_metric="logloss"
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        self.logger.info(f"Model accuracy: {acc:.4f}")
        self.logger.info(f"Model ROC-AUC: {auc:.4f}")


    def save_model(self) -> None:

        if self.model is None:
            raise ValueError("No trained model to save")

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, self.config.model_path)

        self.logger.info(f"Model saved to {self.config.model_path}")


    def load_model(self) -> None:

        if not self.config.model_path.exists():

            raise FileNotFoundError(
                f"Model file not found at {self.config.model_path}"
            )

        self.model = joblib.load(self.config.model_path)

        self.logger.info("Model loaded successfully")


    def predict_win_probability(self, game_features: pd.DataFrame) -> float:

        if self.model is None:

            raise ValueError("Model not loaded")

        probability = self.model.predict_proba(game_features)[0][1]

        return float(probability)


    def run_training_pipeline(self) -> None:

        try:

            self.logger.info("Training pipeline started")

            df = self.load_dataset()

            X, y = self.select_features(df)

            self.train_model(X, y)

            self.save_model()

            self.logger.info("Training pipeline completed successfully")

        except Exception:

            self.logger.critical("Training pipeline failed", exc_info=True)
            raise


if __name__ == "__main__":

    predictor = NBAPredictor()

    predictor.run_training_pipeline()
