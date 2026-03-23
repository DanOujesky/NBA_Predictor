import logging
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.abspath(".")

@dataclass(frozen=True)
class PredictorConfig:
    BASE_PATH = get_base_path()

    dataset_path: Path = Path(BASE_PATH) / "data/processed/nba_processed.csv"
    model_path: Path = Path(BASE_PATH) / "models/nba_win_predictor.pkl"
    log_path: Path = Path(BASE_PATH) / "logs/predictor.log"

    test_size: float = 0.2
    random_state: int = 42

    features: Tuple[str, ...] = (
        "Diff_Points",
        "Diff_FG_pct",
        "Diff_AST",
        "Diff_TOV",
        "Is_Home",
        "Diff_Form",
        "Diff_Streak"
    )

    target: str = "Target"


class NBAPredictor:

    def __init__(self, config: Optional[PredictorConfig] = None):
        self.config = config or PredictorConfig()
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[XGBClassifier] = None

    def _setup_logging(self):
        os.makedirs(self.config.log_path.parent, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )

    def load_dataset(self):
        return pd.read_csv(self.config.dataset_path)

    def prepare_data(self, df):
        X = df[list(self.config.features)]
        y = df[self.config.target]
        return X, y

    def train_model(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42, stratify=y
        )

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        self.logger.info(f"ACC: {accuracy_score(y_test, preds):.4f}")
        self.logger.info(f"AUC: {roc_auc_score(y_test, probs):.4f}")

    def save_model(self):
        os.makedirs(self.config.model_path.parent, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)

    def load_model(self):
        self.model = joblib.load(self.config.model_path)

    def predict_win_probability(self, X):
        if self.model is None:
            self.load_model()
        return float(self.model.predict_proba(X)[0][1])

    def calculate_betting_value(self, prob, odds):

        fair_odds = 1 / prob if prob > 0 else 0
        ev = (prob * odds) - 1

        return {
            "win_probability": prob,
            "fair_odds": fair_odds,
            "expected_value": ev,
            "has_value": ev > 0.05
        }

    def run_training_pipeline(self):
        df = self.load_dataset()
        X, y = self.prepare_data(df)
        self.train_model(X, y)
        self.save_model()


if __name__ == "__main__":
    NBAPredictor().run_training_pipeline()