import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
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
        "Diff_Points", "Diff_FG_pct", "Diff_AST",
        "Diff_TOV", "Diff_Form", "Diff_Streak", "Is_Home"
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
            handlers=[logging.FileHandler(self.config.log_path), logging.StreamHandler()]
        )

    def load_dataset(self) -> pd.DataFrame:
        if not self.config.dataset_path.exists():
            self.logger.error(f"Dataset not found: {self.config.dataset_path}")
            raise FileNotFoundError("Processed dataset is missing. Run the data processor first.")
        return pd.read_csv(self.config.dataset_path)

    def optimize_hyperparameters(self, X, y):

        self.logger.info("Starting hyperparameter optimization...")

        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        model = XGBClassifier(eval_metric="logloss", random_state=self.config.random_state)

        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=10,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.config.random_state
        )

        search.fit(X, y)
        self.logger.info(f"Best parameters: {search.best_params_}")

        return search.best_estimator_

    def train_model(self):

        df = self.load_dataset()
        X = df[list(self.config.features)]
        y = df[self.config.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        self.model = self.optimize_hyperparameters(X_train, y_train)

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]

        self.logger.info("=" * 30)
        self.logger.info(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        self.logger.info(f"ROC-AUC:  {roc_auc_score(y_test, probabilities):.4f}")
        self.logger.info("\n" + classification_report(y_test, predictions))
        self.logger.info("=" * 30)

    def get_feature_importance(self) -> Dict[str, float]:

        if self.model is None:
            return {}

        importances = self.model.feature_importances_
        return dict(zip(self.config.features, importances))

    def save_model(self):
        os.makedirs(self.config.model_path.parent, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)
        self.logger.info(f"Model saved to {self.config.model_path}")

    def load_model(self):
        if not self.config.model_path.exists():
            raise FileNotFoundError("Model file not found. Train the model first.")
        self.model = joblib.load(self.config.model_path)

    def predict_win_probability(self, feature_vector: pd.DataFrame) -> float:

        if self.model is None:
            self.load_model()

        X = feature_vector[list(self.config.features)]
        return float(self.model.predict_proba(X)[0][1])

    def calculate_betting_value(self, prob: float, odds: float) -> Dict:

        fair_odds = 1 / prob if prob > 0 else 0
        expected_value = (prob * odds) - 1

        return {
            "win_probability": prob,
            "fair_odds": round(fair_odds, 2),
            "expected_value": expected_value,
            "has_value": expected_value > 0.05
        }


if __name__ == "__main__":
    predictor = NBAPredictor()
    predictor.train_model()
    predictor.save_model()