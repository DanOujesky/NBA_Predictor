import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def get_base_path() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(os.path.abspath(".")).resolve()

@dataclass(frozen=True)
class PredictorConfig:
    BASE_PATH: Path = get_base_path()
    dataset_path: Path = BASE_PATH / "data/processed/nba_processed.csv"
    model_path: Path = BASE_PATH / "models/nba_win_predictor.pkl"
    log_path: Path = BASE_PATH / "logs/predictor.log"

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
        logging.getLogger().handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def load_dataset(self) -> pd.DataFrame:
        if not self.config.dataset_path.exists():
            self.logger.error(f"Dataset not found: {self.config.dataset_path}")
            raise FileNotFoundError("Processed data missing. Run NBADataProcessor first.")
        
        df = pd.read_csv(self.config.dataset_path)
        df = df.dropna(subset=list(self.config.features) + [self.config.target])
        return df
    
    def needs_retraining(self) -> bool:
        if not self.config.model_path.exists():
            return True
        
        model_time = os.path.getmtime(self.config.model_path)
        data_time = os.path.getmtime(self.config.dataset_path)
        return data_time > model_time

    def optimize_hyperparameters(self, X, y) -> XGBClassifier:
        self.logger.info("Starting hyperparameter tuning (RandomizedSearch)...")

        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        xgb = XGBClassifier(
            eval_metric="logloss",
            random_state=self.config.random_state,
            verbosity=0
        )

        search = RandomizedSearchCV(
            xgb,
            param_grid,
            n_iter=15,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.config.random_state
        )

        search.fit(X, y)
        self.logger.info(f"Best parameters found: {search.best_params_}")
        return search.best_estimator_

    def train_model(self):
        self.logger.info("Phase 5: Training NBA Prediction Model...")
        df = self.load_dataset()
        
        X = df[list(self.config.features)]
        y = df[self.config.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )

        self.model = self.optimize_hyperparameters(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        self.logger.info("-" * 40)
        self.logger.info(f"MODEL PERFORMANCE:")
        self.logger.info(f"Accuracy: {acc:.4f}")
        self.logger.info(f"ROC-AUC:  {auc:.4f}")
        self.logger.info("\n" + classification_report(y_test, preds))
        self.logger.info("-" * 40)

    def save_model(self):
        if self.model is None:
            self.logger.error("Cannot save a model that has not been trained.")
            return
            
        os.makedirs(self.config.model_path.parent, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)
        self.logger.info(f"Model successfully saved to: {self.config.model_path}")

    def load_model(self):
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.config.model_path}")
        self.model = joblib.load(self.config.model_path)
        self.logger.info("Model loaded from disk.")

    def predict_win_probability(self, feature_vector: pd.DataFrame) -> float:
        if self.model is None:
            self.load_model()

        X = feature_vector[list(self.config.features)]
        return float(self.model.predict_proba(X)[0][1])

    def calculate_betting_value(self, prob: float, odds: float) -> Dict[str, Any]:
        fair_odds = 1 / prob if prob > 0.01 else 999.0
        edge = (prob * odds) - 1
        
        return {
            "win_probability": round(prob, 4),
            "fair_odds": round(fair_odds, 2),
            "edge_pct": round(edge * 100, 2),
            "has_value": edge > 0.05
        }

if __name__ == "__main__":
    predictor = NBAPredictor()
    
    if predictor.needs_retraining():
        predictor.train_model()
        predictor.save_model()
    else:
        print("Model is up to date, loading from disk...")
        predictor.load_model()