import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score



@dataclass(frozen=True)
class Config:
    dataset_path: Path = Path("data/processed/nba_processed.csv")
    model_path: Path = Path("models/nba_predictor.pkl")

    test_size: float = 0.2
    random_state: int = 42




class NBAPredictor:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_logging()
        self.logger = logging.getLogger("NBAPredictor")
        self.model: Optional[XGBClassifier] = None

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )

  

    def load_data(self) -> pd.DataFrame:
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        df = pd.read_csv(self.config.dataset_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        self.logger.info(f"Loaded {len(df)} rows")
        return df



    def time_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_index = int(len(df) * (1 - self.config.test_size))

        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        self.logger.info(f"Train size: {len(train)}")
        self.logger.info(f"Test size: {len(test)}")

        return train, test

  

    def get_features(self, df: pd.DataFrame):

        feature_columns = [
            "HomeGame",
            "FG_pct",
            "ThreeP_pct",
            "FT_pct",
            "Total_Rebounds",
            "AST_TOV_Ratio"
        ]

      
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        X = df[feature_columns].fillna(0)
        y = df["Win"]

        return X, y



    def train(self, X_train, y_train, X_test, y_test):

        self.logger.info("Training model...")

        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=self.config.random_state
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        self.logger.info(f"Accuracy: {acc:.4f}")
        self.logger.info(f"ROC-AUC: {auc:.4f}")



    def save_model(self):
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.config.model_path)
        self.logger.info(f"Model saved to {self.config.model_path}")


    def load_model(self):
        if not self.config.model_path.exists():
            raise FileNotFoundError("Model not found")

        self.model = joblib.load(self.config.model_path)
        self.logger.info("Model loaded")



    def predict(self, features: dict) -> float:
        if self.model is None:
            raise ValueError("Model not loaded")

        df = pd.DataFrame([features])
        prob = self.model.predict_proba(df)[0][1]
        return float(prob)



    def run(self):
        self.logger.info("Starting training pipeline")

        df = self.load_data()

        train_df, test_df = self.time_split(df)

        X_train, y_train = self.get_features(train_df)
        X_test, y_test = self.get_features(test_df)

        self.train(X_train, y_train, X_test, y_test)
        self.save_model()

        self.logger.info("Training completed successfully")



if __name__ == "__main__":
    predictor = NBAPredictor()
    predictor.run()