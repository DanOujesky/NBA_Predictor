"""Model training, evaluation, and side-by-side comparison."""

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from config import CV_FOLDS, MODEL_DIR, RANDOM_STATE, TEST_SIZE
from models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Stores evaluation metrics for a single model."""

    name: str
    accuracy: float
    roc_auc: float
    f1: float
    log_loss_val: float
    cv_mean: float
    cv_std: float


class ModelEvaluator:
    """Trains multiple models on the same dataset and compares performance.

    Usage:
        evaluator = ModelEvaluator([LogisticModel(), XGBoostModel(), RandomForestModel()])
        results = evaluator.run(feature_df, feature_columns)
        best = evaluator.best_model
    """

    def __init__(self, models: list[BaseModel]):
        self.models = models
        self.results: list[ModelResult] = []
        self.best_model: BaseModel | None = None

    def run(self, df: pd.DataFrame, feature_cols: list[str], target: str = "Win") -> pd.DataFrame:
        """Train and evaluate all registered models.

        Args:
            df: DataFrame with features and target column.
            feature_cols: List of feature column names to use.
            target: Name of the binary target column.

        Returns:
            DataFrame comparing all models on key metrics.
        """
        X = df[feature_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )

        self.results.clear()
        best_auc = -1.0

        for model in self.models:
            logger.info("Training %s...", model.name)
            model.train(X_train, y_train)

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            inner_model = getattr(model, "model", None) or model
            try:
                cv = cross_val_score(inner_model, X, y, cv=CV_FOLDS, scoring="accuracy")
            except Exception:
                cv = np.array([accuracy_score(y_test, preds)])

            result = ModelResult(
                name=model.name,
                accuracy=accuracy_score(y_test, preds),
                roc_auc=roc_auc_score(y_test, probs),
                f1=f1_score(y_test, preds),
                log_loss_val=log_loss(y_test, probs),
                cv_mean=cv.mean(),
                cv_std=cv.std(),
            )
            self.results.append(result)
            logger.info(
                "%s -> Accuracy: %.4f | AUC: %.4f | CV: %.4f +/- %.4f",
                result.name, result.accuracy, result.roc_auc, result.cv_mean, result.cv_std,
            )

            if result.roc_auc > best_auc:
                best_auc = result.roc_auc
                self.best_model = model

        self._save_best()
        return self.comparison_table()

    def comparison_table(self) -> pd.DataFrame:
        """Return a formatted DataFrame comparing all model results."""
        rows = [
            {
                "Model": r.name,
                "Accuracy": round(r.accuracy, 4),
                "ROC-AUC": round(r.roc_auc, 4),
                "F1 Score": round(r.f1, 4),
                "Log Loss": round(r.log_loss_val, 4),
                "CV Mean": round(r.cv_mean, 4),
                "CV Std": round(r.cv_std, 4),
            }
            for r in self.results
        ]
        return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    def _save_best(self) -> None:
        if self.best_model is None:
            return
        path = MODEL_DIR / "best_model.pkl"
        obj = self.best_model.model if hasattr(self.best_model, "model") else self.best_model
        joblib.dump(obj, path)
        meta = MODEL_DIR / "best_model_meta.txt"
        meta.write_text(self.best_model.name)
        logger.info("Best model saved: %s (to %s)", self.best_model.name, path)
