"""XGBoost gradient-boosted tree model for NBA win prediction."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from config import CV_FOLDS, RANDOM_STATE
from models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier with optional hyperparameter tuning."""

    PARAM_GRID = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
    }

    def __init__(self, auto_tune: bool = True, **kwargs):
        self.auto_tune = auto_tune
        self.params = kwargs
        self.model: XGBClassifier | None = None

    @property
    def name(self) -> str:
        return "XGBoost"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.auto_tune:
            self.model = self._tune(X, y)
        else:
            self.model = XGBClassifier(
                eval_metric="logloss", random_state=RANDOM_STATE,
                verbosity=0, **self.params,
            )
            self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        if self.model:
            return self.model.get_params()
        return self.params

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
        base = XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
        )
        search = RandomizedSearchCV(
            base, self.PARAM_GRID, n_iter=20, cv=CV_FOLDS,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
        )
        search.fit(X, y)
        return search.best_estimator_
