"""Random Forest model for NBA win prediction."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from config import CV_FOLDS, RANDOM_STATE
from models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier with optional hyperparameter tuning."""

    PARAM_GRID = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    def __init__(self, auto_tune: bool = True, **kwargs):
        self.auto_tune = auto_tune
        self.params = kwargs
        self.model: RandomForestClassifier | None = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.auto_tune:
            self.model = self._tune(X, y)
        else:
            self.model = RandomForestClassifier(
                random_state=RANDOM_STATE, **self.params,
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

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        base = RandomForestClassifier(random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            base, self.PARAM_GRID, n_iter=20, cv=CV_FOLDS,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
        )
        search.fit(X, y)
        return search.best_estimator_
