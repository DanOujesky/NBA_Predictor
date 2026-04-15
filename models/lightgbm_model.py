from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from config import CV_FOLDS, RANDOM_STATE
from models.base import BaseModel


class LightGBMModel(BaseModel):

    PARAM_GRID = {
        "n_estimators": [200, 300, 400, 500, 700],
        "max_depth": [4, 5, 6, 7, 8, -1],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [10, 20, 30, 50],
        "reg_alpha": [0, 0.01, 0.1, 0.5],
        "reg_lambda": [0, 0.01, 0.1, 0.5, 1.0],
    }

    def __init__(self, auto_tune: bool = True, **kwargs):
        self.auto_tune = auto_tune
        self.params = kwargs
        self.model = None

    @property
    def name(self) -> str:
        return "LightGBM"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        from lightgbm import LGBMClassifier
        if self.auto_tune:
            self.model = self._tune(X, y)
        else:
            self.model = LGBMClassifier(
                random_state=RANDOM_STATE,
                verbosity=-1,
                **self.params,
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

    def _tune(self, X: pd.DataFrame, y: pd.Series):
        from lightgbm import LGBMClassifier
        base = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
        search = RandomizedSearchCV(
            base, self.PARAM_GRID, n_iter=50, cv=CV_FOLDS,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X, y)
        return search.best_estimator_
