from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CV_FOLDS, RANDOM_STATE
from models.base import BaseModel


class LogisticModel(BaseModel):

    PARAM_GRID = {
        "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        "clf__solver": ["lbfgs", "liblinear"],
    }

    def __init__(self, C: float = 1.0, max_iter: int = 2000, auto_tune: bool = True):
        self.C = C
        self.max_iter = max_iter
        self.auto_tune = auto_tune
        self.model = self._make_pipeline(C)

    def _make_pipeline(self, C: float) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, max_iter=self.max_iter, random_state=RANDOM_STATE,
            )),
        ])

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.auto_tune:
            self.model = self._tune(X, y)
        else:
            self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"C": self.C, "max_iter": self.max_iter, "auto_tune": self.auto_tune}

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        coefs = self.model.named_steps["clf"].coef_[0]
        return dict(zip(feature_names, coefs))

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        search = GridSearchCV(
            self._make_pipeline(1.0),
            self.PARAM_GRID,
            cv=CV_FOLDS,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        return search.best_estimator_
