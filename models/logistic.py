"""Logistic Regression model for NBA win prediction.

Logistic regression is used here as the 'linear' baseline model. It is the
standard linear approach for binary classification and provides interpretable
coefficients that show which features drive predictions.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BaseModel


class LogisticModel(BaseModel):
    """Logistic Regression with built-in feature scaling via sklearn Pipeline."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=self.max_iter, solver="lbfgs", random_state=42,
            )),
        ])

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"C": self.C, "max_iter": self.max_iter}

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature coefficients as importance scores."""
        coefs = self.model.named_steps["clf"].coef_[0]
        return dict(zip(feature_names, coefs))
