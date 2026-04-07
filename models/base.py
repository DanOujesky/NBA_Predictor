"""Abstract base class for all prediction models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Interface that every NBA prediction model must implement.

    Subclasses provide a specific algorithm (logistic regression, XGBoost, etc.)
    while the evaluator and registry interact with this uniform API.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates for the positive class."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return current hyperparameters as a dictionary."""

    def __repr__(self) -> str:
        return f"<{self.name}>"
