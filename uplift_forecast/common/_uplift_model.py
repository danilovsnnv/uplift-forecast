__all__ = ['UpliftModel']


from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _to_numpy_1d(arr: ArrayLike) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.reshape(-1)
    if hasattr(arr, 'to_numpy'):
        return arr.to_numpy().reshape(-1)
    return np.asarray(arr).reshape(-1)


def _to_array(arr: ArrayLike) -> np.ndarray | pd.DataFrame:
    """Pass DataFrames through unchanged so sklearn estimators keep feature names."""
    if isinstance(arr, pd.DataFrame):
        return arr
    return np.asarray(arr)


def _row_subset(x: Any, mask: np.ndarray) -> Any:
    if hasattr(x, 'iloc'):
        return x.iloc[mask]
    return x[mask]


class UpliftModel:
    """Base interface for all uplift models.

    Both neural (`BaseNeuralUpliftModel`) and classical (`BaseMetaUpliftModel`)
    models share this contract, so `UpliftForecast` never branches on type.
    """

    alias: str | None = None

    @property
    def display_name(self) -> str:
        return self.alias or type(self).__name__

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **fit_params: Any,
    ) -> 'UpliftModel':
        """Fit the model.

        Args:
            X: Feature matrix.
            treatment: Binary treatment vector (0/1).
            y: Outcome vector.
            eval_set: Optional (X_val, treatment_val, y_val) for validation.
        """
        raise NotImplementedError

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict uplift. If return_components=True, return (uplift, y0, y1)."""
        raise NotImplementedError

    @staticmethod
    def _predict_outcome(estimator: Any, X: ArrayLike) -> np.ndarray:
        """Predict the outcome estimate E[Y|x] from a fitted outcome estimator.

        For a probabilistic classifier this is ``predict_proba(X)[:, 1]`` (the positive-class
        probability P(Y=1|x) for a binary outcome); for a regressor it is ``predict(X)``. Lets the
        meta-learners take either a classifier or a regressor as the outcome model without the
        caller wrapping it.
        """
        if hasattr(estimator, 'predict_proba'):
            return np.asarray(estimator.predict_proba(X))[:, 1]
        return np.asarray(estimator.predict(X)).reshape(-1)
