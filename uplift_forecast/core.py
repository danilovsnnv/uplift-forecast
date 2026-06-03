__all__ = ['UpliftForecast']


import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .common._uplift_model import UpliftModel


class UpliftForecast:
    """Run a collection of uplift models on the same data.

    Takes a list of ``UpliftModel`` instances and fits / predicts them on the same dataset,
    collecting predictions into a single DataFrame.

    Args:
        models (list): UpliftModel instances — any mix of neural models
            (BaseNeuralUpliftModel) and meta-learners (BaseMetaUpliftModel).

    Example:
        >>> nf = UpliftForecast(models=[TLearner(CatBoostRegressor()), CFRNet(42)])
        >>> nf.fit(X_train, t_train, y_train, val_df=(X_val, t_val, y_val))
        >>> preds = nf.predict(X_test)
    """

    def __init__(self, models: list[UpliftModel]):
        self.models = list(models)
        for model in self.models:
            if not isinstance(model, UpliftModel):
                raise TypeError(
                    f'{type(model).__name__} must subclass BaseMetaUpliftModel '
                    f'or BaseNeuralUpliftModel.'
                )

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        val_df: tuple | None = None,
        **fit_params,
    ) -> 'UpliftForecast':
        """Fit every model on (X, treatment, y).

        Args:
            X: Feature matrix.
            treatment: Binary treatment vector (0/1).
            y: Outcome vector.
            val_df: Optional (X_val, treatment_val, y_val) forwarded as eval_set.
            **fit_params: Passed through to each model's fit method.
        """
        for model in self.models:
            model.fit(X, treatment, y, eval_set=val_df, **fit_params)
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> pd.DataFrame:
        """Predict uplift with every model on X.

        Returns:
            DataFrame with one ``uplift_<model>`` column per model, plus
            ``<model>_y0_pred`` / ``<model>_y1_pred`` when return_components=True.
            Multi-arm models (predicting ``[n, K-1]`` uplift) emit one
            ``uplift_<model>_arm{k}`` column per treated arm instead.
        """
        result = {}
        for model in self.models:
            name = model.display_name
            if return_components:
                uplift, y0, y1 = model.predict(X, return_components=True)
                self._add_uplift_columns(result, name, uplift)
                result[f'{name}_y0_pred'] = y0
                self._add_component_columns(result, name, y1)
            else:
                self._add_uplift_columns(result, name, model.predict(X))
        return pd.DataFrame(result)

    @staticmethod
    def _add_uplift_columns(result: dict, name: str, uplift: ArrayLike) -> None:
        uplift = np.asarray(uplift)
        if uplift.ndim == 1:
            result[f'uplift_{name}'] = uplift
            return
        for col in range(uplift.shape[1]):
            result[f'uplift_{name}_arm{col + 1}'] = uplift[:, col]

    @staticmethod
    def _add_component_columns(result: dict, name: str, y1: ArrayLike) -> None:
        y1 = np.asarray(y1)
        if y1.ndim == 1:
            result[f'{name}_y1_pred'] = y1
            return
        for col in range(y1.shape[1]):
            result[f'{name}_arm{col + 1}_y1_pred'] = y1[:, col]

    def save(self, path: str | Path) -> None:
        """Pickle the model list to a single file at path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self.models, f)

    @classmethod
    def load(cls, path: str | Path) -> 'UpliftForecast':
        """Load an UpliftForecast saved with save()."""
        with Path(path).open('rb') as f:
            models = pickle.load(f)
        return cls(models=models)
