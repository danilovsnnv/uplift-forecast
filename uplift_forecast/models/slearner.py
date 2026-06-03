__all__ = ['SLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..common._base_meta import BaseMetaUpliftModel


def _stack_treatment(x: np.ndarray | pd.DataFrame, t: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Prepend a treatment column to x."""
    t = t.reshape(-1, 1)
    if isinstance(x, pd.DataFrame):
        out = x.copy()
        out.insert(0, '__treatment__', t.ravel())
        return out
    return np.hstack([t, x])


class SLearner(BaseMetaUpliftModel):
    """Single-model meta-learner (S-learner).

    Trains one base estimator on [treatment, X] -> y, then contrasts
    predictions with treatment forced to 0 vs 1.

    Args:
        model: Base estimator with sklearn-style fit/predict (CatBoost, lgbm, sklearn).
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(self, model: Any, alias: str | None = None):
        super().__init__(alias=alias)
        self.model = model
        self._fitted_model = None

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        x_aug = _stack_treatment(X, treatment.astype(np.float32))
        if eval_set is not None:
            x_val, t_val, y_val = eval_set
            fit_params.setdefault('eval_set', (_stack_treatment(x_val, t_val.astype(np.float32)), y_val))
        self._fitted_model = deepcopy(self.model)
        self._fitted_model.fit(x_aug, y, **fit_params)

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        y0 = np.asarray(self._fitted_model.predict(_stack_treatment(X, np.zeros(n, dtype=np.float32)))).reshape(-1)
        y1 = np.asarray(self._fitted_model.predict(_stack_treatment(X, np.ones(n, dtype=np.float32)))).reshape(-1)
        return y0, y1
