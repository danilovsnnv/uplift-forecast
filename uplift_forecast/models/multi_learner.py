__all__ = ['MultiSLearner', 'MultiTLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..common._uplift_model import UpliftModel, _row_subset, _to_array, _to_numpy_1d


def _stack_treatment(x: np.ndarray | pd.DataFrame, t: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Prepend an integer treatment-arm column to x."""
    t = t.reshape(-1, 1)
    if isinstance(x, pd.DataFrame):
        out = x.copy()
        out.insert(0, '__treatment__', t.ravel())
        return out
    return np.hstack([t, x])


class _BaseMultiArmLearner(UpliftModel):
    """Shared logic for multi-arm meta-learners (control arm is 0).

    ``predict`` returns one uplift column per treated arm vs control as a
    ``[n, K-1]`` array; when there is a single treated arm (binary case) it
    collapses to a flat ``[n]`` array, so binary behaviour is preserved.
    """

    def __init__(self, alias: str | None = None):
        self.alias = alias
        self._fitted = False
        self.arms_: list[int] = []

    def fit(
        self,
        X: Any,
        treatment: Any,
        y: Any,
        eval_set: tuple | None = None,
        **fit_params: Any,
    ) -> '_BaseMultiArmLearner':
        x = _to_array(X)
        t = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y)
        self.arms_ = sorted(set(t.tolist()))
        if self.arms_[0] != 0:
            raise ValueError(f'treatment must include a control arm coded 0; got arms {self.arms_}.')
        self._fit_arms(x, t, y_arr, **fit_params)
        self._fitted = True
        return self

    def predict(
        self,
        X: Any,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError(f'{type(self).__name__} has not been fitted yet. Call .fit() first.')
        x = _to_array(X)
        mu = [self._predict_arm(x, arm) for arm in self.arms_]
        mu0 = mu[0]
        treated = mu[1:]
        uplift = np.stack([m - mu0 for m in treated], axis=1)
        y1 = np.stack(treated, axis=1)
        if uplift.shape[1] == 1:
            uplift, y1 = uplift[:, 0], y1[:, 0]
        if return_components:
            return uplift, mu0, y1
        return uplift

    def _fit_arms(self, X: Any, treatment: np.ndarray, y: np.ndarray, **fit_params: Any) -> None:
        raise NotImplementedError

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        raise NotImplementedError


class MultiTLearner(_BaseMultiArmLearner):
    """Multi-arm T-learner: one outcome regressor per treatment arm.

    Fits an independent estimator on each arm's rows (including the control arm 0)
    and reports, for every treated arm ``k``, ``mu_k(x) - mu_0(x)``.

    Args:
        model: Base estimator with sklearn-style ``fit`` / ``predict``.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(self, model: Any, alias: str | None = None):
        super().__init__(alias=alias)
        self.model = model
        self._models: dict[int, Any] = {}

    def _fit_arms(self, X: Any, treatment: np.ndarray, y: np.ndarray, **fit_params: Any) -> None:
        self._models = {}
        for arm in self.arms_:
            mask = treatment == arm
            est = deepcopy(self.model)
            est.fit(_row_subset(X, mask), y[mask], **fit_params)
            self._models[arm] = est

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        return np.asarray(self._models[arm].predict(X)).reshape(-1)


class MultiSLearner(_BaseMultiArmLearner):
    """Multi-arm S-learner: one estimator on ``[treatment_arm, X]``.

    Trains a single estimator with the integer treatment arm as an extra feature,
    then contrasts predictions with the arm forced to ``k`` vs ``0``.

    Args:
        model: Base estimator with sklearn-style ``fit`` / ``predict``.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(self, model: Any, alias: str | None = None):
        super().__init__(alias=alias)
        self.model = model
        self._model: Any | None = None

    def _fit_arms(self, X: Any, treatment: np.ndarray, y: np.ndarray, **fit_params: Any) -> None:
        x_aug = _stack_treatment(X, treatment.astype(np.float32))
        self._model = deepcopy(self.model)
        self._model.fit(x_aug, y, **fit_params)

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        n = len(X)
        x_aug = _stack_treatment(X, np.full(n, float(arm), dtype=np.float32))
        return np.asarray(self._model.predict(x_aug)).reshape(-1)
