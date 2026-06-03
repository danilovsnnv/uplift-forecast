__all__ = ['BaseMetaUpliftModel']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold

from ._uplift_model import UpliftModel, _row_subset, _to_array, _to_numpy_1d


def _resolve_max_features(max_features: float | str | None, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        raise ValueError(f"max_features string must be 'sqrt' or 'log2'; got {max_features!r}.")
    if isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(f'max_features float must be in (0, 1]; got {max_features}.')
        return max(1, int(max_features * n_features))
    return max(1, min(int(max_features), n_features))


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(scores), dtype=int)
    if k <= 0:
        return out
    top = np.argsort(scores)[::-1][:k]
    out[top] = 1
    return out


def _oof_propensity(
    propensity_model: Any,
    X: np.ndarray | pd.DataFrame,
    t: np.ndarray,
    n_folds: int,
    propensity_clip: float,
    random_state: int,
) -> tuple[np.ndarray, Any]:
    """Cross-fit propensity scores; returns (clipped OOF scores, model refitted on all data)."""
    lo, hi = propensity_clip, 1.0 - propensity_clip
    n = len(t)
    if n_folds <= 1:
        fitted = deepcopy(propensity_model)
        fitted.fit(X, t)
        return np.clip(fitted.predict_proba(X)[:, 1], lo, hi), fitted
    out = np.empty(n, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(np.arange(n)):
        clf = deepcopy(propensity_model)
        clf.fit(_row_subset(X, train_idx), t[train_idx])
        out[test_idx] = clf.predict_proba(_row_subset(X, test_idx))[:, 1]
    fitted = deepcopy(propensity_model)
    fitted.fit(X, t)
    return np.clip(out, lo, hi), fitted


def _oof_predict_arm(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    fit_mask: np.ndarray,
    n_folds: int,
    random_state: int,
) -> np.ndarray:
    """Cross-fit an arm-specific outcome regressor over all rows."""
    n = len(y)
    if n_folds <= 1:
        est = deepcopy(model)
        est.fit(_row_subset(X, fit_mask), y[fit_mask])
        return np.asarray(est.predict(X)).reshape(-1)
    out = np.empty(n, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(np.arange(n)):
        arm_idx = train_idx[fit_mask[train_idx]]
        est = deepcopy(model)
        est.fit(_row_subset(X, arm_idx), y[arm_idx])
        out[test_idx] = np.asarray(est.predict(_row_subset(X, test_idx))).reshape(-1)
    return out


class BaseMetaUpliftModel(UpliftModel):
    """Base for classical meta-learner uplift models.

    Subclasses override two template methods:
    - ``_fit_estimators`` — train the internal sklearn-style estimator(s).
    - ``_predict_components`` — return (y0_pred, y1_pred) arrays.

    Input conversion, the uplift = y1 - y0 formula, and the return_components
    switch are handled here so concrete classes stay small.

    Args:
        alias (str): Optional display name used by UpliftForecast.
    """

    def __init__(self, alias: str | None = None):
        self.alias = alias
        self._fitted = False

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **fit_params: Any,
    ) -> 'BaseMetaUpliftModel':
        X_arr = _to_array(X)
        treatment_arr = _to_numpy_1d(treatment)
        y_arr = _to_numpy_1d(y)

        eval_arr = None
        if eval_set is not None:
            X_val, t_val, y_val = eval_set
            eval_arr = (_to_array(X_val), _to_numpy_1d(t_val), _to_numpy_1d(y_val))

        self._fit_estimators(X_arr, treatment_arr, y_arr, eval_set=eval_arr, **fit_params)
        self._fitted = True
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError(
                f'{type(self).__name__} has not been fitted yet. Call .fit() first.'
            )
        y0, y1 = self._predict_components(_to_array(X))
        uplift = y1 - y0
        if return_components:
            return uplift, y0, y1
        return uplift

    def _fit_estimators(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        raise NotImplementedError

    def _predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
