__all__ = ['BaseMetaUpliftModel']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold

from ._uplift_model import UpliftModel, _row_subset, _to_array, _to_numpy_1d


def _stack_treatment(x: np.ndarray | pd.DataFrame, t: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Prepend a treatment-arm column to ``x`` (integer arm for multi-arm, 0/1 for binary)."""
    t = t.reshape(-1, 1)
    if isinstance(x, pd.DataFrame):
        out = x.copy()
        out.insert(0, '__treatment__', t.ravel())
        return out
    return np.hstack([t, x])


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


def _aligned_multiclass_proba(clf: Any, rows: np.ndarray | pd.DataFrame, arms: list[int]) -> np.ndarray:
    """Per-arm probabilities ``P(T=arm|rows)`` aligned to ``arms`` order via ``clf.classes_``.

    Columns follow the ``arms`` order; an arm absent from ``clf.classes_`` (e.g. missing from a
    fold) is left at probability 0.
    """
    arm_to_col = {arm: j for j, arm in enumerate(arms)}
    proba = np.asarray(clf.predict_proba(rows))
    out = np.zeros((proba.shape[0], len(arms)), dtype=np.float64)
    for col, cls in enumerate(clf.classes_):
        if int(cls) in arm_to_col:
            out[:, arm_to_col[int(cls)]] = proba[:, col]
    return out


def _oof_multiclass_propensity(
    propensity_model: Any,
    X: np.ndarray | pd.DataFrame,
    t: np.ndarray,
    arms: list[int],
    n_folds: int,
    propensity_clip: float,
    random_state: int,
) -> tuple[np.ndarray, Any]:
    """Cross-fit multinomial propensities P(T=arm|X) for every arm.

    Returns ``(scores, fitted)`` where ``scores`` is a clipped ``[n, len(arms)]``
    matrix whose columns follow the ``arms`` order, and ``fitted`` is the model
    refitted on all rows. Columns are mapped via ``classifier.classes_`` so that a
    fold missing an arm leaves that arm's probability at 0 (clipped up to the floor).
    """
    lo, hi = propensity_clip, 1.0 - propensity_clip
    n = len(t)
    k = len(arms)

    if n_folds <= 1:
        fitted = deepcopy(propensity_model)
        fitted.fit(X, t)
        return np.clip(_aligned_multiclass_proba(fitted, X, arms), lo, hi), fitted
    out = np.empty((n, k), dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(np.arange(n)):
        clf = deepcopy(propensity_model)
        clf.fit(_row_subset(X, train_idx), t[train_idx])
        out[test_idx] = _aligned_multiclass_proba(clf, _row_subset(X, test_idx), arms)
    fitted = deepcopy(propensity_model)
    fitted.fit(X, t)
    return np.clip(out, lo, hi), fitted


def _ipw_weights_binary(propensity: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Inverse-propensity sample weights for a binary treatment.

    Treated rows are weighted by ``1 / e(x)`` and control rows by ``1 / (1 - e(x))`` so each
    arm's weighted sample approximates the full population (Horvitz-Thompson IPW). ``propensity``
    must already be clipped away from 0/1 to bound the weights.
    """
    t = treatment.astype(int)
    return np.where(t == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))


def _ipw_weights_multiclass(proba: np.ndarray, treatment: np.ndarray, arms: list[int]) -> np.ndarray:
    """Generalized-propensity inverse weights ``1 / P(T=a_i|x_i)`` for the assigned arm.

    Multi-arm IPTW following McCaffrey et al. (2013): each row is weighted by the inverse of
    the (clipped) propensity of the treatment it actually received. ``proba`` columns follow
    the ``arms`` order, as returned by :func:`_oof_multiclass_propensity`.
    """
    arm_to_col = {arm: j for j, arm in enumerate(arms)}
    cols = np.fromiter((arm_to_col[int(a)] for a in treatment), dtype=int, count=len(treatment))
    return 1.0 / proba[np.arange(len(treatment)), cols]


def _with_sample_weight(fit_params: dict[str, Any], weight: np.ndarray) -> dict[str, Any]:
    """Fold ``weight`` into ``fit_params['sample_weight']``, multiplying any caller-supplied one."""
    out = dict(fit_params)
    existing = out.get('sample_weight')
    out['sample_weight'] = weight if existing is None else np.asarray(existing) * weight
    return out


def _oof_predict_arm(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    fit_mask: np.ndarray,
    n_folds: int,
    random_state: int,
) -> np.ndarray:
    """Cross-fit an arm-specific outcome estimator over all rows."""
    n = len(y)
    if n_folds <= 1:
        est = deepcopy(model)
        est.fit(_row_subset(X, fit_mask), y[fit_mask])
        return UpliftModel._predict_outcome(est, X)
    out = np.empty(n, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(np.arange(n)):
        arm_idx = train_idx[fit_mask[train_idx]]
        est = deepcopy(model)
        est.fit(_row_subset(X, arm_idx), y[arm_idx])
        out[test_idx] = UpliftModel._predict_outcome(est, _row_subset(X, test_idx))
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


class _BaseMultiArmLearner(BaseMetaUpliftModel):
    """Shared logic for arm-decomposable meta-learners (S/T/DR), control arm coded 0.

    Generalises the binary meta-learner to ``K`` discrete arms. ``predict`` returns one
    uplift column per treated arm vs control as a ``[n, K-1]`` array; with a single treated
    arm (the binary case) it collapses to a flat ``[n]`` array, so binary behaviour is
    identical to a plain two-arm fit. Subclasses override ``_fit_arms`` / ``_predict_arm``
    instead of the binary ``_fit_estimators`` / ``_predict_components`` template.

    Passing ``eval_set=(X_val, treatment_val, y_val)`` to ``fit`` forwards a per-arm validation
    pool to each base estimator's ``fit`` (so ``early_stopping_rounds`` works); the base estimator
    must accept an ``eval_set`` argument.
    """

    def __init__(self, alias: str | None = None):
        super().__init__(alias=alias)
        self.arms_: list[int] = []

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **fit_params: Any,
    ) -> '_BaseMultiArmLearner':
        x = _to_array(X)
        t = _to_numpy_1d(treatment).astype(int)
        y_arr = _to_numpy_1d(y)
        self.arms_ = sorted(set(t.tolist()))
        if self.arms_[0] != 0:
            raise ValueError(f'treatment must include a control arm coded 0; got arms {self.arms_}.')
        eval_arr = None
        if eval_set is not None:
            x_val, t_val, y_val = eval_set
            eval_arr = (_to_array(x_val), _to_numpy_1d(t_val).astype(int), _to_numpy_1d(y_val))
        self._fit_arms(x, t, y_arr, eval_set=eval_arr, **fit_params)
        self._fitted = True
        return self

    def predict(
        self,
        X: ArrayLike,
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

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        raise NotImplementedError

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        raise NotImplementedError


def _multi_ipw_weights(learner: _BaseMultiArmLearner, X: Any, treatment: np.ndarray) -> np.ndarray | None:
    """Cross-fit multi-arm propensities and return assigned-arm IPW weights, or None if disabled.

    Stores the all-rows-refitted propensity model on ``learner._propensity_model``.
    """
    if learner.propensity_model is None:
        return None
    proba, learner._propensity_model = _oof_multiclass_propensity(
        learner.propensity_model, X, treatment, learner.arms_,
        learner.n_folds, learner.propensity_clip, learner.random_state,
    )
    return _ipw_weights_multiclass(proba, treatment, learner.arms_)
