__all__ = ['RLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..common._base_meta import _BaseMultiArmLearner
from ..common._uplift_model import UpliftModel, _row_subset


def _oof_predict(estimator: Any, X: np.ndarray | pd.DataFrame, y: np.ndarray, kf: KFold) -> np.ndarray:
    """Cross-fitted out-of-fold outcome means: each row is predicted by a clone never trained on it."""
    oof = np.empty(len(y), dtype=np.float64)
    for train_idx, test_idx in kf.split(np.arange(len(y))):
        clone = deepcopy(estimator)
        clone.fit(_row_subset(X, train_idx), y[train_idx])
        oof[test_idx] = UpliftModel._predict_outcome(clone, _row_subset(X, test_idx))
    return oof


def _oof_predict_proba(classifier: Any, X: np.ndarray | pd.DataFrame, t: np.ndarray, kf: KFold) -> np.ndarray:
    """Cross-fitted out-of-fold P(treatment=1 | X)."""
    oof = np.empty(len(t), dtype=np.float64)
    for train_idx, test_idx in kf.split(np.arange(len(t))):
        clone = deepcopy(classifier)
        clone.fit(_row_subset(X, train_idx), t[train_idx])
        oof[test_idx] = np.asarray(clone.predict_proba(_row_subset(X, test_idx)))[:, 1]
    return oof


class RLearner(_BaseMultiArmLearner):
    """R-learner meta-learner (Nie & Wager, 2021; arXiv:1712.04912), binary or multi-arm.

    Estimates the CATE tau(x) via the Robinson residualization. The outcome model
    m(x)=E[Y|X] and propensity e(x)=E[T|X] are fitted with cross-fitting, then the
    effect model is fitted to minimize the R-loss sum_i (r_y_i - r_t_i * tau(x_i))^2,
    solved as a weighted regression of target r_y/r_t with weights r_t^2.

    With ``K`` arms each treated arm vs control is residualized on its own ``{0, k}``
    subset (Zhao & Harinen, arXiv:1908.05372), yielding one effect model per arm.
    ``predict`` returns ``[n, K-1]`` (one column per treated arm vs control),
    collapsing to a flat ``[n]`` in the binary case. Because there is no per-arm
    outcome model, ``predict`` reports y0 = 0 and y1 = tau_k(x), so the reported
    uplift equals tau_k(x).

    Args:
        outcome_model: Regressor for m(x)=E[Y|X], sklearn-style fit/predict.
        effect_model: Regressor for tau(x); its `fit` must accept `sample_weight`.
        propensity_model: Optional classifier with `predict_proba` for e(x)=E[T|X].
            If None, the per-arm treatment rate is used as a constant.
        n_folds (int): Number of folds for cross-fitting m(x) and e(x). If <= 1,
            nuisances are fitted on all rows and predicted in-sample.
        propensity_clip (float): Clip e(x) into [propensity_clip, 1 - propensity_clip].
        residual_threshold (float): Drop rows whose |r_t| is below this before fitting
            the effect model, to avoid dividing by a near-zero treatment residual.
        random_state (int): Seed for the KFold shuffle.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        outcome_model: Any,
        effect_model: Any,
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        residual_threshold: float = 1e-2,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.residual_threshold = residual_threshold
        self.random_state = random_state
        self._effect: dict[int, Any] = {}

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        del eval_set
        t = treatment.astype(int)
        y = y.astype(np.float64)
        self._effect = {}
        for arm in self.arms_[1:]:
            mask = (t == 0) | (t == arm)
            sub_t = (t[mask] == arm).astype(np.float64)
            self._effect[arm] = self._fit_effect(_row_subset(X, mask), sub_t, y[mask], **fit_params)

    def _fit_effect(self, X: Any, t: np.ndarray, y: np.ndarray, **fit_params: Any) -> Any:
        """Robinson-residualized effect model for a single treated-arm-vs-control subset."""
        if self.n_folds and self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            m_oof = _oof_predict(self.outcome_model, X, y, kf)
            e_oof = (
                np.full(len(t), float(t.mean()))
                if self.propensity_model is None
                else _oof_predict_proba(self.propensity_model, X, t, kf)
            )
        else:
            outcome_fitted = deepcopy(self.outcome_model)
            outcome_fitted.fit(X, y)
            m_oof = self._predict_outcome(outcome_fitted, X)
            if self.propensity_model is None:
                e_oof = np.full(len(t), float(t.mean()))
            else:
                propensity_fitted = deepcopy(self.propensity_model)
                propensity_fitted.fit(X, t)
                e_oof = np.asarray(propensity_fitted.predict_proba(X))[:, 1]

        e_oof = np.clip(e_oof, self.propensity_clip, 1.0 - self.propensity_clip)
        r_y = y - m_oof
        r_t = t - e_oof

        keep = np.abs(r_t) >= self.residual_threshold
        if not keep.any():
            raise ValueError(
                'All treatment residuals are below residual_threshold='
                f'{self.residual_threshold}; the effect model cannot be fitted. '
                'Lower residual_threshold or check the propensity model.'
            )
        target = r_y[keep] / r_t[keep]
        weight = r_t[keep] ** 2
        effect_fitted = deepcopy(self.effect_model)
        effect_fitted.fit(_row_subset(X, keep), target, sample_weight=weight)
        return effect_fitted

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        if arm == 0:
            return np.zeros(len(X), dtype=np.float64)
        return np.asarray(self._effect[arm].predict(X)).reshape(-1)
