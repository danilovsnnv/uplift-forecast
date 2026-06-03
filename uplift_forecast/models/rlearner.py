__all__ = ['RLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..common._base_meta import BaseMetaUpliftModel
from ..common._uplift_model import _row_subset


def _oof_predict(estimator: Any, X: np.ndarray | pd.DataFrame, y: np.ndarray, kf: KFold) -> np.ndarray:
    """Cross-fitted out-of-fold predictions: each row is predicted by a clone never trained on it."""
    oof = np.empty(len(y), dtype=np.float64)
    for train_idx, test_idx in kf.split(np.arange(len(y))):
        clone = deepcopy(estimator)
        clone.fit(_row_subset(X, train_idx), y[train_idx])
        oof[test_idx] = np.asarray(clone.predict(_row_subset(X, test_idx))).reshape(-1)
    return oof


def _oof_predict_proba(classifier: Any, X: np.ndarray | pd.DataFrame, t: np.ndarray, kf: KFold) -> np.ndarray:
    """Cross-fitted out-of-fold P(treatment=1 | X)."""
    oof = np.empty(len(t), dtype=np.float64)
    for train_idx, test_idx in kf.split(np.arange(len(t))):
        clone = deepcopy(classifier)
        clone.fit(_row_subset(X, train_idx), t[train_idx])
        oof[test_idx] = np.asarray(clone.predict_proba(_row_subset(X, test_idx)))[:, 1]
    return oof


class RLearner(BaseMetaUpliftModel):
    """R-learner meta-learner (Nie & Wager, 2021; arXiv:1712.04912).

    Estimates the CATE tau(x) via the Robinson residualization. The outcome model
    m(x)=E[Y|X] and propensity e(x)=E[T|X] are fitted with cross-fitting, then the
    effect model is fitted to minimize the R-loss sum_i (r_y_i - r_t_i * tau(x_i))^2,
    solved as a weighted regression of target r_y/r_t with weights r_t^2.

    Because there is no per-arm outcome model, `predict` reports `y0 = 0` and
    `y1 = tau(x)`, so the reported uplift equals tau(x).

    Args:
        outcome_model: Regressor for m(x)=E[Y|X], sklearn-style fit/predict.
        effect_model: Regressor for tau(x); its `fit` must accept `sample_weight`.
        propensity_model: Optional classifier with `predict_proba` for e(x)=E[T|X].
            If None, the global treatment rate mean(treatment) is used as a constant.
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
        self._outcome_fitted = None
        self._propensity_fitted = None
        self._effect_fitted = None

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        t = treatment.astype(np.float64)
        y = y.astype(np.float64)

        # Nuisances are stored fitted-on-all-data; cross-fitting only supplies the
        # out-of-fold residuals used to train the effect model (Nie & Wager, 2021).
        self._outcome_fitted = deepcopy(self.outcome_model)
        self._outcome_fitted.fit(X, y)
        if self.propensity_model is None:
            self._propensity_fitted = float(t.mean())
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(X, t)

        if self.n_folds and self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            m_oof = _oof_predict(self.outcome_model, X, y, kf)
            if self.propensity_model is None:
                e_oof = np.full(len(t), float(t.mean()))
            else:
                e_oof = _oof_predict_proba(self.propensity_model, X, t, kf)
        else:
            m_oof = np.asarray(self._outcome_fitted.predict(X)).reshape(-1)
            if self.propensity_model is None:
                e_oof = np.full(len(t), float(t.mean()))
            else:
                e_oof = np.asarray(self._propensity_fitted.predict_proba(X))[:, 1]

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

        self._effect_fitted = deepcopy(self.effect_model)
        self._effect_fitted.fit(_row_subset(X, keep), target, sample_weight=weight)

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(self._effect_fitted.predict(X)).reshape(-1)
        y0 = np.zeros(len(tau), dtype=np.float64)
        return y0, tau
