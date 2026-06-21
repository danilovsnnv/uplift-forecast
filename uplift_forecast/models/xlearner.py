__all__ = ['XLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..common._base_meta import BaseMetaUpliftModel, _oof_predict_arm
from ..common._uplift_model import _row_subset


class XLearner(BaseMetaUpliftModel):
    """X-learner meta-learner (Künzel, Sekhon, Bickel & Yu, 2019; arXiv:1706.03461).

    Two outcome models are fitted per arm: mu0 on control rows and mu1 on treated
    rows. Each is applied to the *opposite* arm to impute individual treatment
    effects: D1 = Y - mu0(X) for treated rows and D0 = mu1(X) - Y for control rows.
    Two effect models are then fitted to these imputed effects -- tau1 on (treated,
    D1) and tau0 on (control, D0) -- and combined by the propensity score:
    tau(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x).

    The imputation is leakage-free by construction (each arm's outcome model only
    scores rows it was never trained on); cross-fitting via ``n_folds`` further
    decorrelates the imputed effects from the outcome fit.

    ``predict`` reports y0 = mu0(X) and y1 = mu0(X) + tau(x), so the reported uplift
    equals tau(x).

    Args:
        model: Base regressor for the control outcome mu0 (sklearn-style fit/predict).
        model_treated: Optional separate regressor for the treated outcome mu1.
            Defaults to a deepcopy of model.
        effect_model: Optional regressor for the control-imputed effect tau0.
            Defaults to a deepcopy of model.
        effect_model_treated: Optional regressor for the treated-imputed effect tau1.
            Defaults to a deepcopy of effect_model (else model).
        propensity_model: Optional classifier with predict_proba for e(x)=P(T=1|X).
            If None, the global treatment rate mean(treatment) is used as a constant.
        n_folds (int): Number of folds for cross-fitting the outcome models used to
            impute effects. If <= 1, each outcome model is fitted once on its arm and
            applied to the opposite arm.
        propensity_clip (float): Clip e(x) into [propensity_clip, 1 - propensity_clip].
        random_state (int): Seed for the KFold shuffle.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        model: Any,
        model_treated: Any | None = None,
        effect_model: Any | None = None,
        effect_model_treated: Any | None = None,
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        self.model = model
        self.model_treated = model_treated
        self.effect_model = effect_model
        self.effect_model_treated = effect_model_treated
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._outcome_model_control = None
        self._effect_model_control = None
        self._effect_model_treated = None
        self._propensity_fitted = None
        self._global_rate = None

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        t = treatment.astype(int)
        y = y.astype(np.float64)
        mask_ct = t == 0
        mask_tr = t == 1
        if not mask_ct.any() or not mask_tr.any():
            raise ValueError('XLearner requires both treated and control samples in the data.')

        mu0_oof = _oof_predict_arm(self.model, X, y, mask_ct, self.n_folds, self.random_state)
        mu1_oof = _oof_predict_arm(self.model_treated or self.model, X, y, mask_tr, self.n_folds, self.random_state)

        d1 = y[mask_tr] - mu0_oof[mask_tr]
        d0 = mu1_oof[mask_ct] - y[mask_ct]

        self._effect_model_treated = deepcopy(self.effect_model_treated or self.effect_model or self.model)
        self._effect_model_treated.fit(_row_subset(X, mask_tr), d1)
        self._effect_model_control = deepcopy(self.effect_model or self.model)
        self._effect_model_control.fit(_row_subset(X, mask_ct), d0)

        # Full-control fit reused for the reported y0 baseline component.
        self._outcome_model_control = deepcopy(self.model)
        self._outcome_model_control.fit(_row_subset(X, mask_ct), y[mask_ct])

        if self.propensity_model is None:
            self._global_rate = float(t.mean())
            self._propensity_fitted = None
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(X, t)

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        tau0 = np.asarray(self._effect_model_control.predict(X)).reshape(-1)
        tau1 = np.asarray(self._effect_model_treated.predict(X)).reshape(-1)
        e = self._propensity(X, len(tau0))
        tau = e * tau0 + (1.0 - e) * tau1
        y0 = self._predict_outcome(self._outcome_model_control, X)
        return y0, y0 + tau

    def _propensity(self, X: np.ndarray | pd.DataFrame, n: int) -> np.ndarray:
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self._propensity_fitted is None:
            return np.full(n, np.clip(self._global_rate, lo, hi))
        return np.clip(np.asarray(self._propensity_fitted.predict_proba(X))[:, 1], lo, hi)
