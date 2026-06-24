__all__ = ['XLearner']


from copy import deepcopy
from typing import Any

import numpy as np

from ..common._base_meta import _aligned_multiclass_proba, _BaseMultiArmLearner, _oof_predict_arm
from ..common._uplift_model import _row_subset


class XLearner(_BaseMultiArmLearner):
    """X-learner meta-learner (Künzel, Sekhon, Bickel & Yu, 2019; arXiv:1706.03461), binary or multi-arm.

    Two outcome models are fitted per arm: mu0 on control rows and mu1 on treated
    rows. Each is applied to the *opposite* arm to impute individual treatment
    effects: D1 = Y - mu0(X) for treated rows and D0 = mu1(X) - Y for control rows.
    Two effect models are then fitted to these imputed effects -- tau1 on (treated,
    D1) and tau0 on (control, D0) -- and combined by the propensity score:
    tau(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x).

    The imputation is leakage-free by construction (each arm's outcome model only
    scores rows it was never trained on); cross-fitting via ``n_folds`` further
    decorrelates the imputed effects from the outcome fit.

    With ``K`` arms the X-learner is extended per treated arm vs control (Zhao &
    Harinen, arXiv:1908.05372): for arm ``k`` it imputes against the control
    outcome ``mu0`` and the arm-``k`` outcome ``mu_k``, then combines the two
    imputed effects with the *relative* propensity ``g_k(x) = P(T=k | T in {0,k})``.
    ``predict`` returns ``[n, K-1]`` (one column per treated arm vs control),
    collapsing to a flat ``[n]`` in the binary case. For every arm ``predict``
    reports y0 = mu0(X) and y1 = mu0(X) + tau_k(x), so the reported uplift equals
    tau_k(x).

    Args:
        model: Base regressor for the control outcome mu0 (sklearn-style fit/predict).
        model_treated: Optional separate regressor for the treated outcomes mu_k.
            Defaults to a deepcopy of model (one independent copy per treated arm).
        effect_model: Optional regressor for the control-imputed effect tau0.
            Defaults to a deepcopy of model.
        effect_model_treated: Optional regressor for the treated-imputed effect tau1.
            Defaults to a deepcopy of effect_model (else model).
        propensity_model: Optional classifier with predict_proba for the per-arm
            propensities. If None, the global per-arm rates are used as constants.
        n_folds (int): Number of folds for cross-fitting the outcome models used to
            impute effects. If <= 1, each outcome model is fitted once on its arm and
            applied to the opposite arm.
        propensity_clip (float): Clip each propensity into [propensity_clip, 1 - propensity_clip].
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
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError(f'propensity_clip must be in (0, 0.5); got {propensity_clip}.')
        self.model = model
        self.model_treated = model_treated
        self.effect_model = effect_model
        self.effect_model_treated = effect_model_treated
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._outcome_model_control: Any | None = None
        self._effect_treated: dict[int, Any] = {}
        self._effect_control: dict[int, Any] = {}
        self._propensity_fitted: Any | None = None
        self._global_rates: dict[int, float] = {}

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        del eval_set, fit_params  # X-learner does not early-stop its nuisances
        t = treatment.astype(int)
        y = y.astype(np.float64)
        mask_ct = t == 0

        mu_oof = {
            arm: _oof_predict_arm(self._outcome_for(arm), X, y, t == arm, self.n_folds, self.random_state)
            for arm in self.arms_
        }

        self._effect_treated, self._effect_control = {}, {}
        for arm in self.arms_[1:]:
            mask_k = t == arm
            d1 = y[mask_k] - mu_oof[0][mask_k]
            d0 = mu_oof[arm][mask_ct] - y[mask_ct]
            est_treated = deepcopy(self.effect_model_treated or self.effect_model or self.model)
            est_treated.fit(_row_subset(X, mask_k), d1)
            est_control = deepcopy(self.effect_model or self.model)
            est_control.fit(_row_subset(X, mask_ct), d0)
            self._effect_treated[arm] = est_treated
            self._effect_control[arm] = est_control

        # Full-control fit reused for the reported y0 baseline component.
        self._outcome_model_control = deepcopy(self.model)
        self._outcome_model_control.fit(_row_subset(X, mask_ct), y[mask_ct])

        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            self._global_rates = {arm: float(np.clip((t == arm).mean(), lo, hi)) for arm in self.arms_}
            self._propensity_fitted = None
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(X, t)

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        mu0 = self._predict_outcome(self._outcome_model_control, X)
        if arm == 0:
            return mu0
        g = self._relative_propensity(X, arm)
        tau0 = np.asarray(self._effect_control[arm].predict(X)).reshape(-1)
        tau1 = np.asarray(self._effect_treated[arm].predict(X)).reshape(-1)
        return mu0 + g * tau0 + (1.0 - g) * tau1

    def _outcome_for(self, arm: int) -> Any:
        if arm == 0 or self.model_treated is None:
            return self.model
        return self.model_treated

    def _relative_propensity(self, X: Any, arm: int) -> np.ndarray:
        """Relative propensity ``g_k(x) = P(T=k | T in {0,k}) = e_k / (e_0 + e_k)``."""
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self._propensity_fitted is None:
            e0, ek = self._global_rates[0], self._global_rates[arm]
            return np.full(len(X), ek / (e0 + ek))
        proba = np.clip(_aligned_multiclass_proba(self._propensity_fitted, X, self.arms_), lo, hi)
        e0, ek = proba[:, 0], proba[:, self.arms_.index(arm)]
        return ek / (e0 + ek)
