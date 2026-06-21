__all__ = ['DRLearner']


from copy import deepcopy
from typing import Any

import numpy as np

from ..common._base_meta import (
    _BaseMultiArmLearner,
    _aligned_multiclass_proba,
    _oof_multiclass_propensity,
    _oof_predict_arm,
)
from ..common._uplift_model import _row_subset


class DRLearner(_BaseMultiArmLearner):
    """Doubly-robust meta-learner (DR-learner, Kennedy 2023, arXiv:2004.14497), binary or multi-arm.

    Builds an AIPW pseudo-outcome from cross-fitted outcome and propensity nuisance estimates,
    then regresses it on the features to estimate the conditional treatment effect ``tau(x)``.
    The estimator is doubly robust: consistent if *either* the outcome models or the propensity
    model is correctly specified.

    For every treated arm ``k`` the pseudo-outcome contrasting arm ``k`` with the control arm is

    ``phi_k = mu_k(X) - mu_0(X) + 1{T=k}(Y - mu_k(X)) / e_k(X) - 1{T=0}(Y - mu_0(X)) / e_0(X)``,

    where ``e_a(X) = P(T=a|X)`` and ``mu_a`` are computed out-of-fold via K-fold cross-fitting.
    With a single treated arm this reduces exactly to the binary DR-learner; ``predict`` then
    returns a flat ``[n]`` uplift instead of the ``[n, K-1]`` multi-arm array. A control-arm
    outcome model fitted on all control rows provides the baseline ``mu_0(x)`` for
    ``predict(return_components=True)``, so ``uplift == tau`` holds.

    With an ``eval_set``, each effect model early-stops on the validation AIPW pseudo-outcome
    ``phi_k`` (scored from all-rows outcome models + the propensity, so the eval pool is
    leakage-free). The cross-fitted outcome/propensity nuisances themselves are not early-stopped
    — cross-fitting refits them per fold at fixed iterations.

    Args:
        outcome_model: Base regressor for the control outcome mu_0 (sklearn-style). A
            deepcopy is cross-fitted per fold and once on all control rows for the baseline.
        effect_model: Final regressor fitted per treated arm on (X, pseudo-outcome) to
            estimate tau_k. One deepcopy is trained per treated arm.
        outcome_model_treated: Optional separate regressor for the treated outcomes mu_k.
            Defaults to a deepcopy of outcome_model.
        propensity_model: Optional multiclass classifier with predict_proba for the
            propensities e_a(X). If None, per-arm global rates e_a = mean(T == a) are used.
        n_folds (int): Number of cross-fitting folds. Values <= 1 disable cross-fitting
            (nuisances are fit on all rows and scored in-sample).
        propensity_clip (float): Each propensity is clipped to
            [propensity_clip, 1 - propensity_clip] to bound the inverse-propensity weights.
        random_state (int): Seed for the KFold shuffle.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        outcome_model: Any,
        effect_model: Any,
        outcome_model_treated: Any | None = None,
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError(f'propensity_clip must be in (0, 0.5); got {propensity_clip}.')
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.outcome_model_treated = outcome_model_treated
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._effect_models: dict[int, Any] = {}
        self._outcome_model_control: Any | None = None
        self._outcome_full: dict[int, Any] = {}
        self._global_rates: dict[int, float] = {}
        self._propensity_model: Any | None = None

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        t = treatment.astype(int)
        mu = {
            arm: _oof_predict_arm(self._outcome_for(arm), X, y, t == arm, self.n_folds, self.random_state)
            for arm in self.arms_
        }
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            self._global_rates = {arm: float(np.clip((t == arm).mean(), lo, hi)) for arm in self.arms_}
            e = {arm: np.full(len(t), self._global_rates[arm]) for arm in self.arms_}
        else:
            proba, self._propensity_model = _oof_multiclass_propensity(
                self.propensity_model, X, t, self.arms_, self.n_folds, self.propensity_clip, self.random_state,
            )
            e = {arm: proba[:, j] for j, arm in enumerate(self.arms_)}

        # Baseline mu_0(X) fitted on ALL control rows (separate from the OOF nuisances) so
        # predict() can return mu_0(X) and mu_0(X) + tau_k(X). When an eval_set is given, the
        # treated-arm outcomes are refit on all rows too, only to score the validation AIPW
        # pseudo-outcome that the effect model early-stops against (so the eval pool is leakage-free).
        self._outcome_full = {0: deepcopy(self._outcome_for(0))}
        self._outcome_full[0].fit(_row_subset(X, t == 0), y[t == 0])
        self._outcome_model_control = self._outcome_full[0]
        if eval_set is not None:
            for arm in self.arms_[1:]:
                self._outcome_full[arm] = deepcopy(self._outcome_for(arm))
                self._outcome_full[arm].fit(_row_subset(X, t == arm), y[t == arm])
        val_phi = self._validation_pseudo_outcomes(eval_set) if eval_set is not None else None

        mu0, e0, ind0 = mu[0], e[0], (t == 0).astype(float)
        self._effect_models = {}
        for arm in self.arms_[1:]:
            ind_k = (t == arm).astype(float)
            phi = mu[arm] - mu0 + ind_k * (y - mu[arm]) / e[arm] - ind0 * (y - mu0) / e0
            kwargs = dict(fit_params)
            if val_phi is not None:
                kwargs.setdefault('eval_set', (eval_set[0], val_phi[arm]))
            est = deepcopy(self.effect_model)
            est.fit(X, phi, **kwargs)
            self._effect_models[arm] = est

    def _validation_pseudo_outcomes(self, eval_set: tuple) -> dict[int, np.ndarray]:
        """AIPW pseudo-outcome phi_k on the validation rows, used as the effect model's eval target."""
        x_val, t_val, y_val = eval_set
        mu_val = {arm: self._predict_outcome(self._outcome_full[arm], x_val) for arm in self.arms_}
        e_val = self._propensity_on(x_val)
        mu0, e0, ind0 = mu_val[0], e_val[0], (t_val == 0).astype(float)
        out = {}
        for arm in self.arms_[1:]:
            ind_k = (t_val == arm).astype(float)
            out[arm] = mu_val[arm] - mu0 + ind_k * (y_val - mu_val[arm]) / e_val[arm] - ind0 * (y_val - mu0) / e0
        return out

    def _propensity_on(self, X: Any) -> dict[int, np.ndarray]:
        """Per-arm propensity e_a on the given rows, reusing the training nuisance (no refit)."""
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        n = len(X)
        if self.propensity_model is None:
            return {arm: np.full(n, self._global_rates[arm]) for arm in self.arms_}
        proba = np.clip(_aligned_multiclass_proba(self._propensity_model, X, self.arms_), lo, hi)
        return {arm: proba[:, j] for j, arm in enumerate(self.arms_)}

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        mu0 = self._predict_outcome(self._outcome_model_control, X)
        if arm == 0:
            return mu0
        return mu0 + np.asarray(self._effect_models[arm].predict(X)).reshape(-1)

    def _outcome_for(self, arm: int) -> Any:
        if arm == 0 or self.outcome_model_treated is None:
            return self.outcome_model
        return self.outcome_model_treated
