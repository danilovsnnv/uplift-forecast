__all__ = ['DRLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..common._base_meta import BaseMetaUpliftModel, _oof_predict_arm, _oof_propensity
from ..common._uplift_model import _row_subset


class DRLearner(BaseMetaUpliftModel):
    """Doubly-robust meta-learner (DR-learner, Kennedy 2023, arXiv:2004.14497).

    Builds an AIPW pseudo-outcome from cross-fitted outcome and propensity
    nuisance estimates, then regresses it on the features to estimate the
    conditional treatment effect tau(x). The estimator is consistent if *either*
    the outcome models or the propensity model is correctly specified.

    The pseudo-outcome is
    ``phi = mu1(X) - mu0(X) + T * (Y - mu1(X)) / e(X) - (1 - T) * (Y - mu0(X)) / (1 - e(X))``,
    where mu0/mu1/e are computed out-of-fold via K-fold cross-fitting to avoid
    overfitting bias. The final effect model is fit on (X, phi); tau(x) is its
    prediction. A separate mu0 estimator is fit on all control rows to provide the
    baseline component returned by ``predict(return_components=True)`` so that
    ``uplift == tau``.

    Args:
        outcome_model: Base regressor for the control outcome mu0 (sklearn-style).
            A deepcopy is fitted on control rows in every fold and once on all
            control rows for the baseline component.
        effect_model: Final regressor fitted on (X, pseudo-outcome) to estimate tau.
        outcome_model_treated: Optional separate regressor for the treated outcome
            mu1. Defaults to a deepcopy of outcome_model.
        propensity_model: Optional classifier with predict_proba for the propensity
            e(X). If None, a global treatment rate scalar e = mean(treatment) is used.
        n_folds (int): Number of cross-fitting folds. Values <= 1 disable
            cross-fitting (nuisances are fit on all rows and scored in-sample).
        propensity_clip (float): Propensity scores are clipped to
            [propensity_clip, 1 - propensity_clip] to bound the inverse-propensity
            weights (positivity).
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
        super(DRLearner, self).__init__(alias=alias)
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError(
                f'propensity_clip must be in (0, 0.5); got {propensity_clip}.'
            )
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.outcome_model_treated = outcome_model_treated
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._effect_model = None
        self._outcome_model_control = None
        self._global_rate = None
        self._propensity_model = None

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        t = treatment.astype(int)
        mu0 = _oof_predict_arm(self.outcome_model, X, y, t == 0, self.n_folds, self.random_state)
        mu1 = _oof_predict_arm(
            self.outcome_model_treated or self.outcome_model, X, y, t == 1, self.n_folds, self.random_state,
        )
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            self._global_rate = float(t.mean())
            e = np.full(len(t), np.clip(self._global_rate, lo, hi))
        else:
            e, self._propensity_model = _oof_propensity(
                self.propensity_model, X, t, self.n_folds, self.propensity_clip, self.random_state,
            )
        phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self._effect_model = deepcopy(self.effect_model)
        self._effect_model.fit(X, phi, **fit_params)

        # Baseline component fitted on ALL control rows (separate from the OOF
        # nuisances) so that predict() can return mu0(X) and mu0(X) + tau(X).
        mask_ct = t == 0
        self._outcome_model_control = deepcopy(self.outcome_model)
        self._outcome_model_control.fit(_row_subset(X, mask_ct), y[mask_ct])

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y0 = np.asarray(self._outcome_model_control.predict(X)).reshape(-1)
        tau = np.asarray(self._effect_model.predict(X)).reshape(-1)
        return y0, y0 + tau
