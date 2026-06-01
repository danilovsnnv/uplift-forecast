__all__ = ['ZLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..common._base_meta import BaseMetaUpliftModel, _oof_propensity


class ZLearner(BaseMetaUpliftModel):
    """Z-learner / transformed-outcome meta-learner (Athey & Imbens, 2016, PNAS).

    Reduces CATE estimation to a single regression on the inverse-propensity
    transformed (Horvitz-Thompson) outcome
    ``Z = T * Y / e(X) - (1 - T) * Y / (1 - e(X))``, which satisfies E[Z | X] = tau(X).
    The effect model is regressed on (X, Z); its prediction is the estimated CATE.

    The propensity e(X) is estimated out-of-fold and clipped to bound the
    inverse-propensity weights. ``predict`` reports y0 = 0 and y1 = tau(x), so the
    reported uplift equals tau(x).

    Args:
        effect_model: Regressor fitted on (X, Z) to estimate tau(x) (sklearn-style).
        propensity_model: Optional classifier with predict_proba for e(x)=P(T=1|X).
            If None, the global treatment rate mean(treatment) is used as a constant.
        n_folds (int): Number of folds for cross-fitting e(x). If <= 1, the propensity
            model is fitted on all rows and scored in-sample.
        propensity_clip (float): Clip e(x) into [propensity_clip, 1 - propensity_clip].
        random_state (int): Seed for the KFold shuffle.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        effect_model: Any,
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super(ZLearner, self).__init__(alias=alias)
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError(f'propensity_clip must be in (0, 0.5); got {propensity_clip}.')
        self.effect_model = effect_model
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._effect_fitted = None
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
        t = treatment.astype(np.float64)
        y = y.astype(np.float64)
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            self._global_rate = float(t.mean())
            e = np.full(len(t), np.clip(self._global_rate, lo, hi))
        else:
            e, self._propensity_fitted = _oof_propensity(
                self.propensity_model, X, t, self.n_folds, self.propensity_clip, self.random_state,
            )
        z = t * y / e - (1.0 - t) * y / (1.0 - e)
        self._effect_fitted = deepcopy(self.effect_model)
        self._effect_fitted.fit(X, z, **fit_params)

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(self._effect_fitted.predict(X)).reshape(-1)
        return np.zeros(len(tau), dtype=np.float64), tau
