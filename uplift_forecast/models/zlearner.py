__all__ = ['ZLearner']


from copy import deepcopy
from typing import Any

import numpy as np

from ..common._base_meta import _BaseMultiArmLearner, _oof_propensity
from ..common._uplift_model import _row_subset


class ZLearner(_BaseMultiArmLearner):
    """Z-learner / transformed-outcome meta-learner (Athey & Imbens, 2016, PNAS), binary or multi-arm.

    Reduces CATE estimation to a single regression on the inverse-propensity
    transformed (Horvitz-Thompson) outcome
    ``Z = T * Y / e(X) - (1 - T) * Y / (1 - e(X))``, which satisfies E[Z | X] = tau(X).
    The effect model is regressed on (X, Z); its prediction is the estimated CATE.

    With ``K`` arms each treated arm vs control is transformed on its own ``{0, k}``
    subset, yielding one effect model per arm. ``predict`` returns ``[n, K-1]`` (one
    column per treated arm vs control), collapsing to a flat ``[n]`` in the binary
    case. ``predict`` reports y0 = 0 and y1 = tau_k(x), so the reported uplift equals
    tau_k(x).

    Args:
        effect_model: Regressor fitted on (X, Z) to estimate tau(x) (sklearn-style).
        propensity_model: Optional classifier with predict_proba for e(x)=P(T=1|X).
            If None, the per-arm treatment rate is used as a constant.
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
        super().__init__(alias=alias)
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError(f'propensity_clip must be in (0, 0.5); got {propensity_clip}.')
        self.effect_model = effect_model
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
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
        """Transformed-outcome effect model for a single treated-arm-vs-control subset."""
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            e = np.full(len(t), np.clip(float(t.mean()), lo, hi))
        else:
            e, _ = _oof_propensity(
                self.propensity_model, X, t, self.n_folds, self.propensity_clip, self.random_state,
            )
        z = t * y / e - (1.0 - t) * y / (1.0 - e)
        effect_fitted = deepcopy(self.effect_model)
        effect_fitted.fit(X, z, **fit_params)
        return effect_fitted

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        if arm == 0:
            return np.zeros(len(X), dtype=np.float64)
        return np.asarray(self._effect[arm].predict(X)).reshape(-1)
