__all__ = ['SLearner']


from copy import deepcopy
from typing import Any

import numpy as np

from ..common._base_meta import (
    _BaseMultiArmLearner,
    _multi_ipw_weights,
    _stack_treatment,
    _with_sample_weight,
)


class SLearner(_BaseMultiArmLearner):
    """Single-model meta-learner (S-learner), binary or multi-arm.

    Trains one base estimator on ``[treatment_arm, X] -> y`` with the (integer) treatment
    arm as an extra feature, then contrasts predictions with the arm forced to ``k`` vs the
    control arm ``0``. With a binary treatment (arms ``{0, 1}``) ``predict`` returns a flat
    ``[n]`` uplift; with ``K`` arms it returns ``[n, K-1]`` (one column per treated arm vs
    control).

    When ``propensity_model`` is given, the regression is fit with inverse-propensity sample
    weights. For binary treatment these are ``1 / e(x)`` for treated rows and ``1 / (1 - e(x))``
    for control rows; for multi-arm treatment each row is weighted by ``1 / P(T=a_i|x_i)`` (IPTW
    per McCaffrey et al., 2013), reducing treatment-assignment imbalance. The propensities are
    cross-fitted on the original features (not the arm-augmented matrix); the base estimator must
    accept a ``sample_weight`` fit argument.

    Args:
        model: Base estimator with sklearn-style fit/predict (CatBoost, lgbm, sklearn).
        propensity_model: Optional classifier with ``predict_proba`` for the per-arm
            propensities. If None, the model is fit unweighted (plain S-learner).
        n_folds (int): Number of folds for cross-fitting the propensity scores used as weights.
            Values <= 1 fit the propensity model on all rows and score in-sample.
        propensity_clip (float): Clip each propensity into ``[propensity_clip, 1 - propensity_clip]``
            to bound the inverse-propensity weights (positivity).
        random_state (int): Seed for the KFold shuffle.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        model: Any,
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if propensity_model is not None and not 0.0 < propensity_clip < 0.5:
            raise ValueError(f'propensity_clip must be in (0, 0.5); got {propensity_clip}.')
        self.model = model
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state
        self._model: Any | None = None
        self._propensity_model: Any | None = None

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        x_aug = _stack_treatment(X, treatment.astype(np.float32))
        weight = _multi_ipw_weights(self, X, treatment)
        if weight is not None:
            fit_params = _with_sample_weight(fit_params, weight)
        if eval_set is not None:
            x_val, t_val, y_val = eval_set
            fit_params = dict(fit_params)
            fit_params.setdefault('eval_set', (_stack_treatment(x_val, t_val.astype(np.float32)), y_val))
        self._model = deepcopy(self.model)
        self._model.fit(x_aug, y, **fit_params)

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        n = len(X)
        x_aug = _stack_treatment(X, np.full(n, float(arm), dtype=np.float32))
        return self._predict_outcome(self._model, x_aug)
