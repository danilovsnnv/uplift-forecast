__all__ = ['TLearner']


from copy import deepcopy
from typing import Any

import numpy as np

from ..common._base_meta import _BaseMultiArmLearner, _multi_ipw_weights, _with_sample_weight
from ..common._uplift_model import _row_subset


class TLearner(_BaseMultiArmLearner):
    """Multi-model meta-learner (T-learner), binary or multi-arm.

    Fits an independent outcome estimator on each arm's rows (including the control arm 0)
    and reports, for every treated arm ``k``, ``mu_k(x) - mu_0(x)``. With a binary treatment
    (arms ``{0, 1}``) ``predict`` returns a flat ``[n]`` uplift; with ``K`` arms it returns
    ``[n, K-1]`` (one column per treated arm vs control).

    When ``propensity_model`` is given, each arm's regression is fit with inverse-propensity
    sample weights so the per-arm fit targets the full population: ``1 / e(x)`` / ``1 / (1 - e(x))``
    in the binary case, ``1 / P(T=a_i|x_i)`` (IPTW per McCaffrey et al., 2013) in the multi-arm
    case. The propensities are cross-fitted first; the base estimators must accept a
    ``sample_weight`` fit argument.

    Args:
        model: Base estimator for the control arm.
        model_treated: Optional separate estimator template for the treated arms.
            Defaults to a deepcopy of ``model`` (one independent copy per treated arm).
        propensity_model: Optional classifier with ``predict_proba`` for the per-arm propensities.
            If None, the arms are fit unweighted (plain T-learner).
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
        model_treated: Any | None = None,
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
        self.model_treated = model_treated
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state
        self._models: dict[int, Any] = {}
        self._propensity_model: Any | None = None

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        self._models = {}
        weight = _multi_ipw_weights(self, X, treatment)
        for arm in self.arms_:
            mask = treatment == arm
            kwargs = dict(fit_params) if weight is None else _with_sample_weight(fit_params, weight[mask])
            if eval_set is not None:
                x_val, t_val, y_val = eval_set
                vmask = t_val == arm
                kwargs.setdefault('eval_set', (_row_subset(x_val, vmask), y_val[vmask]))
            template = self.model if arm == 0 or self.model_treated is None else self.model_treated
            est = deepcopy(template)
            est.fit(_row_subset(X, mask), y[mask], **kwargs)
            self._models[arm] = est

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        return self._predict_outcome(self._models[arm], X)
