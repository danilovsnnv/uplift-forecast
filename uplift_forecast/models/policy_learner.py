__all__ = ['PolicyLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..common._base_meta import BaseMetaUpliftModel, _top_k_mask
from ..common._uplift_model import _to_array, _to_numpy_1d


class PolicyLearner(BaseMetaUpliftModel):
    """Plug-in policy learner (Athey & Wager, 2021; Kitagawa & Tetenov, 2018).

    Wraps any ``UpliftModel`` CATE estimator and turns its effect estimate into a
    treatment-assignment rule. It fits the wrapped estimator, thresholds its CATE to
    obtain policy labels (treat where tau(x) > threshold) and trains a classifier on
    those labels, weighting by |tau| so confident decisions dominate.

    ``predict`` returns the CATE *score* (so the model stays rankable by AUUC/Qini);
    ``assign`` returns the 0/1 treatment decisions (optionally under a budget or
    top-k constraint) and ``policy_value`` gives an inverse-propensity off-policy
    value estimate.

    Args:
        cate_estimator: An ``UpliftModel`` (e.g. TLearner) whose predict gives tau(x).
        policy_model: Optional sklearn classifier for the policy. Defaults to
            LogisticRegression(max_iter=1000).
        threshold (float): Treat where tau(x) > threshold when forming policy labels
            and as the default decision rule.
        propensity_model: Optional classifier with predict_proba used only by
            ``policy_value``. If None, the global treatment rate is used.
        propensity_clip (float): Clip e(x) into [propensity_clip, 1 - propensity_clip].
        random_state (int): Reserved for reproducibility of the policy model.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        cate_estimator: Any,
        policy_model: Any | None = None,
        threshold: float = 0.0,
        propensity_model: Any | None = None,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super(PolicyLearner, self).__init__(alias=alias)
        self.cate_estimator = cate_estimator
        self.policy_model = policy_model
        self.threshold = threshold
        self.propensity_model = propensity_model
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._cate_fitted = None
        self._policy_fitted = None
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
        self._cate_fitted = deepcopy(self.cate_estimator)
        self._cate_fitted.fit(X, treatment, y, **fit_params)

        tau = np.asarray(self._cate_fitted.predict(X)).reshape(-1)
        labels = (tau > self.threshold).astype(int)
        if len(np.unique(labels)) >= 2:
            self._policy_fitted = (
                deepcopy(self.policy_model) if self.policy_model is not None else LogisticRegression(max_iter=1000)
            )
            weight = np.abs(tau)
            self._policy_fitted.fit(_to_array(X), labels, sample_weight=weight if np.any(weight > 0) else None)
        else:
            # Degenerate single-class labels: fall back to thresholding the score.
            self._policy_fitted = None

        if self.propensity_model is None:
            self._global_rate = float(t.mean())
            self._propensity_fitted = None
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(_to_array(X), t)

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(self._cate_fitted.predict(X)).reshape(-1)
        return np.zeros(len(tau), dtype=np.float64), tau

    def assign(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        threshold: float | None = None,
        budget: float | None = None,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Binary treatment decisions; budget (fraction) or top_k override the rule."""
        if budget is not None and top_k is not None:
            raise ValueError('Pass at most one of budget or top_k.')
        scores = np.asarray(self._cate_fitted.predict(X)).reshape(-1)
        n = len(scores)
        if budget is not None:
            if not 0.0 < budget <= 1.0:
                raise ValueError(f'budget must be in (0, 1]; got {budget}.')
            return _top_k_mask(scores, int(round(budget * n)))
        if top_k is not None:
            if not 0 <= top_k <= n:
                raise ValueError(f'top_k must be in [0, {n}]; got {top_k}.')
            return _top_k_mask(scores, top_k)
        if self._policy_fitted is not None and threshold is None:
            return np.asarray(self._policy_fitted.predict(_to_array(X))).astype(int)
        thr = self.threshold if threshold is None else threshold
        return (scores > thr).astype(int)

    def policy_value(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        *,
        policy: np.ndarray | None = None,
    ) -> float:
        """Inverse-propensity off-policy value V(pi) = E[Y * 1{T=pi(X)} / P(T=pi|X)]."""
        t = _to_numpy_1d(treatment).astype(int)
        y = _to_numpy_1d(y).astype(np.float64)
        pi = self.assign(X) if policy is None else _to_numpy_1d(policy).astype(int)
        e = self._propensity(X, len(t))
        p_obs = np.where(pi == 1, e, 1.0 - e)
        return float(np.mean((t == pi).astype(np.float64) * y / p_obs))

    def _propensity(self, X: np.ndarray | pd.DataFrame, n: int) -> np.ndarray:
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self._propensity_fitted is None:
            return np.full(n, np.clip(self._global_rate, lo, hi))
        return np.clip(np.asarray(self._propensity_fitted.predict_proba(_to_array(X)))[:, 1], lo, hi)
