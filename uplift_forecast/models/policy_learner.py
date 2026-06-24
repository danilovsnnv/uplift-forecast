__all__ = ['PolicyLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..common._base_meta import BaseMetaUpliftModel, _aligned_multiclass_proba, _top_k_mask
from ..common._uplift_model import _to_array, _to_numpy_1d
from ..metrics import optimal_treatment_assignment


class PolicyLearner(BaseMetaUpliftModel):
    """Plug-in policy learner (Athey & Wager, 2021; Kitagawa & Tetenov, 2018), binary or multi-arm.

    Wraps any ``UpliftModel`` CATE estimator and turns its effect estimate into a
    treatment-assignment rule. It fits the wrapped estimator, derives policy labels
    from its CATE (treat where tau(x) > threshold for binary; the best positive-uplift
    arm for multi-arm) and trains a classifier on those labels, weighting by the effect
    magnitude so confident decisions dominate.

    ``predict`` returns the CATE *score* (``[n]`` binary or ``[n, K-1]`` multi-arm, so
    the model stays rankable by AUUC/Qini); ``assign`` returns the chosen action
    (``0/1`` binary, or the arm ``0..K-1`` for multi-arm, optionally under a budget or
    top-k constraint) and ``policy_value`` gives an inverse-propensity off-policy value.

    Args:
        cate_estimator: An ``UpliftModel`` (e.g. TLearner) whose predict gives tau(x).
            A multi-arm estimator (predicting ``[n, K-1]``) yields a multi-arm policy.
        policy_model: Optional sklearn classifier for the policy. Defaults to
            LogisticRegression(max_iter=1000) (multinomial for multi-arm labels).
        threshold (float): Treat where tau(x) > threshold when forming binary policy
            labels and as the default binary decision rule (ignored for multi-arm).
        propensity_model: Optional classifier with predict_proba used only by
            ``policy_value``. If None, the global per-arm rate(s) are used.
        propensity_clip (float): Clip each propensity into [propensity_clip, 1 - propensity_clip].
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
        super().__init__(alias=alias)
        self.cate_estimator = cate_estimator
        self.policy_model = policy_model
        self.threshold = threshold
        self.propensity_model = propensity_model
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._cate_fitted = None
        self._policy_fitted = None
        self._propensity_fitted = None
        self._arms: list[int] = []
        self._global_rates: dict[int, float] = {}

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        t = treatment.astype(int)
        self._arms = sorted(set(t.tolist()))
        self._cate_fitted = deepcopy(self.cate_estimator)
        self._cate_fitted.fit(X, treatment, y, **fit_params)

        tau = np.asarray(self._cate_fitted.predict(X))
        if tau.ndim == 1:
            labels = (tau > self.threshold).astype(int)
            weight = np.abs(tau)
        else:
            labels = optimal_treatment_assignment(tau)
            weight = np.clip(tau.max(axis=1), 0.0, None)
        if len(np.unique(labels)) >= 2:
            self._policy_fitted = (
                deepcopy(self.policy_model) if self.policy_model is not None else LogisticRegression(max_iter=1000)
            )
            self._policy_fitted.fit(_to_array(X), labels, sample_weight=weight if np.any(weight > 0) else None)
        else:
            # Degenerate single-class labels: fall back to the score/argmax rule.
            self._policy_fitted = None

        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            self._global_rates = {arm: float(np.clip((t == arm).mean(), lo, hi)) for arm in self._arms}
            self._propensity_fitted = None
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(_to_array(X), t)

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError('PolicyLearner has not been fitted yet. Call .fit() first.')
        uplift = np.asarray(self._cate_fitted.predict(X))
        if not return_components:
            return uplift
        return uplift, np.zeros(len(uplift), dtype=np.float64), uplift

    def assign(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        threshold: float | None = None,
        budget: float | None = None,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Treatment decisions; budget (fraction) or top_k override the default rule.

        Returns ``0/1`` for a binary policy or the chosen arm ``0..K-1`` for multi-arm.
        """
        if budget is not None and top_k is not None:
            raise ValueError('Pass at most one of budget or top_k.')
        scores = np.asarray(self._cate_fitted.predict(X))
        if scores.ndim == 1:
            return self._assign_binary(X, scores, threshold, budget, top_k)
        if budget is None and top_k is None:
            if self._policy_fitted is not None and threshold is None:
                return np.asarray(self._policy_fitted.predict(_to_array(X))).astype(int)
            return optimal_treatment_assignment(scores)
        targeted = self._budget_mask(scores.max(axis=1), budget, top_k)
        return np.where(targeted == 1, optimal_treatment_assignment(scores), 0)

    def _assign_binary(
        self,
        X: np.ndarray | pd.DataFrame,
        scores: np.ndarray,
        threshold: float | None,
        budget: float | None,
        top_k: int | None,
    ) -> np.ndarray:
        if budget is not None or top_k is not None:
            return self._budget_mask(scores, budget, top_k)
        if self._policy_fitted is not None and threshold is None:
            return np.asarray(self._policy_fitted.predict(_to_array(X))).astype(int)
        thr = self.threshold if threshold is None else threshold
        return (scores > thr).astype(int)

    def _budget_mask(self, score: np.ndarray, budget: float | None, top_k: int | None) -> np.ndarray:
        n = len(score)
        if budget is not None:
            if not 0.0 < budget <= 1.0:
                raise ValueError(f'budget must be in (0, 1]; got {budget}.')
            return _top_k_mask(score, round(budget * n))
        if not 0 <= top_k <= n:
            raise ValueError(f'top_k must be in [0, {n}]; got {top_k}.')
        return _top_k_mask(score, top_k)

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
        gps = self._gps(X)
        col = {arm: j for j, arm in enumerate(self._arms)}
        p_pi = gps[np.arange(len(t)), np.array([col[int(a)] for a in pi])]
        return float(np.mean((t == pi).astype(np.float64) * y / p_pi))

    def _gps(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generalized propensity ``[n, len(arms)]`` (columns by arm order)."""
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self._propensity_fitted is None:
            rates = np.array([self._global_rates[arm] for arm in self._arms])
            return np.tile(rates, (len(_to_array(X)), 1))
        return np.clip(_aligned_multiclass_proba(self._propensity_fitted, _to_array(X), self._arms), lo, hi)
