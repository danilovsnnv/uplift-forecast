__all__ = ['PolicyForest']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..common._base_meta import (
    _aligned_multiclass_proba,
    _BaseMultiArmLearner,
    _resolve_max_features,
    _top_k_mask,
)
from ..common._uplift_model import _to_array, _to_numpy_1d
from ..metrics import optimal_treatment_assignment

# Shallow defaults for the internal outcome nuisances feeding the DR gain score.
_NUISANCE_MAX_DEPTH = 5
_NUISANCE_MIN_SAMPLES_LEAF = 10


class PolicyForest(_BaseMultiArmLearner):
    """Policy forest for treatment-assignment rules (Athey & Wager, 2021; policytree).

    A lighter, sklearn-based approximation of a policy forest: a bag of
    ``DecisionTreeClassifier`` trees that predict the optimal action 1{Gamma > 0},
    weighted by the magnitude |Gamma| of a doubly-robust (or IPW) individual gain
    score, with bootstrap, random feature subsampling and optional honest leaf
    estimation. With honesty (the default) each leaf's expected gain and action are
    recomputed on a held-out estimation half; weighted classification then reduces
    welfare maximization to a standard learning problem.

    This is NOT exact policy-value tree optimization, but it preserves the
    DR-score / honest / ensemble structure of a policy forest.

    With ``K`` arms an independent gain forest is grown per treated arm vs control,
    and ``predict`` returns the per-arm expected gain ``[n, K-1]`` (a rankable score),
    collapsing to a flat ``[n]`` in the binary case. ``assign`` returns the chosen arm
    (``0`` = control, ``k`` = treated arm ``k``; the best positive-gain arm, optionally
    under a budget or top-k constraint) and ``policy_value`` gives an inverse-propensity
    off-policy value.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree (None for unlimited).
        min_samples_leaf (int): Minimum samples per leaf when growing each tree.
        max_features: Features sampled per tree: 'sqrt'/'log2', a fraction in (0, 1],
            an int count, or None for all features.
        bootstrap (bool): If True, grow each tree on a bootstrap resample.
        honest (bool): If True, use separate structure/estimation halves and recompute
            each leaf's expected gain and action on the estimation half.
        score_method (str): 'dr' (doubly-robust AIPW) or 'ipw' for the gain score.
        propensity_model: Optional classifier with predict_proba for the (per-arm)
            propensities. If None, the global per-arm rates are used.
        n_folds (int): Folds for cross-fitting the nuisances used by the gain score.
        propensity_clip (float): Clip each propensity into [propensity_clip, 1 - propensity_clip].
        random_state (int): Base seed; tree i uses random_state + i.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        max_features: float | str | None = 'sqrt',
        bootstrap: bool = True,
        honest: bool = True,
        score_method: str = 'dr',
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if score_method not in ('dr', 'ipw'):
            raise ValueError(f"score_method must be 'dr' or 'ipw'; got {score_method!r}.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.honest = honest
        self.score_method = score_method
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._arm_forests: dict[int, list[dict]] = {}
        self._propensity_fitted: Any | None = None
        self._global_rates: dict[int, float] = {}

    @property
    def policy_trees(self) -> list[dict]:
        """All per-arm trees flattened into one ensemble (used by feature-importance tooling)."""
        return [tree for forest in self._arm_forests.values() for tree in forest]

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        del eval_set, fit_params
        x = np.asarray(_to_array(X), dtype=np.float64)
        t = treatment.astype(int)
        y = y.astype(np.float64)
        n_features = x.shape[1]
        self._arm_forests = {}
        for arm in self.arms_[1:]:
            mask = (t == 0) | (t == arm)
            self._arm_forests[arm] = self._fit_forest(x[mask], (t[mask] == arm).astype(int), y[mask], n_features)

        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            self._global_rates = {arm: float(np.clip((t == arm).mean(), lo, hi)) for arm in self.arms_}
            self._propensity_fitted = None
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(x, t)

    def _fit_forest(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, n_features: int) -> list[dict]:
        gamma = self._gain_score(x, t, y)
        k = _resolve_max_features(self.max_features, n_features)
        forest = []
        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.random_state + i)
            feat = np.sort(rng.choice(n_features, size=k, replace=False))
            forest.append(self._build_tree(x, gamma, feat, rng))
        return forest

    def _build_tree(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        feat: np.ndarray,
        rng: np.random.Generator,
    ) -> dict:
        # Disjoint, de-duplicated honest halves (see CausalForest); bootstrap resamples
        # only the structure half used to choose splits.
        n = len(x)
        if self.honest and n >= 4:
            perm = rng.permutation(n)
            half = n // 2
            struct, est = perm[:half], perm[half:]
            if self.bootstrap:
                struct = rng.choice(struct, size=len(struct), replace=True)
        else:
            struct = est = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)

        labels = (gamma[struct] > 0).astype(int)
        fallback = float(gamma[est].mean()) if len(est) else 0.0
        if len(np.unique(labels)) < 2:
            return {'tree': None, 'leaf_gain': None, 'features': feat, 'const_gain': fallback}

        weight = np.abs(gamma[struct])
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        tree.fit(x[np.ix_(struct, feat)], labels, sample_weight=weight if np.any(weight > 0) else None)

        leaves = tree.apply(x[np.ix_(est, feat)])
        leaf_gain = {int(leaf): float(gamma[est[leaves == leaf]].mean()) for leaf in np.unique(leaves)}
        return {'tree': tree, 'leaf_gain': leaf_gain, 'features': feat, 'const_gain': None, 'fallback': fallback}

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        if arm == 0:
            return np.zeros(len(X), dtype=np.float64)
        return self._tree_gains(X, self._arm_forests[arm]).mean(axis=0)

    def assign(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        budget: float | None = None,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Treatment recommendations; budget (fraction) or top_k override the gain rule.

        Returns the chosen arm per unit (``0`` = control, ``k`` = treated arm ``k``).
        """
        if budget is not None and top_k is not None:
            raise ValueError('Pass at most one of budget or top_k.')
        score = np.asarray(self.predict(X))
        if score.ndim == 1:
            return self._budget_mask(score, budget, top_k) if (budget is not None or top_k is not None) \
                else (score > 0.0).astype(int)
        best_arm = optimal_treatment_assignment(score)
        if budget is None and top_k is None:
            return best_arm
        targeted = self._budget_mask(score.max(axis=1), budget, top_k)
        return np.where(targeted == 1, best_arm, 0)

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
        col = {arm: j for j, arm in enumerate(self.arms_)}
        p_pi = gps[np.arange(len(t)), np.array([col[int(a)] for a in pi])]
        return float(np.mean((t == pi).astype(np.float64) * y / p_pi))

    def _gps(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generalized propensity ``[n, len(arms_)]`` (columns by arm order)."""
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self._propensity_fitted is None:
            rates = np.array([self._global_rates[arm] for arm in self.arms_])
            return np.tile(rates, (len(_to_array(X)), 1))
        return np.clip(_aligned_multiclass_proba(self._propensity_fitted, _to_array(X), self.arms_), lo, hi)

    def _tree_gains(self, X: np.ndarray | pd.DataFrame, forest: list[dict]) -> np.ndarray:
        x = np.asarray(_to_array(X), dtype=np.float64)
        out = np.empty((len(forest), len(x)), dtype=np.float64)
        for j, entry in enumerate(forest):
            if entry['tree'] is None:
                out[j] = entry['const_gain']
                continue
            leaves = entry['tree'].apply(x[:, entry['features']])
            leaf_gain, fallback = entry['leaf_gain'], entry['fallback']
            out[j] = np.array([leaf_gain.get(int(leaf), fallback) for leaf in leaves], dtype=np.float64)
        return out

    def _gain_score(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        e = self._propensity(x, t)
        if self.score_method == 'ipw':
            return t * y / e - (1 - t) * y / (1 - e)
        mu0 = self._oof_arm(x, y, t == 0)
        mu1 = self._oof_arm(x, y, t == 1)
        return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)

    def _propensity(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self.propensity_model is None:
            return np.full(len(t), np.clip(float(t.mean()), lo, hi))
        n = len(t)
        if self.n_folds <= 1:
            clf = deepcopy(self.propensity_model)
            clf.fit(x, t)
            return np.clip(clf.predict_proba(x)[:, 1], lo, hi)
        out = np.empty(n, dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(np.arange(n)):
            clf = deepcopy(self.propensity_model)
            clf.fit(x[train_idx], t[train_idx])
            out[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
        return np.clip(out, lo, hi)

    def _oof_arm(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n = len(y)
        leaf = max(_NUISANCE_MIN_SAMPLES_LEAF, self.min_samples_leaf)
        if self.n_folds <= 1:
            est = DecisionTreeRegressor(max_depth=_NUISANCE_MAX_DEPTH, min_samples_leaf=leaf, random_state=self.random_state)
            est.fit(x[mask], y[mask])
            return est.predict(x)
        out = np.empty(n, dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(np.arange(n)):
            arm_idx = train_idx[mask[train_idx]]
            est = DecisionTreeRegressor(max_depth=_NUISANCE_MAX_DEPTH, min_samples_leaf=leaf, random_state=self.random_state)
            est.fit(x[arm_idx], y[arm_idx])
            out[test_idx] = est.predict(x[test_idx])
        return out
