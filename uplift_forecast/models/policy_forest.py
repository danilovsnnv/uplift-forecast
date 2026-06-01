__all__ = ['PolicyForest']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..common._base_meta import BaseMetaUpliftModel, _resolve_max_features, _top_k_mask
from ..common._uplift_model import _to_array, _to_numpy_1d

# Shallow defaults for the internal outcome nuisances feeding the DR gain score.
_NUISANCE_MAX_DEPTH = 5
_NUISANCE_MIN_SAMPLES_LEAF = 10


class PolicyForest(BaseMetaUpliftModel):
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

    ``predict`` returns the mean across-tree expected gain (a rankable score);
    ``assign`` returns the 0/1 recommendation (optionally under a budget or top-k
    constraint) and ``policy_value`` gives an inverse-propensity off-policy value.

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
        propensity_model: Optional classifier with predict_proba for e(x)=P(T=1|X).
            If None, the global treatment rate is used as a constant.
        n_folds (int): Folds for cross-fitting the nuisances used by the gain score.
        propensity_clip (float): Clip e(x) into [propensity_clip, 1 - propensity_clip].
        random_state (int): Base seed; tree i uses random_state + i.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        max_features: float | str | int | None = 'sqrt',
        bootstrap: bool = True,
        honest: bool = True,
        score_method: str = 'dr',
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super(PolicyForest, self).__init__(alias=alias)
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

        self.policy_trees: list[dict] = []

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
        x = np.asarray(_to_array(X), dtype=np.float64)
        t = treatment.astype(int)
        y = y.astype(np.float64)
        p = x.shape[1]
        if not (t == 0).any() or not (t == 1).any():
            raise ValueError('PolicyForest requires both treated and control samples in the data.')

        gamma = self._gain_score(x, t, y)
        k = _resolve_max_features(self.max_features, p)

        self.policy_trees = []
        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.random_state + i)
            feat = np.sort(rng.choice(p, size=k, replace=False))
            self.policy_trees.append(self._build_tree(x, gamma, feat, rng))

        if self.propensity_model is None:
            self._global_rate = float(t.mean())
            self._propensity_fitted = None
        else:
            self._propensity_fitted = deepcopy(self.propensity_model)
            self._propensity_fitted.fit(x, t)

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

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        score = self._tree_gains(X).mean(axis=0)
        return np.zeros(len(score), dtype=np.float64), score

    def assign(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        budget: float | None = None,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Binary recommendations; budget (fraction) or top_k override the gain rule."""
        if budget is not None and top_k is not None:
            raise ValueError('Pass at most one of budget or top_k.')
        score = self._tree_gains(X).mean(axis=0)
        n = len(score)
        if budget is not None:
            if not 0.0 < budget <= 1.0:
                raise ValueError(f'budget must be in (0, 1]; got {budget}.')
            return _top_k_mask(score, int(round(budget * n)))
        if top_k is not None:
            if not 0 <= top_k <= n:
                raise ValueError(f'top_k must be in [0, {n}]; got {top_k}.')
            return _top_k_mask(score, top_k)
        return (score > 0.0).astype(int)

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
        e = self._propensity_predict(X, len(t))
        p_obs = np.where(pi == 1, e, 1.0 - e)
        return float(np.mean((t == pi).astype(np.float64) * y / p_obs))

    def _tree_gains(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        x = np.asarray(_to_array(X), dtype=np.float64)
        out = np.empty((len(self.policy_trees), len(x)), dtype=np.float64)
        for j, entry in enumerate(self.policy_trees):
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

    def _propensity_predict(self, X: np.ndarray | pd.DataFrame, n: int) -> np.ndarray:
        lo, hi = self.propensity_clip, 1.0 - self.propensity_clip
        if self._propensity_fitted is None:
            return np.full(n, np.clip(self._global_rate, lo, hi))
        x = np.asarray(_to_array(X), dtype=np.float64)
        return np.clip(self._propensity_fitted.predict_proba(x)[:, 1], lo, hi)

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
