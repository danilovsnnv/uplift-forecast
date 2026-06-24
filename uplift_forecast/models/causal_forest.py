__all__ = ['CausalForest']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from ..common._base_meta import _BaseMultiArmLearner, _resolve_max_features
from ..common._uplift_model import _to_array

# Shallow defaults for the internal outcome nuisances feeding the DR pseudo-outcome;
# their role is only to guide split selection, so they need not be strong learners.
_NUISANCE_MAX_DEPTH = 5
_NUISANCE_MIN_SAMPLES_LEAF = 10


class CausalForest(_BaseMultiArmLearner):
    """Causal forest for heterogeneous treatment effects (Wager & Athey, 2018; GRF, 2019).

    A lighter, sklearn-based approximation of a causal forest: a bag of
    ``DecisionTreeRegressor`` trees grown on a doubly-robust (or IPW) pseudo-outcome,
    with bootstrap resampling, random feature subsampling and optional *honest*
    leaf estimation. With honesty (the default) the tree structure is grown on one
    half of each tree's sample and the leaf treatment effects are recomputed on the
    held-out half as mean(Y_treated) - mean(Y_control), enforcing a minimum treated
    and control count per leaf; the pseudo-outcome only guides split selection.

    This is NOT the exact GRF gradient/label splitting criterion -- splits are chosen
    by ordinary squared-error reduction on the pseudo-outcome -- but it preserves the
    honest, ensemble, leaf-contrast structure of a causal forest.

    With ``K`` arms an independent forest is grown per treated arm vs control on its
    ``{0, k}`` subset, and ``predict`` returns ``[n, K-1]`` (collapsing to a flat
    ``[n]`` in the binary case). ``predict`` reports y0 = 0 and y1 = tau(x);
    ``predict_variance`` / ``predict_interval`` return the across-tree spread per arm
    (matching the ``predict`` shape).

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree (None for unlimited).
        min_samples_leaf (int): Minimum samples per leaf when growing each tree.
        max_features: Features sampled per tree: 'sqrt'/'log2', a fraction in (0, 1],
            an int count, or None for all features.
        bootstrap (bool): If True, grow each tree on a bootstrap resample.
        honest (bool): If True, use separate structure/estimation halves and recompute
            leaf effects as mean(Y_treated) - mean(Y_control) on the estimation half.
        min_treated_leaf (int): Minimum treated rows for an honest leaf contrast;
            leaves below this fall back to the leaf pseudo-outcome mean.
        min_control_leaf (int): Minimum control rows for an honest leaf contrast.
        pseudo_outcome (str): 'dr' (doubly-robust AIPW) or 'ipw' (inverse propensity).
        propensity_model: Optional classifier with predict_proba for e(x)=P(T=1|X).
            If None, the global treatment rate is used as a constant.
        n_folds (int): Folds for cross-fitting the nuisances used by the pseudo-outcome.
        propensity_clip (float): Clip e(x) into [propensity_clip, 1 - propensity_clip].
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
        min_treated_leaf: int = 1,
        min_control_leaf: int = 1,
        pseudo_outcome: str = 'dr',
        propensity_model: Any | None = None,
        n_folds: int = 5,
        propensity_clip: float = 1e-3,
        random_state: int = 0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if pseudo_outcome not in ('dr', 'ipw'):
            raise ValueError(f"pseudo_outcome must be 'dr' or 'ipw'; got {pseudo_outcome!r}.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.honest = honest
        self.min_treated_leaf = min_treated_leaf
        self.min_control_leaf = min_control_leaf
        self.pseudo_outcome = pseudo_outcome
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.random_state = random_state

        self._arm_forests: dict[int, list[dict]] = {}

    @property
    def causal_trees(self) -> list[dict]:
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

    def _fit_forest(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, n_features: int) -> list[dict]:
        psi = self._pseudo_outcome(x, t, y)
        k = _resolve_max_features(self.max_features, n_features)
        forest = []
        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.random_state + i)
            feat = np.sort(rng.choice(n_features, size=k, replace=False))
            forest.append(self._build_tree(x, t, y, psi, feat, rng))
        return forest

    def _build_tree(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        psi: np.ndarray,
        feat: np.ndarray,
        rng: np.random.Generator,
    ) -> dict:
        # Honest trees use disjoint, de-duplicated structure/estimation halves so the
        # leaf contrast stays unbiased; bootstrap (when enabled) resamples only the
        # structure half used to choose splits, never the estimation half.
        n = len(x)
        if self.honest and n >= 4:
            perm = rng.permutation(n)
            half = n // 2
            struct, est = perm[:half], perm[half:]
            if self.bootstrap:
                struct = rng.choice(struct, size=len(struct), replace=True)
        else:
            struct = est = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        tree.fit(x[np.ix_(struct, feat)], psi[struct])

        leaves = tree.apply(x[np.ix_(est, feat)])
        leaf_tau: dict[int, float] = {}
        for leaf in np.unique(leaves):
            rows = est[leaves == leaf]
            treated = rows[t[rows] == 1]
            control = rows[t[rows] == 0]
            if self.honest and len(treated) >= self.min_treated_leaf and len(control) >= self.min_control_leaf:
                leaf_tau[int(leaf)] = float(y[treated].mean() - y[control].mean())
            else:
                leaf_tau[int(leaf)] = float(psi[rows].mean())
        fallback = float(psi[est].mean())
        return {'tree': tree, 'leaf_tau': leaf_tau, 'features': feat, 'fallback': fallback}

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        if arm == 0:
            return np.zeros(len(X), dtype=np.float64)
        return self._tree_predictions(X, self._arm_forests[arm]).mean(axis=0)

    def predict_variance(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Across-tree variance of the per-tree CATE estimates (uncertainty proxy).

        Shape matches ``predict``: ``[n]`` for binary treatment, ``[n, K-1]`` per arm.
        """
        if not self._fitted:
            raise RuntimeError('CausalForest has not been fitted yet. Call .fit() first.')
        out = np.stack([self._tree_predictions(X, self._arm_forests[arm]).var(axis=0) for arm in self.arms_[1:]], axis=1)
        return out[:, 0] if out.shape[1] == 1 else out

    def predict_interval(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate ``(1 - alpha)`` confidence interval for the CATE.

        Builds a normal interval ``tau(x) +/- z * sd(x)`` from the across-tree
        standard deviation of the (honest) per-tree estimates. This is an
        approximate interval from the ensemble spread, not the exact
        infinitesimal-jackknife interval of GRF. Bounds match ``predict``'s shape
        (``[n]`` binary, ``[n, K-1]`` per arm).

        Args:
            X: Features to score.
            alpha: Significance level (``0.05`` -> 95% interval).

        Returns:
            ``(lower, upper)`` arrays of CATE bounds.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'alpha must be in (0, 1); got {alpha}.')
        from statistics import NormalDist  # noqa: PLC0415  (lazy: only used by predict_interval)

        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        lowers, uppers = [], []
        for arm in self.arms_[1:]:
            preds = self._tree_predictions(X, self._arm_forests[arm])
            tau, sd = preds.mean(axis=0), preds.std(axis=0)
            lowers.append(tau - z * sd)
            uppers.append(tau + z * sd)
        lower, upper = np.stack(lowers, axis=1), np.stack(uppers, axis=1)
        if lower.shape[1] == 1:
            return lower[:, 0], upper[:, 0]
        return lower, upper

    def _tree_predictions(self, X: np.ndarray | pd.DataFrame, forest: list[dict]) -> np.ndarray:
        x = np.asarray(_to_array(X), dtype=np.float64)
        preds = np.empty((len(forest), len(x)), dtype=np.float64)
        for j, entry in enumerate(forest):
            leaves = entry['tree'].apply(x[:, entry['features']])
            leaf_tau, fallback = entry['leaf_tau'], entry['fallback']
            preds[j] = np.array([leaf_tau.get(int(leaf), fallback) for leaf in leaves], dtype=np.float64)
        return preds

    def _pseudo_outcome(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        e = self._propensity(x, t)
        if self.pseudo_outcome == 'ipw':
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
        if self.n_folds <= 1:
            est = DecisionTreeRegressor(
                max_depth=_NUISANCE_MAX_DEPTH,
                min_samples_leaf=max(_NUISANCE_MIN_SAMPLES_LEAF, self.min_samples_leaf),
                random_state=self.random_state,
            )
            est.fit(x[mask], y[mask])
            return est.predict(x)
        out = np.empty(n, dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(np.arange(n)):
            arm_idx = train_idx[mask[train_idx]]
            est = DecisionTreeRegressor(
                max_depth=_NUISANCE_MAX_DEPTH,
                min_samples_leaf=max(_NUISANCE_MIN_SAMPLES_LEAF, self.min_samples_leaf),
                random_state=self.random_state,
            )
            est.fit(x[arm_idx], y[arm_idx])
            out[test_idx] = est.predict(x[test_idx])
        return out
