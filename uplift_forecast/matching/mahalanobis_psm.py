__all__ = ['MahalanobisPSCaliperMatcher']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..common._uplift_model import _to_array, _to_numpy_1d
from ._base_matcher import BaseMatcher
from .mahalanobis import _fit_whitener, _whiten


class MahalanobisPSCaliperMatcher(BaseMatcher):
    """Mahalanobis matching within a propensity-score caliper.

    A two-stage matcher: a caller-supplied classifier estimates propensity
    scores, and for each treated unit only control units whose propensity score
    lies within ``caliper`` are eligible. Among those eligible controls the
    nearest by Mahalanobis distance are matched. Treated units with no eligible
    control are dropped. This avoids the pairs of very different treatment
    probability that pure Mahalanobis matching can produce.

    Args:
        model: Classifier with sklearn-style ``fit`` / ``predict_proba`` (CatBoost,
            lgbm, sklearn). Trained to predict treatment.
        caliper (float): Maximum absolute propensity-score difference for a control
            to be eligible for a treated unit.
        n_neighbors (int): Number of control units matched to each treated unit.
        replace (bool): If True a control may match several treated units; if False
            matching is greedy and each control is used at most once.
        standardize (bool): Standardise covariates before computing the Mahalanobis
            whitener (numerically steadier for features on very different scales).
        return_unmatched (bool): If True, ``transform`` returns
            ``(matched_df, unmatched_treated_indices)`` instead of just the frame.
        reg (float): Ridge added to the covariance diagonal before factorisation.
        alias (str): Optional display name.
    """

    def __init__(
        self,
        model: Any,
        caliper: float,
        n_neighbors: int = 1,
        replace: bool = True,
        standardize: bool = True,
        return_unmatched: bool = False,
        reg: float = 1e-6,
        alias: str | None = None,
    ):
        super(MahalanobisPSCaliperMatcher, self).__init__(
            n_neighbors=n_neighbors, caliper=caliper, replace=replace, alias=alias,
        )
        if not hasattr(model, 'predict_proba'):
            raise TypeError('model must expose predict_proba for propensity-score matching.')
        self.model = model
        self.standardize = standardize
        self.return_unmatched = return_unmatched
        self.reg = reg
        self._fitted_model = None
        self._mean: np.ndarray | None = None
        self._whitener: np.ndarray | None = None
        self._std_shift: np.ndarray | None = None
        self._std_scale: np.ndarray | None = None
        self.unmatched_treated_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if not self.standardize:
            return x
        return (x - self._std_shift) / self._std_scale

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        self._fitted_model = deepcopy(self.model)
        self._fitted_model.fit(X, treatment)
        if self.standardize:
            x = np.asarray(X, dtype=float)
            self._std_shift = x.mean(axis=0, keepdims=True)
            self._std_scale = x.std(axis=0, keepdims=True) + 1e-12
        self._mean, self._whitener = _fit_whitener(self._standardize(X), self.reg)

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return _whiten(self._standardize(X), self._mean, self._whitener)

    def _propensity(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(self._fitted_model.predict_proba(X))[:, 1]

    def transform(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        if not self._fitted:
            raise RuntimeError(f'{type(self).__name__} has not been fitted yet. Call .fit() first.')
        X_arr = _to_array(X)
        treatment_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = None if y is None else _to_numpy_1d(y)

        propensity = self._propensity(X_arr)
        z = self._embed(X_arr)
        treated_idx = np.flatnonzero(treatment_arr == 1)
        control_idx = np.flatnonzero(treatment_arr == 0)
        if treated_idx.size == 0 or control_idx.size == 0:
            raise ValueError('Matching needs both treated and control units in the data.')

        p_control = propensity[control_idx]
        z_control = z[control_idx]
        used = np.zeros(control_idx.size, dtype=bool)

        treated_keep: list[int] = []
        unmatched: list[int] = []
        control_counts: dict[int, int] = {}
        for global_t in treated_idx:
            eligible = np.flatnonzero(np.abs(propensity[global_t] - p_control) <= self.caliper)
            if not self.replace:
                eligible = eligible[~used[eligible]]
            if eligible.size == 0:
                unmatched.append(int(global_t))
                continue
            order = np.argsort(np.linalg.norm(z_control[eligible] - z[global_t], axis=1))
            chosen = eligible[order[: self.n_neighbors]]
            if not self.replace:
                used[chosen] = True
            treated_keep.append(int(global_t))
            for pos in chosen:
                control_counts[int(pos)] = control_counts.get(int(pos), 0) + 1

        self.unmatched_treated_ = np.asarray(unmatched, dtype=int)
        if not treated_keep:
            raise ValueError('No treated unit found a control within the caliper; relax `caliper`.')

        control_positions = np.fromiter(control_counts, dtype=int)
        control_global = control_idx[control_positions]
        control_weight = np.array([control_counts[p] for p in control_positions], dtype=float) / self.n_neighbors

        rows = np.concatenate([np.asarray(treated_keep, dtype=int), control_global])
        weight = np.concatenate([np.ones(len(treated_keep)), control_weight])
        matched = self._build_frame(X_arr, treatment_arr, y_arr, rows, weight)
        if self.return_unmatched:
            return matched, self.unmatched_treated_
        return matched
