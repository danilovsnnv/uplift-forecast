__all__ = ['NearestNeighborMatcher']


from collections.abc import Callable

import numpy as np
import pandas as pd

from ._backends import NeighborBackend, SklearnNeighborBackend
from ._base_matcher import BaseMatcher

_SKLEARN_METRICS = {'euclidean', 'manhattan', 'cosine', 'mahalanobis'}


class NearestNeighborMatcher(BaseMatcher):
    """Flexible nearest-neighbor matching.

    Matches each treated unit to control units under a configurable distance
    metric and neighbor-search backend. Supports 1:1 / 1:k matching, radius
    matching, and matching with or without replacement. The default backend is
    exact (scikit-learn); a custom :class:`NeighborBackend` can be supplied for
    approximate search on large datasets.

    Args:
        metric: ``'euclidean'`` / ``'manhattan'`` / ``'cosine'`` / ``'mahalanobis'``
            or a callable distance ``f(a, b) -> float``. ``'mahalanobis'`` uses the
            pooled inverse covariance learned on ``fit`` (unless ``VI`` is given via
            ``metric_params``).
        n_neighbors (int): Controls matched per treated unit (k-NN mode).
        radius (float): If set, switches to radius matching (every control within
            ``radius`` is matched); mutually exclusive with ``caliper``.
        replace (bool): Whether a control may match several treated units.
        caliper (float): Maximum distance for a valid k-NN match. None disables it.
        backend: Optional :class:`NeighborBackend` instance. Defaults to an exact
            sklearn backend configured from ``metric`` / ``metric_params``.
        metric_params (dict): Extra metric parameters forwarded to the backend.
        alias (str): Optional display name.
    """

    def __init__(
        self,
        metric: str | Callable = 'euclidean',
        n_neighbors: int = 1,
        radius: float | None = None,
        replace: bool = True,
        caliper: float | None = None,
        backend: NeighborBackend | None = None,
        metric_params: dict | None = None,
        alias: str | None = None,
    ):
        super(NearestNeighborMatcher, self).__init__(
            n_neighbors=n_neighbors, caliper=caliper, replace=replace, alias=alias,
        )
        if radius is not None:
            if radius <= 0:
                raise ValueError(f'radius must be positive or None, got {radius!r}.')
            if caliper is not None:
                raise ValueError('Set either radius (radius matching) or caliper (k-NN matching), not both.')
        if isinstance(metric, str) and metric not in _SKLEARN_METRICS:
            raise ValueError(f"Unknown metric {metric!r}. Use one of {sorted(_SKLEARN_METRICS)} or a callable.")
        self.metric = metric
        self.radius = radius
        self.backend = backend
        self.metric_params = metric_params
        self._vi: np.ndarray | None = None

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        if self.metric == 'mahalanobis' and (self.metric_params is None or 'VI' not in self.metric_params):
            x = np.asarray(X, dtype=float)
            cov = np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]) + 1e-6 * np.eye(x.shape[1])
            self._vi = np.linalg.inv(cov)

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def _resolve_backend(self) -> NeighborBackend:
        if self.backend is not None:
            return self.backend
        metric_params = dict(self.metric_params) if self.metric_params else None
        if self.metric == 'mahalanobis' and self._vi is not None:
            metric_params = {**(metric_params or {}), 'VI': self._vi}
        return SklearnNeighborBackend(metric=self.metric, metric_params=metric_params)

    def _nearest_neighbors(self, treated_emb: np.ndarray, control_emb: np.ndarray) -> list[np.ndarray]:
        backend = self._resolve_backend().fit(control_emb)
        n_control = control_emb.shape[0]
        if self.replace:
            dist, nbr = backend.kneighbors(treated_emb, self.n_neighbors)
            if self.caliper is None:
                return [row for row in nbr]
            return [nbr[i][dist[i] <= self.caliper] for i in range(treated_emb.shape[0])]

        # Without replacement: greedy assignment over distance-sorted neighbor lists.
        # When a caliper is set, use radius search to pre-filter candidates instead of
        # fetching all n_control neighbours and discarding most of them in the loop.
        if self.caliper is not None:
            r_dist, r_nbr = backend.radius_neighbors(treated_emb, self.caliper)
            sorted_pairs = []
            for d_row, n_row in zip(r_dist, r_nbr):
                d_arr, n_arr = np.asarray(d_row), np.asarray(n_row, dtype=int)
                order = np.argsort(d_arr)
                sorted_pairs.append(n_arr[order])
        else:
            dist, nbr = backend.kneighbors(treated_emb, n_control)
            sorted_pairs = [nbr[i] for i in range(treated_emb.shape[0])]

        used = np.zeros(n_control, dtype=bool)
        matches: list[np.ndarray] = []
        for n_sorted in sorted_pairs:
            picks: list[int] = []
            for c in n_sorted:
                c = int(c)
                if used[c]:
                    continue
                picks.append(c)
                used[c] = True
                if len(picks) == self.n_neighbors:
                    break
            matches.append(np.asarray(picks, dtype=int))
        return matches

    def _match(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray | None,
        embedding: np.ndarray,
    ) -> pd.DataFrame:
        if self.radius is None:
            return super()._match(X, treatment, y, embedding)

        treated_idx = np.flatnonzero(treatment == 1)
        control_idx = np.flatnonzero(treatment == 0)
        if treated_idx.size == 0 or control_idx.size == 0:
            raise ValueError('Matching needs both treated and control units in the data.')

        backend = self._resolve_backend().fit(embedding[control_idx])
        _, idxs = backend.radius_neighbors(embedding[treated_idx], self.radius)

        treated_keep: list[int] = []
        control_weight: dict[int, float] = {}
        for global_t, neigh in zip(treated_idx, idxs):
            if len(neigh) == 0:
                continue
            treated_keep.append(int(global_t))
            share = 1.0 / len(neigh)  # each treated distributes total control weight 1
            for pos in neigh:
                control_weight[int(pos)] = control_weight.get(int(pos), 0.0) + share

        if not treated_keep:
            raise ValueError('No treated unit found a control within `radius`; increase it.')

        control_positions = np.fromiter(control_weight, dtype=int)
        control_global = control_idx[control_positions]
        cw = np.array([control_weight[p] for p in control_positions], dtype=float)

        rows = np.concatenate([np.asarray(treated_keep, dtype=int), control_global])
        weight = np.concatenate([np.ones(len(treated_keep)), cw])
        return self._build_frame(X, treatment, y, rows, weight)
