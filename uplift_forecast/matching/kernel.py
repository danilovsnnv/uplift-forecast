__all__ = ['KernelMatcher']


import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ._base_matcher import BaseMatcher
from ._kernels import KERNELS


class KernelMatcher(BaseMatcher):
    """Kernel matching — a soft version of k-NN matching.

    Each treated unit is matched to many control units, weighting closer controls
    more heavily via a kernel of the distance. Useful when hard matching is too
    noisy and smoother counterfactual estimates are wanted.

    Args:
        kernel (str): One of ``'gaussian'``, ``'epanechnikov'``, ``'triangular'``,
            ``'uniform'``.
        bandwidth: Positive float, or ``'auto'`` / ``None`` to use the median
            candidate distance (a robust scale).
        candidate_mode (str): How controls are pre-selected per treated unit —
            ``'all'`` (every control), ``'knn'`` (the ``n_neighbors`` nearest), or
            ``'radius'`` (controls within ``radius``).
        n_neighbors (int): Candidate count for ``candidate_mode='knn'``.
        radius (float): Candidate radius for ``candidate_mode='radius'``.
        metric (str): Distance metric for candidate selection and weighting.
        normalize (bool): Normalise each treated unit's kernel weights to sum to 1.
        return_weight_matrix (bool): If True, ``transform`` also returns a sparse
            weight matrix as a list of ``(treated_index, control_index, weight)``
            tuples (indices are positions in the input rows).
        alias (str): Optional display name.
    """

    def __init__(
        self,
        kernel: str = 'gaussian',
        bandwidth: float | str | None = None,
        candidate_mode: str = 'all',
        n_neighbors: int = 10,
        radius: float | None = None,
        metric: str = 'euclidean',
        normalize: bool = True,
        return_weight_matrix: bool = False,
        alias: str | None = None,
    ):
        super(KernelMatcher, self).__init__(n_neighbors=n_neighbors, alias=alias)
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}. Valid: {sorted(KERNELS)}.")
        if candidate_mode not in ('all', 'knn', 'radius'):
            raise ValueError(f"candidate_mode must be 'all', 'knn' or 'radius', got {candidate_mode!r}.")
        if candidate_mode == 'radius' and radius is None:
            raise ValueError("candidate_mode='radius' requires `radius`.")
        if isinstance(bandwidth, (int, float)) and bandwidth <= 0:
            raise ValueError(f'bandwidth must be positive, got {bandwidth!r}.')
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.candidate_mode = candidate_mode
        self.radius = radius
        self.metric = metric
        self.normalize = normalize
        self.return_weight_matrix = return_weight_matrix

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        pass

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def _candidates(self, treated_emb: np.ndarray, control_emb: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per treated unit, the candidate (control_positions, distances)."""
        n_control = control_emb.shape[0]
        if self.candidate_mode == 'all':
            dists = pairwise_distances(treated_emb, control_emb, metric=self.metric)
            full = np.arange(n_control)
            return [(full, dists[i]) for i in range(treated_emb.shape[0])]
        if self.candidate_mode == 'knn':
            self._require_enough_controls(n_control)
            nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(control_emb)
            dist, nbr = nn.kneighbors(treated_emb)
            return [(nbr[i], dist[i]) for i in range(treated_emb.shape[0])]
        nn = NearestNeighbors(metric=self.metric).fit(control_emb)
        dist, nbr = nn.radius_neighbors(treated_emb, radius=self.radius)
        return [(nbr[i], dist[i]) for i in range(treated_emb.shape[0])]

    def _bandwidth(self, candidates: list[tuple[np.ndarray, np.ndarray]]) -> float:
        if isinstance(self.bandwidth, (int, float)):
            return float(self.bandwidth)
        # Median candidate distance: a robust scale that keeps bounded-support
        # kernels (epanechnikov/triangular/uniform) from collapsing to all-zero.
        pooled = np.concatenate([d for _, d in candidates if len(d)]) if candidates else np.array([])
        pooled = pooled[pooled > 0]
        if pooled.size == 0:
            return 1.0
        return float(np.median(pooled))

    def _match(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray | None,
        embedding: np.ndarray,
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[tuple[int, int, float]]]:
        treated_idx = np.flatnonzero(treatment == 1)
        control_idx = np.flatnonzero(treatment == 0)
        if treated_idx.size == 0 or control_idx.size == 0:
            raise ValueError('Matching needs both treated and control units in the data.')

        candidates = self._candidates(embedding[treated_idx], embedding[control_idx])
        bandwidth = self._bandwidth(candidates)
        kernel_fn = KERNELS[self.kernel]

        treated_keep: list[int] = []
        control_weight = np.zeros(control_idx.size, dtype=float)
        matrix: list[tuple[int, int, float]] = []
        for global_t, (positions, dist) in zip(treated_idx, candidates):
            if len(positions) == 0:
                continue
            w = kernel_fn(np.asarray(dist, dtype=float) / bandwidth)
            total = w.sum()
            if total <= 0:
                continue
            if self.normalize:
                w = w / total
            treated_keep.append(int(global_t))
            for pos, wj in zip(positions, w):
                if wj > 0:
                    control_weight[pos] += wj
                    matrix.append((int(global_t), int(control_idx[pos]), float(wj)))

        if not treated_keep:
            raise ValueError('No treated unit had any control with positive kernel weight; widen the bandwidth/radius.')

        used = np.flatnonzero(control_weight > 0)
        rows = np.concatenate([np.asarray(treated_keep, dtype=int), control_idx[used]])
        weight = np.concatenate([np.ones(len(treated_keep)), control_weight[used]])
        matched = self._build_frame(X, treatment, y, rows, weight)
        if self.return_weight_matrix:
            return matched, matrix
        return matched
