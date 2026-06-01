__all__ = ['NeighborBackend', 'SklearnNeighborBackend']


from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.neighbors import NearestNeighbors


@runtime_checkable
class NeighborBackend(Protocol):
    """Nearest-neighbor backend interface.

    Implement this protocol to plug a custom (e.g. approximate) neighbor search
    into ``NearestNeighborMatcher``. A backend is fitted on the control
    embeddings, then queried for the treated embeddings.
    """

    def fit(self, control_emb: np.ndarray) -> 'NeighborBackend':
        """Index the control embeddings."""
        ...

    def kneighbors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, indices)`` of the k nearest controls per query row."""
        ...

    def radius_neighbors(self, query: np.ndarray, radius: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return per-query ``(distances, indices)`` of controls within ``radius``."""
        ...


class SklearnNeighborBackend:
    """Exact neighbor backend backed by ``sklearn.neighbors.NearestNeighbors``.

    This is the default backend. Custom approximate-NN backends (for very large
    datasets) can implement the :class:`NeighborBackend` protocol instead.

    Args:
        metric: Distance metric name or callable accepted by sklearn.
        metric_params (dict): Extra metric parameters (e.g. ``{'VI': inv_cov}`` for
            the Mahalanobis metric).
        **kwargs: Forwarded to ``NearestNeighbors``.
    """

    def __init__(self, metric: Any = 'minkowski', metric_params: dict | None = None, **kwargs: Any):
        self.metric = metric
        self.metric_params = metric_params
        self.kwargs = kwargs
        self._nn: NearestNeighbors | None = None
        self._n_control = 0

    def fit(self, control_emb: np.ndarray) -> 'SklearnNeighborBackend':
        self._n_control = control_emb.shape[0]
        self._nn = NearestNeighbors(
            metric=self.metric, metric_params=self.metric_params, **self.kwargs,
        ).fit(control_emb)
        return self

    def kneighbors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._nn.kneighbors(query, n_neighbors=min(k, self._n_control))

    def radius_neighbors(self, query: np.ndarray, radius: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        dist, idx = self._nn.radius_neighbors(query, radius=radius)
        return list(dist), list(idx)
