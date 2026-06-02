__all__ = ['BaseMatcher']


import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

from ..common._uplift_model import _row_subset, _to_array, _to_numpy_1d


class BaseMatcher:
    """Transformer-style template for covariate matching.

    A matcher learns a low-dimensional embedding of the covariates on `fit`
    (a propensity score, a whitened feature space, ...) and on `transform`
    pairs each treated unit with its nearest control unit(s) in that embedding,
    returning a balanced (ATT-weighted) matched sample as a DataFrame.

    Subclasses implement two methods:
    - ``_fit_embedding`` — learn whatever the metric needs from the training data.
    - ``_embed`` — map a feature matrix to the numeric space matching runs in,
      where Euclidean distance is the matching distance.

    Args:
        n_neighbors (int): Number of control units matched to each treated unit.
        caliper (float): Maximum matching distance; treated units with no control
            within the caliper are dropped. None disables the caliper.
        replace (bool): If True, a control unit may match several treated units
            (matching with replacement). If False, matching is greedy and each
            control is used at most once.
        alias (str): Optional display name.
    """

    alias: str | None = None

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: float | None = None,
        replace: bool = True,
        alias: str | None = None,
    ):
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError(f'n_neighbors must be a positive int, got {n_neighbors!r}.')
        if caliper is not None and caliper <= 0:
            raise ValueError(f'caliper must be positive or None, got {caliper!r}.')
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replace = replace
        self.alias = alias
        self._fitted = False

    @property
    def display_name(self) -> str:
        return self.alias or type(self).__name__

    def fit(self, X: ArrayLike, treatment: ArrayLike, y: ArrayLike | None = None) -> 'BaseMatcher':
        """Learn the matching metric from the (training) covariates and treatment."""
        self._fit_embedding(_to_array(X), _to_numpy_1d(treatment).astype(int))
        self._fitted = True
        return self

    def transform(self, X: ArrayLike, treatment: ArrayLike, y: ArrayLike | None = None) -> pd.DataFrame:
        """Match units in (X, treatment) and return the matched sample.

        Returns:
            A DataFrame with the original feature columns plus ``treatment``,
            ``y`` (when provided), and a ``weight`` column (1 for every treated
            unit, ``times_matched / n_neighbors`` for each control unit).
        """
        if not self._fitted:
            raise RuntimeError(f'{type(self).__name__} has not been fitted yet. Call .fit() first.')
        X_arr = _to_array(X)
        treatment_arr = _to_numpy_1d(treatment).astype(int)
        y_arr = None if y is None else _to_numpy_1d(y)
        return self._match(X_arr, treatment_arr, y_arr, self._embed(X_arr))

    def fit_transform(self, X: ArrayLike, treatment: ArrayLike, y: ArrayLike | None = None) -> pd.DataFrame:
        return self.fit(X, treatment, y).transform(X, treatment, y)

    def _match(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray | None,
        embedding: np.ndarray,
    ) -> pd.DataFrame:
        treated_idx = np.flatnonzero(treatment == 1)
        control_idx = np.flatnonzero(treatment == 0)
        if treated_idx.size == 0 or control_idx.size == 0:
            raise ValueError('Matching needs both treated and control units in the data.')

        matches = self._nearest_neighbors(embedding[treated_idx], embedding[control_idx])

        treated_keep: list[int] = []
        control_counts: dict[int, int] = {}
        for global_t, neigh in zip(treated_idx, matches):
            if neigh.size == 0:
                continue
            treated_keep.append(int(global_t))
            for pos in neigh:
                control_counts[int(pos)] = control_counts.get(int(pos), 0) + 1

        if not treated_keep:
            raise ValueError('No treated unit found a control within the caliper; relax `caliper`.')

        control_positions = np.fromiter(control_counts, dtype=int)
        control_global = control_idx[control_positions]
        control_weight = np.array([control_counts[p] for p in control_positions], dtype=float) / self.n_neighbors

        rows = np.concatenate([np.asarray(treated_keep, dtype=int), control_global])
        weight = np.concatenate([np.ones(len(treated_keep)), control_weight])
        return self._build_frame(X, treatment, y, rows, weight)

    def _require_enough_controls(self, n_control: int) -> None:
        # k-NN matching cannot draw n_neighbors distinct controls when fewer exist;
        # surface that as a clear error instead of silently matching fewer than asked.
        if self.n_neighbors > n_control:
            raise ValueError(
                f'n_neighbors={self.n_neighbors} exceeds the {n_control} available control '
                'unit(s); reduce n_neighbors or supply more control units.'
            )

    def _nearest_neighbors(self, treated_emb: np.ndarray, control_emb: np.ndarray) -> list[np.ndarray]:
        """Return, per treated unit, the positions of its matched controls (within the caliper)."""
        n_control = control_emb.shape[0]
        self._require_enough_controls(n_control)
        if self.replace:
            k = min(self.n_neighbors, n_control)
            dist, nbr = NearestNeighbors(n_neighbors=k).fit(control_emb).kneighbors(treated_emb)
            if self.caliper is None:
                return [row for row in nbr]
            return [nbr[i][dist[i] <= self.caliper] for i in range(treated_emb.shape[0])]

        # Without replacement: greedy assignment over distance-sorted neighbor lists.
        # When a caliper is set, use radius search to pre-filter candidates instead of
        # fetching all n_control neighbours and discarding most of them in the loop.
        nn = NearestNeighbors().fit(control_emb)
        if self.caliper is not None:
            r_dist, r_nbr = nn.radius_neighbors(treated_emb, radius=self.caliper)
            sorted_pairs = []
            for d_row, n_row in zip(r_dist, r_nbr):
                d_arr, n_arr = np.asarray(d_row), np.asarray(n_row, dtype=int)
                order = np.argsort(d_arr)
                sorted_pairs.append(n_arr[order])
        else:
            dist, nbr = nn.kneighbors(treated_emb, n_neighbors=n_control)
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

    @staticmethod
    def _build_frame(
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray | None,
        rows: np.ndarray,
        weight: np.ndarray,
    ) -> pd.DataFrame:
        features = _row_subset(X, rows)
        if isinstance(features, pd.DataFrame):
            out = features.reset_index(drop=True)
        else:
            features = np.asarray(features)
            out = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        out['treatment'] = treatment[rows]
        if y is not None:
            out['y'] = y[rows]
        out['weight'] = weight
        return out

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        raise NotImplementedError

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
