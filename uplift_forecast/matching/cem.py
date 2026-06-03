__all__ = ['CoarsenedExactMatcher']


import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..common._uplift_model import _to_array, _to_numpy_1d
from ._base_matcher import BaseMatcher

_RARE_LABEL = '__rare__'


class CoarsenedExactMatcher(BaseMatcher):
    """Coarsened Exact Matching (CEM).

    Coarsens each covariate (numeric features into bins, categoricals into their
    levels), forms strata from the coarsened values, and matches treated and
    control units only within the same stratum. Strata containing only one arm are
    dropped. Within each retained stratum, treated and control units are
    comparable because they share the same coarsened covariate values.

    Args:
        n_bins (int): Number of bins for numeric features (when ``bin_edges`` is not
            given for that feature).
        binning (str): ``'quantile'`` (equal-frequency) or ``'uniform'`` (equal-width).
        bin_edges (dict): Optional mapping of column (name for a DataFrame, integer
            index for an ndarray) to explicit bin edges, overriding ``n_bins``/``binning``.
        categorical (list): Optional columns to treat as categorical (names/indices).
            When None, categoricals are inferred from DataFrame dtypes; ndarray inputs
            default to all-numeric.
        rare_threshold (float): Optional frequency below which a category is grouped
            into a single ``'__rare__'`` label. None disables rare grouping.
        drop_unmatched (bool): Drop strata that contain only one arm. If False they
            are kept as-is (unbalanced) with unit weights.
        alias (str): Optional display name.
    """

    def __init__(
        self,
        n_bins: int = 5,
        binning: str = 'quantile',
        bin_edges: dict | None = None,
        categorical: list | None = None,
        rare_threshold: float | None = None,
        drop_unmatched: bool = True,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if binning not in ('quantile', 'uniform'):
            raise ValueError(f"binning must be 'quantile' or 'uniform', got {binning!r}.")
        if n_bins < 1:
            raise ValueError(f'n_bins must be >= 1, got {n_bins!r}.')
        self.n_bins = n_bins
        self.binning = binning
        self.bin_edges = bin_edges
        self.categorical = categorical
        self.rare_threshold = rare_threshold
        self.drop_unmatched = drop_unmatched
        self._columns: list | None = None
        self._is_cat: list[bool] | None = None
        self._edges: dict = {}
        self._kept_categories: dict = {}
        self.n_strata_: int | None = None
        self.n_strata_dropped_: int | None = None
        self.match_rate_: float | None = None

    def _resolve_schema(self, X: np.ndarray | pd.DataFrame) -> tuple[list, list[bool]]:
        if isinstance(X, pd.DataFrame):
            columns = list(X.columns)
            if self.categorical is not None:
                is_cat = [c in self.categorical for c in columns]
            else:
                is_cat = [not pd.api.types.is_numeric_dtype(X[c]) for c in columns]
        else:
            columns = list(range(np.asarray(X).shape[1]))
            cat = set(self.categorical) if self.categorical is not None else set()
            is_cat = [c in cat for c in columns]
        return columns, is_cat

    def _column(self, X: np.ndarray | pd.DataFrame, col: int | str) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[col].to_numpy()
        return np.asarray(X)[:, col]

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            'CoarsenedExactMatcher matches via stratum keys, not a numeric embedding. '
            'Call transform() directly.'
        )

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        self._columns, self._is_cat = self._resolve_schema(X)
        self._edges = {}
        self._kept_categories = {}
        for col, is_cat in zip(self._columns, self._is_cat, strict=False):
            values = self._column(X, col)
            if is_cat:
                if self.rare_threshold is not None:
                    labels = values.astype(str)
                    levels, counts = np.unique(labels, return_counts=True)
                    freq = counts / labels.size
                    self._kept_categories[col] = set(levels[freq >= self.rare_threshold])
            elif self.bin_edges is not None and col in self.bin_edges:
                self._edges[col] = np.asarray(self.bin_edges[col], dtype=float)
            else:
                numeric = values.astype(float)
                if self.binning == 'quantile':
                    edges = np.unique(np.quantile(numeric, np.linspace(0.0, 1.0, self.n_bins + 1)))
                else:
                    edges = np.linspace(numeric.min(), numeric.max(), self.n_bins + 1)
                self._edges[col] = edges

    def _coarsen_column(self, X: np.ndarray | pd.DataFrame, col: int | str, is_cat: bool) -> np.ndarray:
        values = self._column(X, col)
        if is_cat:
            labels = values.astype(str)
            if col in self._kept_categories:
                keep = self._kept_categories[col]
                labels = np.where(np.isin(labels, list(keep)), labels, _RARE_LABEL)
            return labels
        interior = self._edges[col][1:-1]  # inner edges define the bins for np.digitize
        return np.digitize(values.astype(float), interior)

    def transform(self, X: ArrayLike, treatment: ArrayLike, y: ArrayLike | None = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f'{type(self).__name__} has not been fitted yet. Call .fit() first.')
        X_arr = _to_array(X)
        t = _to_numpy_1d(treatment).astype(int)
        y_arr = None if y is None else _to_numpy_1d(y)
        n = t.shape[0]

        coarsened = [self._coarsen_column(X_arr, col, is_cat) for col, is_cat in zip(self._columns, self._is_cat, strict=False)]
        keys = list(zip(*[c.tolist() for c in coarsened], strict=False)) if coarsened else [()] * n

        key_to_id: dict = {}
        strata_ids = np.empty(n, dtype=int)
        for i, key in enumerate(keys):
            strata_ids[i] = key_to_id.setdefault(key, len(key_to_id))

        rows: list[int] = []
        weights: list[float] = []
        strata_out: list[int] = []
        n_dropped = 0
        treated_total = int((t == 1).sum())
        treated_kept = 0
        for sid in range(len(key_to_id)):
            idx = np.flatnonzero(strata_ids == sid)
            treated = idx[t[idx] == 1]
            control = idx[t[idx] == 0]
            if treated.size == 0 or control.size == 0:
                n_dropped += 1
                if self.drop_unmatched:
                    continue
                rows.extend(idx.tolist())
                weights.extend([1.0] * idx.size)
                strata_out.extend([sid] * idx.size)
                treated_kept += treated.size
                continue
            treated_kept += treated.size
            control_w = treated.size / control.size  # ATT weighting: balance arms per stratum
            rows.extend(treated.tolist())
            weights.extend([1.0] * treated.size)
            strata_out.extend([sid] * treated.size)
            rows.extend(control.tolist())
            weights.extend([control_w] * control.size)
            strata_out.extend([sid] * control.size)

        if not rows:
            raise ValueError('No stratum contains both treated and control units; coarsen less (fewer bins).')

        rows_arr = np.asarray(rows, dtype=int)
        out = self._build_frame(X_arr, t, y_arr, rows_arr, np.asarray(weights, dtype=float))
        out['stratum'] = np.asarray(strata_out, dtype=int)

        self.n_strata_ = len(key_to_id)
        self.n_strata_dropped_ = n_dropped
        self.match_rate_ = treated_kept / treated_total if treated_total else 0.0
        return out
