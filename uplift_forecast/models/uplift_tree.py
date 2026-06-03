__all__ = ['UpliftTree']


from typing import Any

import numpy as np
import pandas as pd

from ..common._base_meta import BaseMetaUpliftModel
from ..common._uplift_model import _to_array

_EPS = 1e-6


def _clip_prob(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def _divergence(p_t: float, p_c: float, criterion: str) -> float:
    """Divergence between treated/control outcome rates for a binary outcome."""
    if criterion == 'ed':  # Euclidean distance
        return 2.0 * (p_t - p_c) ** 2
    p_t, p_c = _clip_prob(p_t), _clip_prob(p_c)
    if criterion == 'kl':
        return p_t * np.log(p_t / p_c) + (1 - p_t) * np.log((1 - p_t) / (1 - p_c))
    if criterion == 'chi':
        return (p_t - p_c) ** 2 / p_c + ((1 - p_t) - (1 - p_c)) ** 2 / (1 - p_c)
    raise ValueError(f"criterion must be one of 'kl', 'ed', 'chi'; got {criterion!r}.")


class _Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'p_t', 'p_c')

    def __init__(self, p_t: float, p_c: float):
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.p_t = p_t
        self.p_c = p_c


class UpliftTree(BaseMetaUpliftModel):
    """Uplift decision tree with divergence split criteria (binary outcome).

    Grows a single tree whose splits maximise the gain in divergence between the
    treated and control outcome rates (KL, Euclidean, or Chi-square), as in
    CausalML's ``UpliftTreeClassifier``. Each leaf reports control / treated rates
    ``(p_c, p_t)`` and the uplift is ``p_t - p_c``. The outcome must be binary
    (0/1); this is provided mainly for parity and interpretability.

    Args:
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum rows per child node.
        min_samples_treatment: Minimum treated and control rows per child for a
            valid divergence estimate.
        criterion: Split criterion — ``'kl'``, ``'ed'`` (Euclidean), or ``'chi'``.
        max_candidates: Max candidate thresholds evaluated per feature (quantiles).
        min_gain: Minimum divergence gain to accept a split.
        alias: Optional display name for UpliftForecast output columns.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 100,
        min_samples_treatment: int = 10,
        criterion: str = 'kl',
        max_candidates: int = 32,
        min_gain: float = 0.0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if criterion not in ('kl', 'ed', 'chi'):
            raise ValueError(f"criterion must be one of 'kl', 'ed', 'chi'; got {criterion!r}.")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.criterion = criterion
        self.max_candidates = max_candidates
        self.min_gain = min_gain
        self._root: _Node | None = None

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        del eval_set, fit_params
        x = np.asarray(_to_array(X), dtype=np.float64)
        t = treatment.astype(int)
        y = y.astype(np.float64)
        if not (t == 0).any() or not (t == 1).any():
            raise ValueError('UpliftTree requires both treated and control samples.')
        self._root = self._build(x, t, y, depth=0)

    def _rates(self, t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        p_t = float(y[t == 1].mean()) if (t == 1).any() else 0.0
        p_c = float(y[t == 0].mean()) if (t == 0).any() else 0.0
        return p_t, p_c

    def _build(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        p_t, p_c = self._rates(t, y)
        node = _Node(p_t, p_c)
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node

        parent_div = _divergence(p_t, p_c, self.criterion)
        best = self._best_split(x, t, y, parent_div)
        if best is None:
            return node

        feature, threshold, left_mask = best
        node.feature, node.threshold = feature, threshold
        node.left = self._build(x[left_mask], t[left_mask], y[left_mask], depth + 1)
        node.right = self._build(x[~left_mask], t[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, parent_div: float):
        n = len(y)
        best_gain = self.min_gain
        best = None
        for feature in range(x.shape[1]):
            col = x[:, feature]
            uniq = np.unique(col)
            if len(uniq) < 2:
                continue
            if len(uniq) > self.max_candidates:
                qs = np.linspace(0, 1, self.max_candidates + 2)[1:-1]
                candidates = np.unique(np.quantile(col, qs))
            else:
                candidates = (uniq[:-1] + uniq[1:]) / 2.0
            for threshold in candidates:
                left = col <= threshold
                gain = self._split_gain(t, y, left, n, parent_div)
                if gain is not None and gain > best_gain:
                    best_gain = gain
                    best = (feature, float(threshold), left)
        return best

    def _split_gain(self, t, y, left, n, parent_div) -> float | None:
        right = ~left
        if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
            return None
        div = 0.0
        for mask in (left, right):
            nt = int((t[mask] == 1).sum())
            nc = int((t[mask] == 0).sum())
            if nt < self.min_samples_treatment or nc < self.min_samples_treatment:
                return None
            p_t, p_c = self._rates(t[mask], y[mask])
            div += (mask.sum() / n) * _divergence(p_t, p_c, self.criterion)
        return div - parent_div

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(_to_array(X), dtype=np.float64)
        y0 = np.empty(len(x), dtype=np.float64)
        y1 = np.empty(len(x), dtype=np.float64)
        for i, row in enumerate(x):
            node = self._root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            y0[i], y1[i] = node.p_c, node.p_t
        return y0, y1
