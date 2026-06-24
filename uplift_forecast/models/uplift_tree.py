__all__ = ['UpliftTree']


from typing import Any

import numpy as np

from ..common._base_meta import _BaseMultiArmLearner
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
    __slots__ = ('feature', 'left', 'rates', 'right', 'threshold')

    def __init__(self, rates: dict[int, float]):
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.rates = rates  # per-arm outcome rate at this node


class UpliftTree(_BaseMultiArmLearner):
    """Uplift decision tree with divergence split criteria (binary outcome), binary or multi-arm.

    Grows a tree whose splits maximise the gain in divergence between the treated and
    control outcome rates (KL, Euclidean, or Chi-square), as in CausalML's
    ``UpliftTreeClassifier``. Each leaf reports per-arm rates and the uplift for arm
    ``k`` is ``p_k - p_0``. The outcome must be binary (0/1); this is provided mainly
    for parity and interpretability.

    With ``K`` arms two split strategies are available via ``multi_arm_split``:

    - ``'multi_way'`` — one shared tree whose split criterion sums the divergence of
      every treated arm against control; each leaf stores all per-arm rates and
      ``predict`` reports ``y0 = p_0(x)``, ``y1 = [p_k(x)]``.
    - ``'per_arm'`` — one independent two-arm tree per treated arm vs control; each
      tree's uplift ``p_k - p_0`` is reported, so ``predict`` reports ``y0 = 0`` and
      ``y1 = uplift_k`` (no shared baseline across arms).

    With a binary treatment the two strategies coincide (a single two-arm tree) and
    ``predict`` collapses to a flat ``[n]`` uplift with ``y0 = p_c``, ``y1 = p_t``.

    Args:
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum rows per child node.
        min_samples_treatment: Minimum rows per arm in each child for a valid
            divergence estimate (enforced for every arm in a ``multi_way`` split).
        criterion: Split criterion — ``'kl'``, ``'ed'`` (Euclidean), or ``'chi'``.
        multi_arm_split: ``'multi_way'`` (one shared multi-treatment tree) or
            ``'per_arm'`` (one two-arm tree per treated arm). Only affects ``K > 2``.
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
        multi_arm_split: str = 'multi_way',
        max_candidates: int = 32,
        min_gain: float = 0.0,
        alias: str | None = None,
    ):
        super().__init__(alias=alias)
        if criterion not in ('kl', 'ed', 'chi'):
            raise ValueError(f"criterion must be one of 'kl', 'ed', 'chi'; got {criterion!r}.")
        if multi_arm_split not in ('multi_way', 'per_arm'):
            raise ValueError(f"multi_arm_split must be 'multi_way' or 'per_arm'; got {multi_arm_split!r}.")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.criterion = criterion
        self.multi_arm_split = multi_arm_split
        self.max_candidates = max_candidates
        self.min_gain = min_gain
        self._root: _Node | None = None         # shared tree (binary / multi_way)
        self._trees: dict[int, _Node] = {}        # one two-arm tree per arm (per_arm)

    def _fit_arms(
        self, X: Any, treatment: np.ndarray, y: np.ndarray, eval_set: tuple | None = None, **fit_params: Any,
    ) -> None:
        del eval_set, fit_params
        x = np.asarray(_to_array(X), dtype=np.float64)
        t = treatment.astype(int)
        y = y.astype(np.float64)
        if len(self.arms_) == 2 or self.multi_arm_split == 'multi_way':
            self._root = self._build(x, t, y, self.arms_, depth=0)
            self._trees = {}
        else:
            self._root = None
            self._trees = {arm: self._build(*self._subset(x, t, y, arm), [0, arm], depth=0) for arm in self.arms_[1:]}

    @staticmethod
    def _subset(x: np.ndarray, t: np.ndarray, y: np.ndarray, arm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = (t == 0) | (t == arm)
        return x[mask], t[mask], y[mask]

    def _build(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, arms: list[int], depth: int) -> _Node:
        node = _Node(self._rates(t, y, arms))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        parent_div = self._multi_divergence(node.rates, arms)
        best = self._best_split(x, t, y, arms, parent_div)
        if best is None:
            return node
        feature, threshold, left_mask = best
        node.feature, node.threshold = feature, threshold
        node.left = self._build(x[left_mask], t[left_mask], y[left_mask], arms, depth + 1)
        node.right = self._build(x[~left_mask], t[~left_mask], y[~left_mask], arms, depth + 1)
        return node

    def _rates(self, t: np.ndarray, y: np.ndarray, arms: list[int]) -> dict[int, float]:
        return {arm: float(y[t == arm].mean()) if (t == arm).any() else 0.0 for arm in arms}

    def _multi_divergence(self, rates: dict[int, float], arms: list[int]) -> float:
        return sum(_divergence(rates[arm], rates[0], self.criterion) for arm in arms[1:])

    def _best_split(self, x: np.ndarray, t: np.ndarray, y: np.ndarray, arms: list[int], parent_div: float) -> tuple | None:
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
                gain = self._split_gain(t, y, arms, left, n, parent_div)
                if gain is not None and gain > best_gain:
                    best_gain = gain
                    best = (feature, float(threshold), left)
        return best

    def _split_gain(
        self,
        t: np.ndarray,
        y: np.ndarray,
        arms: list[int],
        left: np.ndarray,
        n: int,
        parent_div: float,
    ) -> float | None:
        right = ~left
        if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
            return None
        div = 0.0
        for mask in (left, right):
            if any(int((t[mask] == arm).sum()) < self.min_samples_treatment for arm in arms):
                return None
            div += (mask.sum() / n) * self._multi_divergence(self._rates(t[mask], y[mask], arms), arms)
        return div - parent_div

    def _predict_arm(self, X: Any, arm: int) -> np.ndarray:
        x = np.asarray(_to_array(X), dtype=np.float64)
        if self._root is not None:
            return np.array([self._leaf(self._root, row).rates[arm] for row in x], dtype=np.float64)
        if arm == 0:
            return np.zeros(len(x), dtype=np.float64)
        leaves = (self._leaf(self._trees[arm], row) for row in x)
        return np.array([leaf.rates[arm] - leaf.rates[0] for leaf in leaves], dtype=np.float64)

    @staticmethod
    def _leaf(root: _Node, row: np.ndarray) -> _Node:
        node = root
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node
