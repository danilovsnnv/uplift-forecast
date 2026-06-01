__all__ = ['covariate_balance', 'match_rate', 'standardized_mean_difference']


import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..common._uplift_model import _to_array, _to_numpy_1d


def _mean_var(x: np.ndarray, weight: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if weight is None:
        return x.mean(axis=0), x.var(axis=0)
    w = weight / weight.sum()
    mean = (w[:, None] * x).sum(axis=0)
    var = (w[:, None] * np.square(x - mean)).sum(axis=0)
    return mean, var


def standardized_mean_difference(
    X: ArrayLike,
    treatment: ArrayLike,
    weight: ArrayLike | None = None,
) -> np.ndarray:
    """Per-feature standardized mean difference (SMD) between arms.

    ``SMD_j = (mean_treated_j - mean_control_j) / sqrt((var_treated_j + var_control_j) / 2)``.
    Pass ``weight=None`` for the raw (before-matching) balance and the matched
    ``weight`` column for the after-matching balance. Features with zero pooled
    standard deviation get an SMD of 0 (no division by zero).

    Args:
        X: Feature matrix.
        treatment: Binary treatment vector (0/1).
        weight: Optional per-row weights aligned to ``X``.

    Returns:
        Array of one SMD per feature.
    """
    x = np.asarray(_to_array(X), dtype=float)
    t = _to_numpy_1d(treatment).astype(int)
    w = None if weight is None else _to_numpy_1d(weight).astype(float)
    treated = t == 1
    control = t == 0
    if not treated.any() or not control.any():
        raise ValueError('standardized_mean_difference needs both treated and control units.')

    mean_t, var_t = _mean_var(x[treated], None if w is None else w[treated])
    mean_c, var_c = _mean_var(x[control], None if w is None else w[control])
    pooled = np.sqrt((var_t + var_c) / 2.0)
    return np.where(pooled > 0, (mean_t - mean_c) / pooled, 0.0)


def covariate_balance(
    X: ArrayLike,
    treatment: ArrayLike,
    weight: ArrayLike | None = None,
    feature_names: list | None = None,
) -> pd.DataFrame:
    """Covariate balance table (per-arm means and SMD).

    Call with ``weight=None`` for the before-matching balance and with the matched
    ``weight`` column for the after-matching balance, then compare the ``smd``
    columns.

    Args:
        X: Feature matrix.
        treatment: Binary treatment vector (0/1).
        weight: Optional per-row weights aligned to ``X``.
        feature_names: Optional feature names; inferred from a DataFrame ``X``.

    Returns:
        DataFrame indexed by feature with columns ``mean_treated``,
        ``mean_control`` and ``smd``.
    """
    x_arr = _to_array(X)
    if feature_names is None:
        feature_names = list(x_arr.columns) if isinstance(x_arr, pd.DataFrame) \
            else [f'feature_{i}' for i in range(np.asarray(x_arr).shape[1])]

    x = np.asarray(x_arr, dtype=float)
    t = _to_numpy_1d(treatment).astype(int)
    w = None if weight is None else _to_numpy_1d(weight).astype(float)
    mean_t, _ = _mean_var(x[t == 1], None if w is None else w[t == 1])
    mean_c, _ = _mean_var(x[t == 0], None if w is None else w[t == 0])
    smd = standardized_mean_difference(X, treatment, weight)
    return pd.DataFrame(
        {'mean_treated': mean_t, 'mean_control': mean_c, 'smd': smd},
        index=pd.Index(feature_names, name='feature'),
    )


def match_rate(original_treatment: ArrayLike, matched_treatment: ArrayLike) -> float:
    """Fraction of originally-treated units retained after matching.

    Args:
        original_treatment: Treatment vector before matching.
        matched_treatment: ``treatment`` column of the matched sample.

    Returns:
        ``matched_treated / original_treated`` in ``[0, 1]``.

    Raises:
        ValueError: If there are no treated units in ``original_treatment``.
    """
    original = _to_numpy_1d(original_treatment).astype(int)
    matched = _to_numpy_1d(matched_treatment).astype(int)
    n_original = int((original == 1).sum())
    if n_original == 0:
        raise ValueError('original_treatment contains no treated units.')
    return float((matched == 1).sum()) / float(n_original)
