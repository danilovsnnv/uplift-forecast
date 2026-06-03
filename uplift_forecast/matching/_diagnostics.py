__all__ = [
    'covariate_balance',
    'match_rate',
    'overlap_report',
    'positivity_check',
    'standardized_mean_difference',
    'variance_ratio',
]


import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..common._uplift_model import _to_array, _to_numpy_1d

# Conventional balance thresholds (Austin 2011; Stuart 2010): |SMD| <= 0.1 and a
# treated/control variance ratio in [0.5, 2.0] are treated as acceptable balance.
_SMD_THRESHOLD = 0.1
_VARIANCE_RATIO_LOW = 0.5
_VARIANCE_RATIO_HIGH = 2.0


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


def variance_ratio(X: ArrayLike, treatment: ArrayLike) -> np.ndarray:
    """Per-feature treated/control variance ratio ``var(t=1) / var(t=0)``.

    A ratio far from 1 signals that the two arms differ in spread (not just mean),
    which standardized mean differences alone do not catch. The conventional
    acceptable range is ``[0.5, 2.0]``. Features with zero control variance get
    ``inf`` when the treated variance is positive and ``1.0`` when both are zero.

    Args:
        X: Feature matrix.
        treatment: Binary treatment vector (0/1).

    Returns:
        Array of one variance ratio per feature.

    Raises:
        ValueError: If either arm is empty.
    """
    x = np.asarray(_to_array(X), dtype=float)
    t = _to_numpy_1d(treatment).astype(int)
    treated, control = t == 1, t == 0
    if not treated.any() or not control.any():
        raise ValueError('variance_ratio needs both treated and control units.')
    var_t = x[treated].var(axis=0)
    var_c = x[control].var(axis=0)
    ratio = np.where(var_c > 0, var_t / np.where(var_c > 0, var_c, 1.0), np.inf)
    return np.where((var_c == 0) & (var_t == 0), 1.0, ratio)


def positivity_check(
    propensity: ArrayLike,
    treatment: ArrayLike | None = None,
    *,
    low: float = 0.05,
    high: float = 0.95,
) -> dict:
    """Positivity / common-support check on propensity scores.

    Flags near-deterministic propensities: regions with only treated or only
    control units violate the overlap (positivity) assumption, so treatment-effect
    estimates do not generalise there.

    Args:
        propensity: Estimated ``P(T=1|X)`` per unit.
        treatment: Optional 0/1 treatment vector; when given, the per-arm
            propensity ranges are reported too.
        low: Lower overlap bound; scores below it are out of common support.
        high: Upper overlap bound; scores above it are out of common support.

    Returns:
        Dict with ``share_outside_overlap`` (fraction of units outside
        ``[low, high]``), ``min_propensity``, ``max_propensity``, and a boolean
        ``has_overlap`` (no unit pinned to 0/1 beyond the bounds). When
        ``treatment`` is provided, also ``min_treated_propensity`` and
        ``max_control_propensity``.
    """
    e = _to_numpy_1d(propensity).astype(float)
    outside = (e < low) | (e > high)
    report = {
        'share_outside_overlap': float(np.mean(outside)),
        'min_propensity': float(e.min()),
        'max_propensity': float(e.max()),
        'has_overlap': bool(not outside.all()),
    }
    if treatment is not None:
        t = _to_numpy_1d(treatment).astype(int)
        if (t == 1).any():
            report['min_treated_propensity'] = float(e[t == 1].min())
        if (t == 0).any():
            report['max_control_propensity'] = float(e[t == 0].max())
    return report


def overlap_report(
    X: ArrayLike,
    treatment: ArrayLike,
    propensity: ArrayLike | None = None,
    feature_names: list | None = None,
) -> dict:
    """One-call balance & overlap report (SMD + variance ratio + positivity).

    Aggregates the standardized mean difference and variance ratio per feature
    and, when ``propensity`` is given, the positivity check, then grades each
    against the conventional thresholds (``|SMD| <= 0.1`` and variance ratio in
    ``[0.5, 2.0]``).

    Args:
        X: Feature matrix.
        treatment: Binary treatment vector (0/1).
        propensity: Optional estimated propensity scores for the positivity check.
        feature_names: Optional feature names; inferred from a DataFrame ``X``.

    Returns:
        Dict with a per-feature ``balance`` DataFrame (``smd``, ``variance_ratio``,
        ``smd_ok``, ``variance_ratio_ok``), a ``balanced`` flag (all features pass),
        the ``max_abs_smd``, and a ``positivity`` sub-report when ``propensity`` is given.
    """
    x_arr = _to_array(X)
    if feature_names is None:
        feature_names = list(x_arr.columns) if isinstance(x_arr, pd.DataFrame) \
            else [f'feature_{i}' for i in range(np.asarray(x_arr).shape[1])]

    smd = standardized_mean_difference(X, treatment)
    vratio = variance_ratio(X, treatment)
    smd_ok = np.abs(smd) <= _SMD_THRESHOLD
    vratio_ok = (vratio >= _VARIANCE_RATIO_LOW) & (vratio <= _VARIANCE_RATIO_HIGH)
    balance = pd.DataFrame(
        {'smd': smd, 'variance_ratio': vratio, 'smd_ok': smd_ok, 'variance_ratio_ok': vratio_ok},
        index=pd.Index(feature_names, name='feature'),
    )
    report = {
        'balance': balance,
        'balanced': bool(smd_ok.all() and vratio_ok.all()),
        'max_abs_smd': float(np.abs(smd).max()),
    }
    if propensity is not None:
        report['positivity'] = positivity_check(propensity, treatment)
    return report
