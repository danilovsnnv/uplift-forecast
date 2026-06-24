"""Uplift evaluation metrics.

Implements AUUC and Qini coefficients from scratch (no `causalml` dependency)
and ports the component MAE/MSE/MAPE metrics from the legacy `src/metrics.py`.

Conventions
-----------
- `uplift` is a 1-D array of predicted individual treatment effects, higher = better.
  For multi-arm models it is a `[n, K-1]` array (one column per treated arm vs control).
- `treatment` is a 1-D integer array; `{0, 1}` for binary, `{0, 1, ..., K-1}` (0 = control)
  for multi-arm.
- `y_true` is the observed outcome.
- `auuc_score` / `qini_score` return a single float for binary treatment and a
  `{arm: score}` dict for multi-arm treatment (one score per treated arm vs control).
- AUUC and Qini are normalised against the perfect-ordering curve, so values
  in `[0, 1]` are expected on well-calibrated models. Negative values are
  possible if the model is anti-correlated with the true uplift.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import auc, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# numpy renamed `trapz` to `trapezoid` in 2.0 and removed the old name; resolve
# whichever exists so we support numpy 1.24+ and 2.x alike.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # noqa: NPY201

__all__ = [
    'arm_score_summary',
    'auuc_score',
    'best_dose',
    'cost_based_targeting_curve',
    'cumulative_gain_curve',
    'dose_response_mise',
    'kendall_uplift',
    'lift_at_k',
    'optimal_treatment_assignment',
    'pehe',
    'qini_curve',
    'qini_score',
    'uplift_component_mae',
    'uplift_component_mape',
    'uplift_component_mse',
]


def _to_1d_array(arr: ArrayLike, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim > 1:
        raise ValueError(f'{name} must be 1-D; got shape {out.shape}.')
    return out.reshape(-1)


def _validate_inputs(y_true: ArrayLike, uplift: ArrayLike, treatment: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = _to_1d_array(y_true, 'y_true').astype(float)
    uplift = _to_1d_array(uplift, 'uplift').astype(float)
    treatment = _to_1d_array(treatment, 'treatment').astype(int)

    if not (len(y_true) == len(uplift) == len(treatment)):
        raise ValueError(
            f'y_true ({len(y_true)}), uplift ({len(uplift)}) and treatment '
            f'({len(treatment)}) must have the same length.',
        )
    unique = np.unique(treatment)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f'treatment must be binary (0/1); got unique values {unique.tolist()}.')
    return y_true, uplift, treatment


def cumulative_gain_curve(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Population cumulative-gain curve sorted by predicted uplift.

    At each prefix `k` of items ordered by descending `uplift`, the gain is

        gain(k) = (Y_t(k) / n_t(k) - Y_c(k) / n_c(k)) * k

    where `Y_t/c(k)` are cumulative outcomes among the treated/control inside
    the prefix and `n_t/c(k)` their counts.

    For multi-arm inputs (2-D `uplift` or non-binary `treatment`) it returns a
    `{arm: (x, gain)}` dict, one curve per treated arm on its control-plus-arm subset.

    Returns:
        `(x, gain)` arrays of length `n + 1`, with `x[i] = i` and `gain[0] = 0`.
    """
    if _is_multi_arm(uplift, treatment):
        return _multi_arm_curves(y_true, uplift, treatment, cumulative_gain_curve)
    y_true, uplift, treatment = _validate_inputs(y_true, uplift, treatment)
    n = len(y_true)
    order = np.argsort(-uplift, kind='stable')
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    cum_n_t = np.cumsum(t_sorted)
    cum_n_c = np.cumsum(1 - t_sorted)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_t = np.where(cum_n_t > 0, cum_y_t / cum_n_t, 0.0)
        mean_c = np.where(cum_n_c > 0, cum_y_c / cum_n_c, 0.0)
    gain = (mean_t - mean_c) * np.arange(1, n + 1)

    x = np.concatenate([[0.0], np.arange(1, n + 1, dtype=float)])
    gain = np.concatenate([[0.0], gain])
    return x, gain


def qini_curve(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve sorted by predicted uplift.

    At each prefix `k`:

        qini(k) = Y_t(k) - Y_c(k) * (n_t(k) / n_c(k))

    Falls back to the absolute treated outcome when the prefix has no controls. For
    multi-arm inputs it returns a `{arm: (x, qini)}` dict, one curve per treated arm.
    """
    if _is_multi_arm(uplift, treatment):
        return _multi_arm_curves(y_true, uplift, treatment, qini_curve)
    y_true, uplift, treatment = _validate_inputs(y_true, uplift, treatment)
    n = len(y_true)
    order = np.argsort(-uplift, kind='stable')
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    cum_n_t = np.cumsum(t_sorted)
    cum_n_c = np.cumsum(1 - t_sorted)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cum_n_c > 0, cum_n_t / cum_n_c, 0.0)
    qini = cum_y_t - cum_y_c * ratio

    x = np.concatenate([[0.0], np.arange(1, n + 1, dtype=float)])
    qini = np.concatenate([[0.0], qini])
    return x, qini


def _normalised_auc(
    curve_fn: Callable,
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    *,
    normalize: bool,
) -> float:
    x, gain = curve_fn(y_true, uplift, treatment)
    model_auc = auc(x, gain)
    if not normalize:
        return float(model_auc)

    perfect_uplift = y_true * treatment - y_true * (1 - treatment)
    _, perfect_gain = curve_fn(y_true, perfect_uplift, treatment)
    perfect_auc = auc(x, perfect_gain)
    if perfect_auc == 0:
        return 0.0
    return float(model_auc / perfect_auc)


def _is_multi_arm(uplift: ArrayLike, treatment: ArrayLike) -> bool:
    """True when inputs describe a multi-arm problem (2-D uplift or non-binary treatment)."""
    if np.asarray(uplift).ndim > 1:
        return True
    arms = np.unique(_to_1d_array(treatment, 'treatment').astype(int))
    return not np.all(np.isin(arms, [0, 1]))


def auuc_score(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    *,
    normalize: bool = True,
) -> float | dict[int, float]:
    """Area Under the Uplift Curve.

    Args:
        y_true: Observed outcomes.
        uplift: Predicted individual treatment effects (higher = better). ``[n]`` for binary
            treatment, ``[n, K-1]`` for multi-arm (one column per treated arm vs control).
        treatment: Treatment indicator; ``{0, 1}`` for binary, ``{0, .., K-1}`` (0 = control)
            for multi-arm.
        normalize: If True (default), divide by the area under the perfect-ordering curve so
            the score is in `[0, 1]` for sensible models.

    Returns:
        A single float for binary treatment, or a ``{arm: auuc}`` dict scoring each treated
        arm on its control-plus-arm subset for multi-arm treatment.
    """
    if _is_multi_arm(uplift, treatment):
        return _multi_arm_scores(y_true, uplift, treatment, auuc_score, normalize=normalize)
    y, u, t = _validate_inputs(y_true, uplift, treatment)
    return _normalised_auc(cumulative_gain_curve, y, u, t, normalize=normalize)


def qini_score(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    *,
    normalize: bool = True,
) -> float | dict[int, float]:
    """Qini coefficient: area under the Qini curve (normalised by default).

    Returns a single float for binary treatment, or a ``{arm: qini}`` dict (one score per
    treated arm vs control) for multi-arm treatment. See :func:`auuc_score` for the argument
    conventions.
    """
    if _is_multi_arm(uplift, treatment):
        return _multi_arm_scores(y_true, uplift, treatment, qini_score, normalize=normalize)
    y, u, t = _validate_inputs(y_true, uplift, treatment)
    return _normalised_auc(qini_curve, y, u, t, normalize=normalize)


# ---------------------------------------------------------------------------
# Multi-treatment (K-arm) metrics
# ---------------------------------------------------------------------------


def _multi_arm_uplift(uplift: ArrayLike, treatment: ArrayLike) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Normalise multi-arm inputs to (uplift_2d, treatment_int, treated_arms)."""
    t = _to_1d_array(treatment, 'treatment').astype(int)
    u = np.asarray(uplift, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    arms = sorted(set(t.tolist()))
    if arms[0] != 0:
        raise ValueError(f'treatment must include a control arm coded 0; got arms {arms}.')
    treated = arms[1:]
    if u.shape[1] != len(treated):
        raise ValueError(
            f'uplift has {u.shape[1]} column(s) but there are {len(treated)} treated arm(s) {treated}.',
        )
    return u, t, treated


def _multi_arm_scores(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    score_fn: Callable,
    *,
    normalize: bool,
) -> dict[int, float]:
    y = _to_1d_array(y_true, 'y_true').astype(float)
    u, t, treated = _multi_arm_uplift(uplift, treatment)
    scores = {}
    for col, arm in enumerate(treated):
        mask = (t == 0) | (t == arm)
        sub_t = (t[mask] == arm).astype(int)
        scores[arm] = score_fn(y[mask], u[mask, col], sub_t, normalize=normalize)
    return scores


def _multi_arm_curves(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    curve_fn: Callable,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-arm curve on each treated arm's control-plus-arm subset."""
    y = _to_1d_array(y_true, 'y_true').astype(float)
    u, t, treated = _multi_arm_uplift(uplift, treatment)
    curves = {}
    for col, arm in enumerate(treated):
        mask = (t == 0) | (t == arm)
        sub_t = (t[mask] == arm).astype(int)
        curves[arm] = curve_fn(y[mask], u[mask, col], sub_t)
    return curves


def lift_at_k(
    y_true: ArrayLike,
    uplift: ArrayLike,
    treatment: ArrayLike,
    k: float = 0.3,
) -> float | dict[int, float]:
    """Cumulative gain when targeting the top-``k`` fraction by predicted uplift (LIFT@K).

    The incremental response captured by treating the highest-scored ``k`` share of the
    population, read off the cumulative-gain curve. Returns a single float for binary
    treatment or a ``{arm: lift}`` dict for multi-arm treatment.

    Args:
        y_true: Observed outcomes.
        uplift: Predicted uplift, ``[n]`` or ``[n, K-1]``.
        treatment: Treatment indicator (``{0, 1}`` binary or ``{0, .., K-1}`` multi-arm).
        k: Fraction of the population to target, in ``(0, 1]``.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f'k must be a fraction in (0, 1]; got {k}.')
    if _is_multi_arm(uplift, treatment):
        curves = _multi_arm_curves(y_true, uplift, treatment, cumulative_gain_curve)
        return {arm: _gain_at_fraction(x, gain, k) for arm, (x, gain) in curves.items()}
    x, gain = cumulative_gain_curve(y_true, uplift, treatment)
    return _gain_at_fraction(x, gain, k)


def _gain_at_fraction(x: np.ndarray, gain: np.ndarray, k: float) -> float:
    return float(gain[round(k * (len(x) - 1))])


def pehe(true_effect: ArrayLike, pred_effect: ArrayLike) -> float | dict[int, float]:
    """Precision in Estimation of Heterogeneous Effect: ``sqrt(mean((tau_hat - tau)^2))``.

    Requires the ground-truth effect, so it is a synthetic-data metric. With 2-D inputs
    (``[n, K-1]``, columns aligned to sorted treated arms) it returns a ``{arm: pehe}``
    dict (aggregate it with :func:`arm_score_summary` for mPEHE / sdPEHE).

    Args:
        true_effect: Ground-truth effect, ``[n]`` or ``[n, K-1]``.
        pred_effect: Predicted effect, same shape as ``true_effect``.
    """
    true = np.asarray(true_effect, dtype=float)
    pred = np.asarray(pred_effect, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f'true_effect {true.shape} and pred_effect {pred.shape} must match.')
    if true.ndim == 1:
        return float(np.sqrt(np.mean((pred - true) ** 2)))
    return {col + 1: float(np.sqrt(np.mean((pred[:, col] - true[:, col]) ** 2))) for col in range(true.shape[1])}


def kendall_uplift(true_effect: ArrayLike, pred_effect: ArrayLike) -> float | dict[int, float]:
    """Kendall's tau-b rank correlation between predicted and true effect (synthetic-data).

    Measures how well a model *ranks* units by effect. With 2-D inputs it returns a
    ``{arm: tau}`` dict (aggregate with :func:`arm_score_summary` for mKendall / sdKendall).
    """
    from scipy.stats import kendalltau  # noqa: PLC0415  (scipy is a sklearn dependency; lazy)

    true = np.asarray(true_effect, dtype=float)
    pred = np.asarray(pred_effect, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f'true_effect {true.shape} and pred_effect {pred.shape} must match.')
    if true.ndim == 1:
        return float(kendalltau(pred, true).statistic)
    return {col + 1: float(kendalltau(pred[:, col], true[:, col]).statistic) for col in range(true.shape[1])}


def arm_score_summary(scores: float | dict[int, float]) -> tuple[float, float]:
    """Mean and (population) standard deviation of a per-arm score dict.

    Turns the ``{arm: score}`` dict returned by :func:`auuc_score` / :func:`qini_score`
    / :func:`pehe` / :func:`kendall_uplift` into the mean-plus-dispersion pair reported
    in the multi-treatment literature (e.g. mQini / sdQini). A scalar (binary) score
    returns ``(score, 0.0)``.

    Returns:
        ``(mean, std)`` over the arms.
    """
    if not isinstance(scores, dict):
        return float(scores), 0.0
    values = np.asarray(list(scores.values()), dtype=float)
    return float(values.mean()), float(values.std())


def optimal_treatment_assignment(uplift: ArrayLike, costs: ArrayLike | None = None) -> np.ndarray:
    """Cost-aware best arm per unit (Zhao & Harinen, arXiv:1908.05372).

    Assigns each unit the arm maximising ``uplift_k - cost_k``; if no arm has a
    positive net effect the unit is left in control (arm 0).

    Args:
        uplift: Predicted uplift, ``[n]`` or ``[n, K-1]`` (one column per treated arm).
        costs: Optional per-arm cost, scalar or ``[K-1]`` (defaults to 0).

    Returns:
        Integer arm per unit (``0`` = control, ``k`` = treated arm ``k``).
    """
    u = np.asarray(uplift, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    cost = 0.0 if costs is None else np.asarray(costs, dtype=float).reshape(1, -1)
    net = u - cost
    best_col = net.argmax(axis=1)
    best_val = net[np.arange(len(net)), best_col]
    return np.where(best_val > 0, best_col + 1, 0).astype(int)


def cost_based_targeting_curve(
    uplift: ArrayLike,
    costs: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Model-expected net-uplift targeting curve for cost-aware multi-arm targeting.

    Units are ordered by their best cost-adjusted net uplift; the curve is the
    cumulative expected net uplift as more of the population is targeted with its
    best arm. This is a model-based planning curve (how many to target), not a
    realised-value estimate — for that, score the induced policy with
    ``uplift_forecast.ope``.

    Args:
        uplift: Predicted uplift, ``[n]`` or ``[n, K-1]``.
        costs: Optional per-arm cost, scalar or ``[K-1]``.

    Returns:
        ``(x, value)`` arrays of length ``n + 1`` with ``x[i] = i`` (units targeted)
        and ``value`` the cumulative expected net uplift.
    """
    u = np.asarray(uplift, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    cost = 0.0 if costs is None else np.asarray(costs, dtype=float).reshape(1, -1)
    best_val = (u - cost).max(axis=1)
    order = np.argsort(-best_val, kind='stable')
    cum = np.concatenate([[0.0], np.cumsum(best_val[order])])
    x = np.arange(len(u) + 1, dtype=float)
    return x, cum


# ---------------------------------------------------------------------------
# Continuous-treatment (dose-response) metrics
# ---------------------------------------------------------------------------


def dose_response_mise(true_curves: ArrayLike, pred_curves: ArrayLike, t_grid: ArrayLike) -> float:
    """Mean Integrated Squared Error between true and predicted dose-response curves.

    ``MISE = mean_x integral_t (mu_hat(x, t) - mu(x, t))^2 dt`` (trapezoidal over the
    dose grid). Requires the ground-truth ADRF, so it is a synthetic-data metric.

    Args:
        true_curves: True dose-response ``[n, len(t_grid)]``.
        pred_curves: Predicted dose-response ``[n, len(t_grid)]``.
        t_grid: Dose values the columns correspond to.

    Returns:
        The MISE (lower is better).
    """
    true = np.asarray(true_curves, dtype=float)
    pred = np.asarray(pred_curves, dtype=float)
    grid = np.asarray(t_grid, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f'true_curves {true.shape} and pred_curves {pred.shape} must match.')
    if true.shape[1] != len(grid):
        raise ValueError(f'curves have {true.shape[1]} doses but t_grid has {len(grid)}.')
    integrated = _trapezoid((pred - true) ** 2, grid, axis=1)
    return float(np.mean(integrated))


def best_dose(pred_curves: ArrayLike, t_grid: ArrayLike) -> np.ndarray:
    """Per-unit dose maximising the predicted response over the grid; shape ``[n]``."""
    pred = np.asarray(pred_curves, dtype=float)
    grid = np.asarray(t_grid, dtype=float).reshape(-1)
    if pred.shape[1] != len(grid):
        raise ValueError(f'pred_curves have {pred.shape[1]} doses but t_grid has {len(grid)}.')
    return grid[pred.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Component metrics (per-outcome MAE/MSE/MAPE)
# ---------------------------------------------------------------------------


def _component_metric(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    y_pred_ct_col: str,
    y_pred_tr_col: str,
    metric_func: Callable,
) -> tuple[float, float, float]:
    n = df.shape[0]
    n_tr = int(df[treatment_col].sum())
    n_ct = n - n_tr

    is_ct = df[treatment_col] == 0
    is_tr = df[treatment_col] == 1
    metric_ct = metric_func(df[outcome_col][is_ct], df[y_pred_ct_col][is_ct])
    metric_tr = metric_func(df[outcome_col][is_tr], df[y_pred_tr_col][is_tr])
    metric_total = float((metric_ct * n_ct + metric_tr * n_tr) / n)
    return float(metric_ct), float(metric_tr), metric_total


def uplift_component_mae(
    df: pd.DataFrame,
    outcome_col: str = 'y',
    treatment_col: str = 'w',
    y_pred_ct_col: str = 'y_pred_ct',
    y_pred_tr_col: str = 'y_pred_tr',
) -> tuple[float, float, float]:
    """Return `(mae_ct, mae_tr, mae_weighted_total)`."""
    return _component_metric(
        df, outcome_col, treatment_col, y_pred_ct_col, y_pred_tr_col, mean_absolute_error,
    )


def uplift_component_mse(
    df: pd.DataFrame,
    outcome_col: str = 'y',
    treatment_col: str = 'w',
    y_pred_ct_col: str = 'y_pred_ct',
    y_pred_tr_col: str = 'y_pred_tr',
) -> tuple[float, float, float]:
    """Return `(mse_ct, mse_tr, mse_weighted_total)`."""
    return _component_metric(
        df, outcome_col, treatment_col, y_pred_ct_col, y_pred_tr_col, mean_squared_error,
    )


def uplift_component_mape(
    df: pd.DataFrame,
    outcome_col: str = 'y',
    treatment_col: str = 'w',
    y_pred_ct_col: str = 'y_pred_ct',
    y_pred_tr_col: str = 'y_pred_tr',
) -> tuple[float, float, float]:
    """Return `(mape_ct, mape_tr, mape_weighted_total)`."""
    return _component_metric(
        df, outcome_col, treatment_col, y_pred_ct_col, y_pred_tr_col, mean_absolute_percentage_error,
    )
