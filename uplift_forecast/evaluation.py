"""Cross-model evaluation: per-model, per-arm metric tables and policy-value scoring.

Where `metrics` holds the metric *primitives* (AUUC, Qini, LIFT@K, ...), this module is
the *report* layer: it runs a fitted `UpliftForecast`'s models over the same data and
collects per-arm scores into one comparison table, and scores a recommended-treatment
policy on real data via the IPW Expected Response. Tables are returned as pandas
(default) or polars (`frame='polars'`).
"""

from typing import Any

import pandas as pd
from numpy.typing import ArrayLike

from .metrics import auuc_score, lift_at_k, qini_score
from .ope import snips

__all__ = ['compare_models', 'expected_response']


_METRICS = ('auuc', 'qini', 'lift')


def _score(name: str, y: ArrayLike, uplift: ArrayLike, treatment: ArrayLike, k: float) -> float | dict[int, float]:
    if name == 'auuc':
        return auuc_score(y, uplift, treatment)
    if name == 'qini':
        return qini_score(y, uplift, treatment)
    return lift_at_k(y, uplift, treatment, k)


def _per_arm(score: float | dict[int, float]) -> dict[int, float]:
    """Normalise a binary float or multi-arm ``{arm: score}`` to a per-arm dict (binary -> arm 1)."""
    return score if isinstance(score, dict) else {1: float(score)}


def _make_frame(columns: dict[str, list], frame: str) -> Any:
    if frame == 'pandas':
        return pd.DataFrame(columns)
    if frame == 'polars':
        import polars as pl  # noqa: PLC0415  (optional dependency; imported only when requested)

        return pl.DataFrame(columns)
    raise ValueError(f"frame must be 'pandas' or 'polars'; got {frame!r}.")


def compare_models(
    forecast: Any,
    X: ArrayLike,
    y: ArrayLike,
    treatment: ArrayLike,
    *,
    metrics: tuple[str, ...] = _METRICS,
    k: float = 0.3,
    frame: str = 'pandas',
) -> Any:
    """Score every model in a fitted ``UpliftForecast`` per treatment arm into one table.

    Each model's predicted uplift is scored with the requested ranking metrics, broken
    down per treated arm (one row per ``(model, arm)``; binary models use arm ``1``).
    The result is a long-format comparison table for ranking and selecting models.

    Args:
        forecast: A fitted ``UpliftForecast``.
        X: Features to score.
        y: Observed outcomes.
        treatment: Treatment indicator (``{0, 1}`` or ``{0, .., K-1}``).
        metrics: Ranking metrics to compute, any of ``'auuc'``, ``'qini'``, ``'lift'``.
        k: Target fraction for the ``'lift'`` metric (LIFT@K).
        frame: Output frame type — ``'pandas'`` (default) or ``'polars'``.

    Returns:
        A DataFrame with columns ``model``, ``arm`` and one column per requested metric.

    Raises:
        ValueError: For an unknown metric name or frame type.
    """
    unknown = set(metrics) - set(_METRICS)
    if unknown:
        raise ValueError(f'unknown metric(s) {sorted(unknown)}; choose from {sorted(_METRICS)}.')

    columns: dict[str, list] = {'model': [], 'arm': []}
    for name in metrics:
        columns[name] = []

    for model in forecast.models:
        uplift = model.predict(X)
        scored = {name: _per_arm(_score(name, y, uplift, treatment, k)) for name in metrics}
        for arm in sorted(next(iter(scored.values()))):
            columns['model'].append(model.display_name)
            columns['arm'].append(arm)
            for name in metrics:
                columns[name].append(scored[name][arm])
    return _make_frame(columns, frame)


def expected_response(
    y_true: ArrayLike,
    recommended_treatment: ArrayLike,
    observed_treatment: ArrayLike,
    propensity: ArrayLike,
) -> float:
    """Expected Response of a recommended-treatment policy (Olaya, Coussement & Verbeke, 2020).

    A unit contributes its observed outcome only when the recommended arm matches the
    observed arm, inverse-weighted by the propensity of the observed arm so the estimate
    targets the full population (self-normalized IPW). Unlike PEHE it needs no
    ground-truth effect, so it is usable on real RCT / observational data.

    Args:
        y_true: Observed outcomes.
        recommended_treatment: Arm recommended per unit by the policy under evaluation
            (e.g. ``metrics.optimal_treatment_assignment(uplift)``).
        observed_treatment: Logged (taken) arm per unit.
        propensity: Behavior probability of the *observed* arm, ``P(T=observed|x)`` per unit.

    Returns:
        The estimated expected outcome under the recommended-treatment policy.
    """
    return snips(reward=y_true, action=observed_treatment, pscore=propensity, policy_action=recommended_treatment)
