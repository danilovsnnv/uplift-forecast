"""Explainability for uplift models.

Two model-agnostic tools plus a tree-model helper:

- `uplift_shap_values` — SHAP on the uplift output directly (lazy `shap` import,
  mirroring `auto.py`'s lazy optuna; SHAP stays an optional extra).
- `permutation_importance` — drop in a *causal* metric (AUUC/Qini) when each
  feature is shuffled, so importance is keyed to the uplift signal, not outcome MSE.
- `tree_feature_importance` — surface native `feature_importances_` for the
  forest models (`CausalForest`, `PolicyForest`).

`shap` is loaded lazily so importing `uplift_forecast` never requires it.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .metrics import auuc_score, qini_score

__all__ = [
    'permutation_importance',
    'tree_feature_importance',
    'uplift_shap_values',
]

_METRICS = {'auuc': auuc_score, 'qini': qini_score}


def _require_shap():
    try:
        import shap
    except ImportError as err:
        raise ImportError(
            'shap is required for uplift_forecast.explain.uplift_shap_values. '
            'Install with `pip install shap`.',
        ) from err
    return shap


def uplift_shap_values(model, X: ArrayLike, *, background: ArrayLike | None = None):
    """SHAP values explaining a model's predicted uplift.

    Explains `model.predict(X)` (the uplift itself), so attributions answer
    "which features drive the treatment effect", not the outcome level.

    Args:
        model: A fitted `UpliftModel`.
        X: Features to explain.
        background: Optional background dataset for the explainer; defaults to `X`.

    Returns:
        A `shap.Explanation` over the uplift output.
    """
    shap = _require_shap()

    def predict_uplift(data: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(data)).reshape(-1)

    explainer = shap.Explainer(predict_uplift, X if background is None else background)
    return explainer(X)


def _feature_view(X: ArrayLike) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(), list(X.columns)
    arr = np.asarray(X)
    return arr, [f'feature_{i}' for i in range(arr.shape[1])]


def permutation_importance(
    model,
    X: ArrayLike,
    treatment: ArrayLike,
    y: ArrayLike,
    *,
    metric: str = 'auuc',
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Permutation importance keyed on a causal metric (AUUC or Qini).

    For each feature, the score drop when its column is shuffled measures how much
    that feature contributes to ranking units by uplift. Using AUUC/Qini (not
    outcome error) keeps the importance aligned with the causal target.

    Args:
        model: A fitted `UpliftModel`.
        X: Features.
        treatment: 0/1 treatment vector.
        y: Observed outcome.
        metric: `'auuc'` or `'qini'`.
        n_repeats: Number of shuffles averaged per feature.
        random_state: Seed for the shuffling RNG.

    Returns:
        DataFrame indexed by feature with columns `importance_mean` and
        `importance_std`, sorted by `importance_mean` descending.

    Raises:
        ValueError: For an unknown metric.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {sorted(_METRICS)}; got {metric!r}.")
    score_fn = _METRICS[metric]
    x, names = _feature_view(X)
    t = np.asarray(treatment).reshape(-1)
    y_arr = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(random_state)

    baseline = score_fn(y_arr, np.asarray(model.predict(x)).reshape(-1), t)
    means, stds = [], []
    for j in range(x.shape[1]):
        drops = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            x_perm = x.copy()
            x_perm[:, j] = x[rng.permutation(len(x)), j]
            shuffled = score_fn(y_arr, np.asarray(model.predict(x_perm)).reshape(-1), t)
            drops[r] = baseline - shuffled
        means.append(float(drops.mean()))
        stds.append(float(drops.std()))
    return pd.DataFrame(
        {'importance_mean': means, 'importance_std': stds},
        index=pd.Index(names, name='feature'),
    ).sort_values('importance_mean', ascending=False)


def tree_feature_importance(model, feature_names: list[str] | None = None) -> pd.Series:
    """Native impurity-based feature importance for the forest uplift models.

    Averages the per-tree `feature_importances_` of `CausalForest` / `PolicyForest`
    over their internal trees (each tree sees a random feature subset, so the
    importance is accumulated by original feature index).

    Args:
        model: A fitted `CausalForest` or `PolicyForest`.
        feature_names: Optional names for the index.

    Returns:
        Series of importances indexed by feature, sorted descending.

    Raises:
        AttributeError: If the model exposes no internal trees.
    """
    trees = getattr(model, 'causal_trees', None) or getattr(model, 'policy_trees', None)
    if not trees:
        raise AttributeError(
            f'{type(model).__name__} exposes no internal trees for feature importance.',
        )
    fitted = [entry for entry in trees if entry.get('tree') is not None]
    if not fitted:
        raise AttributeError(f'{type(model).__name__} has no fitted trees to score.')
    n_features = 1 + max(int(entry['features'].max()) for entry in fitted)
    importance = np.zeros(n_features, dtype=float)
    for entry in fitted:
        feat = entry['features']
        importance[feat] += entry['tree'].feature_importances_
    importance /= len(fitted)
    names = feature_names or [f'feature_{i}' for i in range(n_features)]
    return pd.Series(importance, index=pd.Index(names, name='feature')).sort_values(ascending=False)
