__all__ = ['SurvivalUplift', 'ipcw_survival_pseudo_outcome']


from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..common._uplift_model import UpliftModel, _to_array, _to_numpy_1d


def _split_time_event(y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split a 2-column ``y`` into ``(time, event)`` (event: 1 = event, 0 = censored)."""
    arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError('SurvivalUplift expects y as a 2-column [time, event] array.')
    return arr[:, 0].astype(float), arr[:, 1].astype(int)


def _km_censoring_at(time: np.ndarray, event: np.ndarray, horizon: float) -> float:
    """Kaplan-Meier estimate of the censoring survival ``G(horizon) = P(C > horizon)``.

    Censoring is treated as the event of interest (``event == 0``). Assumes
    censoring independent of covariates (random censoring).
    """
    cens_times = np.unique(time[(event == 0) & (time <= horizon)])
    g = 1.0
    for c in cens_times:
        n_at_risk = int(np.sum(time >= c))
        d = int(np.sum((time == c) & (event == 0)))
        if n_at_risk > 0:
            g *= 1.0 - d / n_at_risk
    return g


def ipcw_survival_pseudo_outcome(
    time: ArrayLike,
    event: ArrayLike,
    horizon: float,
    *,
    censoring_clip: float = 1e-3,
) -> np.ndarray:
    """IPCW pseudo-outcome for the survival probability ``S(h) = P(T > h)``.

    Returns ``I(X_i > h) / G(h)`` where ``X = min(T, C)`` is the observed time and
    ``G`` is the Kaplan-Meier censoring survival. Its conditional mean equals
    ``S(h | x)`` under independent censoring, so regressing it per arm yields the
    survival-probability uplift ``S(h | t=1) - S(h | t=0)``.

    Args:
        time: Observed time ``min(T, C)``.
        event: Event indicator (1 = event of interest, 0 = censored).
        horizon: Horizon ``h`` at which to evaluate survival.
        censoring_clip: Lower bound on ``G(h)`` to avoid blow-up.

    Returns:
        The pseudo-outcome array.
    """
    t = _to_numpy_1d(time).astype(float)
    ev = _to_numpy_1d(event).astype(int)
    g = max(_km_censoring_at(t, ev, horizon), censoring_clip)
    return (t > horizon).astype(float) / g


class SurvivalUplift(UpliftModel):
    """Survival/retention uplift via IPCW reduction to a standard uplift model.

    Transforms right-censored ``(time, event)`` outcomes into an inverse-probability-
    of-censoring-weighted (IPCW) pseudo-outcome for the survival probability at a
    horizon, then fits any base ``UpliftModel`` on it. The resulting uplift is the
    treatment effect on ``S(h) = P(T > h)`` — the standard target for churn-time /
    retention uplift (cf. causal survival forests, Cui et al., arXiv:2001.09887).

    ``y`` is passed as a 2-column ``[time, event]`` array (event: 1 = event,
    0 = censored), so it slots into ``UpliftForecast`` when called the same way.

    Args:
        model: Base ``UpliftModel`` fitted on the pseudo-outcome (e.g. a meta-learner
            or ``CausalForest``).
        horizon: Horizon ``h`` for the survival probability ``S(h)``.
        censoring_clip: Lower bound on the censoring survival ``G(h)``.
        alias: Optional display name for UpliftForecast output columns.
    """

    def __init__(self, model: UpliftModel, horizon: float, *, censoring_clip: float = 1e-3, alias: str | None = None):
        if not isinstance(model, UpliftModel):
            raise TypeError('SurvivalUplift.model must be an UpliftModel.')
        self.model = model
        self.horizon = horizon
        self.censoring_clip = censoring_clip
        self.alias = alias
        self._fitted = False

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **fit_params: Any,
    ) -> 'SurvivalUplift':
        """Fit on (X, treatment, [time, event]); transforms y to the IPCW pseudo-outcome."""
        time, event = _split_time_event(y)
        pseudo = ipcw_survival_pseudo_outcome(time, event, self.horizon, censoring_clip=self.censoring_clip)

        eval_arg = None
        if eval_set is not None:
            x_val, t_val, y_val = eval_set
            v_time, v_event = _split_time_event(y_val)
            v_pseudo = ipcw_survival_pseudo_outcome(v_time, v_event, self.horizon, censoring_clip=self.censoring_clip)
            eval_arg = (x_val, t_val, v_pseudo)

        self.model.fit(_to_array(X), treatment, pseudo, eval_set=eval_arg, **fit_params)
        self._fitted = True
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the survival-probability uplift ``S(h | t=1) - S(h | t=0)``."""
        if not self._fitted:
            raise RuntimeError('SurvivalUplift has not been fitted yet. Call .fit() first.')
        return self.model.predict(X, return_components=return_components)
