"""Off-policy evaluation (OPE) estimators for scoring a learned policy offline.

Where `PolicyLearner` / `PolicyForest` *learn* a targeting policy, these estimators
*score* one against logged data without a new experiment. They are pure-numpy and
mirror the API names of the Open Bandit Pipeline (`st-tech/zr-obp`,
arXiv:1907.09623), which is the reference, not a dependency.

Conventions
-----------
- `reward` is the logged outcome `r_i` (1-D).
- `action` is the logged (taken) action `a_i`, an int in `{0..K-1}` (for uplift, the
  0/1 treatment).
- `pscore` is the behavior policy's probability of the *taken* action, `p(a_i | x_i)`.
  For a binary treatment with propensity `e(x) = P(T=1|x)` this is
  `a_i * e_i + (1 - a_i) * (1 - e_i)`.
- `policy_action` is the action the *evaluation* policy takes per unit (deterministic).
- `q_hat` (the regression-model rewards) has shape `[n, K]`: `q_hat[i, a]` estimates
  `E[r | x_i, a]`. Needed by the direct-method and doubly-robust families.

`evaluate_policy` wraps these to score a fitted `UpliftModel`'s induced
treat-if-uplift>threshold policy.
"""

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    'direct_method',
    'doubly_robust',
    'dr_os',
    'evaluate_policy',
    'ips',
    'snips',
    'switch_dr',
]


def _as_1d(arr: ArrayLike, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim > 1:
        raise ValueError(f'{name} must be 1-D; got shape {out.shape}.')
    return out.reshape(-1)


def _common(reward: ArrayLike, action: ArrayLike, pscore: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = _as_1d(reward, 'reward').astype(float)
    a = _as_1d(action, 'action').astype(int)
    p = _as_1d(pscore, 'pscore').astype(float)
    if not (len(r) == len(a) == len(p)):
        raise ValueError(f'reward ({len(r)}), action ({len(a)}) and pscore ({len(p)}) must share length.')
    if np.any(p <= 0):
        raise ValueError('pscore must be strictly positive (behavior policy probability of the taken action).')
    return r, a, p


def _indicator_weight(action: np.ndarray, policy_action: np.ndarray, pscore: np.ndarray) -> np.ndarray:
    return (action == policy_action).astype(float) / pscore


def _q_for_action(q_hat: np.ndarray, action: np.ndarray) -> np.ndarray:
    return q_hat[np.arange(len(action)), action]


def ips(reward: ArrayLike, action: ArrayLike, pscore: ArrayLike, policy_action: ArrayLike) -> float:
    """Inverse propensity score (IPW / IPS) estimate of the policy value."""
    r, a, p = _common(reward, action, pscore)
    pa = _as_1d(policy_action, 'policy_action').astype(int)
    return float(np.mean(_indicator_weight(a, pa, p) * r))


def snips(reward: ArrayLike, action: ArrayLike, pscore: ArrayLike, policy_action: ArrayLike) -> float:
    """Self-normalized IPS (lower-variance, bounded by the observed reward range)."""
    r, a, p = _common(reward, action, pscore)
    pa = _as_1d(policy_action, 'policy_action').astype(int)
    w = _indicator_weight(a, pa, p)
    denom = np.sum(w)
    if denom == 0:
        return 0.0
    return float(np.sum(w * r) / denom)


def direct_method(q_hat: ArrayLike, policy_action: ArrayLike) -> float:
    """Direct method: average the regression model's reward for the policy's action."""
    q = np.asarray(q_hat, dtype=float)
    pa = _as_1d(policy_action, 'policy_action').astype(int)
    return float(np.mean(q[np.arange(len(pa)), pa]))


def doubly_robust(
    reward: ArrayLike,
    action: ArrayLike,
    pscore: ArrayLike,
    policy_action: ArrayLike,
    q_hat: ArrayLike,
) -> float:
    """Doubly-robust estimate: direct method plus an IPS correction on the residual."""
    r, a, p = _common(reward, action, pscore)
    pa = _as_1d(policy_action, 'policy_action').astype(int)
    q = np.asarray(q_hat, dtype=float)
    direct = q[np.arange(len(pa)), pa]
    correction = _indicator_weight(a, pa, p) * (r - _q_for_action(q, a))
    return float(np.mean(direct + correction))


def switch_dr(
    reward: ArrayLike,
    action: ArrayLike,
    pscore: ArrayLike,
    policy_action: ArrayLike,
    q_hat: ArrayLike,
    *,
    tau: float = 10.0,
) -> float:
    """Switch-DR: drop the IPS correction where the importance weight exceeds `tau`.

    Large importance weights blow up DR's variance; Switch-DR falls back to the
    direct method for those units, trading a little bias for much less variance.
    """
    r, a, p = _common(reward, action, pscore)
    pa = _as_1d(policy_action, 'policy_action').astype(int)
    q = np.asarray(q_hat, dtype=float)
    w = _indicator_weight(a, pa, p)
    direct = q[np.arange(len(pa)), pa]
    keep = (w <= tau).astype(float)
    return float(np.mean(direct + keep * w * (r - _q_for_action(q, a))))


def dr_os(
    reward: ArrayLike,
    action: ArrayLike,
    pscore: ArrayLike,
    policy_action: ArrayLike,
    q_hat: ArrayLike,
    *,
    lambda_: float = 100.0,
) -> float:
    """DR with optimistic shrinkage (DRos): shrink the importance weights smoothly.

    Replaces `w` by `lambda_ / (w^2 + lambda_) * w`. `lambda_ -> inf` recovers
    doubly-robust; `lambda_ -> 0` recovers the direct method.
    """
    if lambda_ < 0:
        raise ValueError('lambda_ must be non-negative.')
    r, a, p = _common(reward, action, pscore)
    pa = _as_1d(policy_action, 'policy_action').astype(int)
    q = np.asarray(q_hat, dtype=float)
    w = _indicator_weight(a, pa, p)
    w_shrunk = (lambda_ / (w**2 + lambda_)) * w if lambda_ > 0 else np.zeros_like(w)
    direct = q[np.arange(len(pa)), pa]
    return float(np.mean(direct + w_shrunk * (r - _q_for_action(q, a))))


_DIRECT_FREE = {'ips', 'snips'}
_ESTIMATORS = {'ips', 'snips', 'dm', 'dr', 'switch_dr', 'dr_os'}


def evaluate_policy(
    model,
    X: ArrayLike,
    treatment: ArrayLike,
    y: ArrayLike,
    propensity: ArrayLike,
    *,
    estimator: str = 'dr',
    threshold: float = 0.0,
    **estimator_kwargs,
) -> float:
    """Score a fitted `UpliftModel`'s treat-if-uplift>threshold policy via OPE.

    Args:
        model: A fitted `UpliftModel`. The induced policy treats a unit when its
            predicted uplift exceeds `threshold`.
        X: Features to score.
        treatment: Logged 0/1 treatment (the taken action).
        y: Logged outcome (reward).
        propensity: Behavior propensity `e(x) = P(T=1|x)` per unit.
        estimator: One of `'ips'`, `'snips'`, `'dm'`, `'dr'`, `'switch_dr'`, `'dr_os'`.
        threshold: Uplift cutoff for the induced policy.
        **estimator_kwargs: Passed to the chosen estimator (e.g. `tau`, `lambda_`).

    Returns:
        The estimated value of the induced policy.

    Raises:
        ValueError: For an unknown estimator.
    """
    if estimator not in _ESTIMATORS:
        raise ValueError(f'estimator must be one of {sorted(_ESTIMATORS)}; got {estimator!r}.')

    t = _as_1d(treatment, 'treatment').astype(int)
    e = _as_1d(propensity, 'propensity').astype(float)
    pscore = t * e + (1 - t) * (1.0 - e)

    if estimator in _DIRECT_FREE:
        uplift = np.asarray(model.predict(X)).reshape(-1)
        policy_action = (uplift > threshold).astype(int)
        fn = ips if estimator == 'ips' else snips
        return fn(y, t, pscore, policy_action)

    uplift, y0, y1 = model.predict(X, return_components=True)
    uplift = np.asarray(uplift).reshape(-1)
    policy_action = (uplift > threshold).astype(int)
    q_hat = np.column_stack([np.asarray(y0).reshape(-1), np.asarray(y1).reshape(-1)])

    if estimator == 'dm':
        return direct_method(q_hat, policy_action)
    if estimator == 'dr':
        return doubly_robust(y, t, pscore, policy_action, q_hat)
    if estimator == 'switch_dr':
        return switch_dr(y, t, pscore, policy_action, q_hat, **estimator_kwargs)
    return dr_os(y, t, pscore, policy_action, q_hat, **estimator_kwargs)
