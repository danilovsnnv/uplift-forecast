# Targeting under a budget with PolicyForest

Most real interventions are constrained: you can only mail a coupon to a fraction of customers,
or treat a limited number of patients. The question is not just *who has a positive treatment
effect?* but *which subset should we treat when we can only treat so many?* This tutorial trains
a [`PolicyForest`](../api/models.md) on synthetic data and uses it to assign treatment under a
budget.

`PolicyForest` is a forest of policy trees grown on a doubly-robust gain score. Unlike a CATE
estimator, its `predict` returns a **rankable gain score** rather than a calibrated effect — which
is exactly what you need to decide whom to treat first.

## A synthetic dataset with a known effect

We reuse the data-generating process from
[`examples/quickstart.py`](https://github.com/danilovsnnv/uplift-forecast/blob/main/examples/quickstart.py):
a continuous outcome whose treatment effect grows with feature 1, so the true individual uplift is
`tau(x) = 3 * relu(x[:, 1])`. Only units with a large positive feature 1 truly benefit, so a good
policy should concentrate the budget on them.

```python
import numpy as np

from uplift_forecast.models import PolicyForest


def make_dataset(n=6000, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features))
    treatment = rng.integers(0, 2, size=n).astype(float)
    baseline = 2.0 * x[:, 0] + x[:, 2]
    tau = 3.0 * np.clip(x[:, 1], 0.0, None)          # true individual uplift
    y = baseline + treatment * tau + rng.normal(size=n)
    return x, treatment, y, tau


x, treatment, y, tau = make_dataset()
cut = int(0.7 * len(x))
x_tr, t_tr, y_tr = x[:cut], treatment[:cut], y[:cut]
x_te, t_te, y_te = x[cut:], treatment[cut:], y[cut:]
```

## Fit the policy model

```python
model = PolicyForest(n_estimators=100, max_depth=4, n_folds=5, random_state=0)
model.fit(x_tr, t_tr, y_tr)

gain = model.predict(x_te)        # per-unit gain score (higher = treat sooner)
```

## Assign treatment under a budget

`assign` turns gain scores into 0/1 decisions. With no constraint it treats everyone with a
positive gain; pass `budget` (a fraction of the population) or `top_k` (an exact count) to respect
a capacity limit.

```python
treat_all = model.assign(x_te)              # everyone with gain > 0
budget_20 = model.assign(x_te, budget=0.2)  # treat the top 20%
top_100   = model.assign(x_te, top_k=100)   # treat exactly 100 units

print(budget_20.sum(), 'of', len(x_te), 'units treated under a 20% budget')
```

Because the gain score *ranks* units, the budgeted policy automatically picks the highest-value
units first — here, those with the largest positive feature 1.

## Evaluate the policy

`policy_value` gives an inverse-propensity estimate of the expected outcome under a policy, so you
can compare a targeted policy against treating everyone or no one:

```python
v_budget   = model.policy_value(x_te, t_te, y_te, policy=budget_20)
v_treat_all = model.policy_value(x_te, t_te, y_te, policy=np.ones(len(x_te), dtype=int))
v_treat_none = model.policy_value(x_te, t_te, y_te, policy=np.zeros(len(x_te), dtype=int))

print(f'budget-20%: {v_budget:.3f}   treat-all: {v_treat_all:.3f}   treat-none: {v_treat_none:.3f}')
```

A good targeted policy reaches much of the value of treating everyone while spending only a
fraction of the budget — and clearly beats treating no one. From here you can sweep the budget
fraction to trace a cost/value curve, or compare `PolicyForest` against a
[`PolicyLearner`](../api/models.md) wrapped around any CATE estimator.
