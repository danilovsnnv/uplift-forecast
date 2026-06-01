# uplift-forecast

A unified Python library for uplift modelling: estimate causal treatment effects, rank the
individuals most likely to respond, and evaluate decision policies — with a single consistent API
across neural networks, meta-learners, and matching methods.

---

Uplift modelling answers the question: *who benefits from an intervention?* Rather than predicting
an outcome in isolation, it estimates the incremental effect of a treatment (a promotion, a drug,
a policy change) on each individual. This is useful whenever you want to target the people for
whom an action actually makes a difference, not just anyone with a high predicted outcome.

`uplift-forecast` provides a collection of models (neural and sklearn-style), covariate matching
tools, evaluation metrics, and optional hyperparameter search under one lightweight API. It accepts
NumPy arrays or DataFrames and wraps any sklearn-compatible estimator you already have.

---

## Features

- **Neural models** — DragonNet, CFRNet, TARNet; PyTorch + PyTorch Lightning; ZILN revenue
  objectives and IPM representation-balancing included.
- **Meta-learners** — SLearner, TLearner, XLearner, RLearner, DRLearner, ZLearner; wrap any
  sklearn-style regressor; no GPU required.
- **Forest models** — CausalForest (doubly-robust honest trees with variance estimation) and
  PolicyForest (optimal treatment-policy trees).
- **Policy learning** — PolicyLearner wraps any CATE estimator and trains a classifier for
  treatment-assignment decisions, with budget and top-k constraints.
- **Matching** — seven covariate matchers (propensity score, Mahalanobis, CEM, kernel,
  embedding, and more) with balance diagnostics.
- **Metrics** — AUUC, Qini score, cumulative-gain and Qini curves, per-arm MAE/MSE/MAPE.
- **Hyperparameter search** — declarative Optuna integration (optional dependency).
- **RERUM framework** — wraps any neural model with a rank-enhanced ZILN revenue loss
  (He et al., KDD 2024).

---

## Installation

The package is not on PyPI yet. Install directly from the repository:

```bash
pip install -e .                              # core
pip install -e ".[catboost]"                  # + CatBoost for meta-learners
pip install -e ".[polars]"                    # + polars DataFrame support
pip install -e ".[auto]"                      # + Optuna hyperparameter search
pip install -e ".[catboost,polars,auto,dev]"  # everything, including dev tooling
```

Runtime dependencies: `torch`, `pytorch-lightning`, `numpy`, `pandas`, `scikit-learn`.
Python 3.12 or later is required.

---

## Quick start

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from uplift_forecast import UpliftForecast
from uplift_forecast.models import TLearner, RLearner
from uplift_forecast.metrics import auuc_score, qini_score

rng = np.random.default_rng(0)
n = 2000
X = rng.standard_normal((n, 10))
treatment = rng.integers(0, 2, size=n)
y = 2.0 * treatment * (X[:, 0] > 0) + rng.standard_normal(n) * 0.5

X_tr, X_te = X[:1600], X[1600:]
t_tr, t_te = treatment[:1600], treatment[1600:]
y_tr, y_te = y[:1600], y[1600:]

uf = UpliftForecast(models=[
    TLearner(model=GradientBoostingRegressor(n_estimators=100)),
    RLearner(
        outcome_model=GradientBoostingRegressor(),
        effect_model=GradientBoostingRegressor(),
    ),
])
uf.fit(X_tr, t_tr, y_tr)

preds = uf.predict(X_te)
# DataFrame with columns: 'uplift_TLearner', 'uplift_RLearner'

for col in preds.columns:
    name = col.replace('uplift_', '')
    uplift = preds[col].values
    print(f"{name}  AUUC={auuc_score(y_te, uplift, t_te):.4f}"
          f"  Qini={qini_score(y_te, uplift, t_te):.4f}")
```

Return per-arm components alongside uplift:

```python
preds = uf.predict(X_te, return_components=True)
# adds columns: 'TLearner_y0_pred', 'TLearner_y1_pred', etc.
```

---

## Components

### Models

**Neural models** (require PyTorch; subclass `BaseNeuralUpliftModel`):

| Class | Description |
|---|---|
| `DragonNet` | Shared representation + two outcome heads + propensity head; DragonNetLoss with targeted regularisation. |
| `CFRNet` | Counterfactual regression; penalises representation imbalance between arms via an IPM term. |
| `TARNet` | Treatment-agnostic representation network; CFRNet without the IPM penalty. |

All three require `input_size`. Training hyperparameters (`learning_rate`, `batch_size`,
`scaler_type`, `random_seed`, `**trainer_kwargs`) are forwarded to the Lightning `Trainer`.

```python
from uplift_forecast.models import DragonNet

model = DragonNet(input_size=20, hidden_size=200, max_epochs=50, gradient_clip_val=1.0)
model.fit(X_tr, t_tr, y_tr)
uplift = model.predict(X_te)
```

**Meta-learners** (sklearn-style; subclass `BaseMetaUpliftModel`):

| Class | Description |
|---|---|
| `SLearner` | Single model on `[treatment, X]`; contrasts predictions at t=0 vs t=1. |
| `TLearner` | One model per arm; uplift = treated\_model(X) − control\_model(X). |
| `XLearner` | Imputes individual effects per arm, fits effect models per arm, combines via propensity weighting. |
| `RLearner` | Robinson residualisation; cross-fits outcome and propensity nuisances, then regresses the R-loss residual. |
| `DRLearner` | Doubly-robust (AIPW) pseudo-outcome regression; consistent if either nuisance is correctly specified. |
| `ZLearner` | Transformed-outcome (IPW) regression on the Horvitz-Thompson estimator. |
| `CausalForest` | Honest ensemble of trees on a doubly-robust pseudo-outcome; also exposes `predict_variance(X)`. |
| `PolicyForest` | Honest ensemble for the optimal treatment action; exposes `assign(X)` and `policy_value(X, t, y)`. |
| `PolicyLearner` | Wraps any CATE estimator, converts effect estimates into policy labels, trains an assignment classifier. |

```python
from uplift_forecast.models import XLearner, CausalForest
from sklearn.ensemble import RandomForestRegressor

xl = XLearner(model=RandomForestRegressor(), n_folds=5)
cf = CausalForest(n_estimators=200, honest=True, pseudo_outcome='dr')
```

PolicyForest and PolicyLearner support budget-constrained assignment:

```python
from uplift_forecast.models import PolicyForest

pf = PolicyForest(n_estimators=100).fit(X_tr, t_tr, y_tr)
assignments = pf.assign(X_te, budget=0.3)    # treat at most 30 % of the population
value = pf.policy_value(X_te, t_te, y_te)
```

---

### Matching

`uplift_forecast.matching` provides seven covariate matchers with a sklearn-style
`fit` / `transform` / `fit_transform` API. Each returns a weighted DataFrame of matched units
suitable for downstream analysis or re-weighting.

| Class | Method |
|---|---|
| `PropensityScoreMatcher` | Nearest-neighbour matching on a fitted propensity score. |
| `MahalanobisMatcher` | Nearest-neighbour matching using Mahalanobis distance. |
| `MahalanobisPSCaliperMatcher` | Two-stage: restrict to a propensity-score caliper, then match by Mahalanobis distance. |
| `NearestNeighborMatcher` | Flexible NN matching with configurable metric and a pluggable backend for approximate search. |
| `KernelMatcher` | Soft matching: weights each control by a kernel of its distance to the treated unit. |
| `CoarsenedExactMatcher` | CEM: bins features into strata and matches exactly within strata; returns ATT weights. |
| `EmbeddingMatcher` | Matches in a learned or pre-computed representation space via any encoder. |

Three diagnostics help assess balance before and after matching:

```python
from uplift_forecast.matching import (
    PropensityScoreMatcher,
    covariate_balance,
    match_rate,
)
from sklearn.linear_model import LogisticRegression

matcher = PropensityScoreMatcher(model=LogisticRegression(), caliper=0.05)
matched_df = matcher.fit_transform(X, treatment)

print(match_rate(treatment, matched_df['treatment']))
print(covariate_balance(X, treatment, weight=matched_df['weight']))
```

---

### Metrics and evaluation

```python
from uplift_forecast.metrics import (
    auuc_score,
    qini_score,
    cumulative_gain_curve,
    qini_curve,
)

auuc = auuc_score(y_true, uplift, treatment)
qini = qini_score(y_true, uplift, treatment)

x, gain = cumulative_gain_curve(y_true, uplift, treatment)
x, qini_vals = qini_curve(y_true, uplift, treatment)
```

---

### Hyperparameter search

`uplift_forecast.auto` provides declarative Optuna integration (requires the `auto` extra):

```python
from uplift_forecast.auto import OptunaHyperparameterTuner, SuggestInt, SuggestFloat
from uplift_forecast.models import CausalForest

tuner = OptunaHyperparameterTuner(
    model_cls=CausalForest,
    param_space={
        'n_estimators': SuggestInt(50, 500),
        'min_samples_leaf': SuggestInt(1, 20),
    },
    X=X_tr, treatment=t_tr, y=y_tr,
    n_trials=50,
)
tuner.optimize()
print(tuner.best_params)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests -q
ruff check uplift_forecast/
```

A runnable end-to-end example lives in [`examples/rerum_quickstart.py`](examples/rerum_quickstart.py):

```bash
PYTHONPATH=. python examples/rerum_quickstart.py
```
