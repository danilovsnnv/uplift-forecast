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

## Installation

```bash
pip install -e .                              # core
pip install -e ".[catboost]"                  # + CatBoost for meta-learners
pip install -e ".[polars]"                    # + polars DataFrame support
pip install -e ".[auto]"                      # + Optuna hyperparameter search
pip install -e ".[docs]"                      # + MkDocs Material for this site
```

Runtime dependencies: `torch`, `pytorch-lightning`, `numpy`, `pandas`, `scikit-learn`.
Python 3.10 or later is required.

## Quick start

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast
from uplift_forecast.models import TLearner, SLearner
from uplift_forecast.metrics import auuc_score, qini_score

uf = UpliftForecast(models=[
    SLearner(GradientBoostingRegressor(random_state=0), alias='slearner'),
    TLearner(GradientBoostingRegressor(random_state=0), alias='tlearner'),
])
uf.fit(X_train, treatment_train, y_train)
preds = uf.predict(X_test)            # one 'uplift_<model>' column per model
```

## Next steps

- **[Targeting under a budget with PolicyForest](tutorials/targeting-under-a-budget.md)** — an
  end-to-end walkthrough of training a policy model and assigning treatment under a budget.
- **API reference** — [Models](api/models.md), [Matching](api/matching.md),
  [Metrics](api/metrics.md), [Auto](api/auto.md), and [Frameworks](api/frameworks.md).
