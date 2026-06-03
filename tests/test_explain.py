import importlib.util

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import explain
from uplift_forecast.models import CausalForest, TLearner


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, p = 400, 5
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    y = x[:, 0] + treatment * np.clip(x[:, 1], 0, None) + rng.normal(size=n)
    return x, treatment, y


def test_permutation_importance(data):
    x, treatment, y = data
    model = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    importance = explain.permutation_importance(model, x, treatment, y, metric='auuc', n_repeats=2)
    assert importance.shape[0] == x.shape[1]
    assert list(importance.columns) == ['importance_mean', 'importance_std']


def test_permutation_importance_bad_metric(data):
    x, treatment, y = data
    model = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    with pytest.raises(ValueError):
        explain.permutation_importance(model, x, treatment, y, metric='mse')


def test_tree_feature_importance(data):
    x, treatment, y = data
    forest = CausalForest(n_estimators=15, random_state=0).fit(x, treatment, y)
    importance = explain.tree_feature_importance(forest)
    assert len(importance) > 0
    assert np.isfinite(importance.to_numpy()).all()


def test_tree_feature_importance_requires_trees(data):
    x, treatment, y = data
    model = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    with pytest.raises(AttributeError):
        explain.tree_feature_importance(model)


@pytest.mark.skipif(importlib.util.find_spec('shap') is None, reason='shap not installed')
def test_uplift_shap_values(data):
    x, treatment, y = data
    model = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    values = explain.uplift_shap_values(model, x[:20], background=x[:50])
    assert values.values.shape[0] == 20
