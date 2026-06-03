import numpy as np
import pytest

from uplift_forecast.models import CausalForest


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, p = 400, 5
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    y = x[:, 0] + treatment * np.clip(x[:, 1], 0, None) + rng.normal(size=n)
    return x, treatment, y


def test_predict_interval_brackets_point(data):
    x, treatment, y = data
    forest = CausalForest(n_estimators=20, random_state=0).fit(x, treatment, y)
    lower, upper = forest.predict_interval(x, alpha=0.1)
    tau = forest.predict(x)
    assert lower.shape == upper.shape == (len(y),)
    assert (lower <= tau + 1e-9).all()
    assert (tau <= upper + 1e-9).all()


def test_predict_interval_bad_alpha(data):
    x, treatment, y = data
    forest = CausalForest(n_estimators=10, random_state=0).fit(x, treatment, y)
    with pytest.raises(ValueError):
        forest.predict_interval(x, alpha=1.5)
