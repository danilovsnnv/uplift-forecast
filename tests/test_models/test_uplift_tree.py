import numpy as np
import pytest

from uplift_forecast.models import UpliftTree


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n, p = 600, 6
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    rate = 0.3 + 0.2 * treatment + 0.1 * (x[:, 0] > 0)
    y = (rng.random(n) < rate).astype(float)
    return x, treatment, y


@pytest.mark.parametrize('criterion', ['kl', 'ed', 'chi'])
def test_fit_predict(criterion, binary_data):
    x, treatment, y = binary_data
    model = UpliftTree(
        max_depth=3, min_samples_leaf=50, min_samples_treatment=5, criterion=criterion,
    ).fit(x, treatment, y)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert uplift.shape == (len(y),)
    assert np.isfinite(uplift).all()
    assert ((y0 >= 0) & (y0 <= 1)).all()
    assert ((y1 >= 0) & (y1 <= 1)).all()
    assert np.allclose(uplift, y1 - y0)


def test_bad_criterion():
    with pytest.raises(ValueError):
        UpliftTree(criterion='nope')


def test_requires_both_arms(binary_data):
    x, _, y = binary_data
    with pytest.raises(ValueError):
        UpliftTree().fit(x, np.ones(len(y), dtype=int), y)
