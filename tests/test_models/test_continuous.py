import numpy as np
import pytest

from uplift_forecast.metrics import best_dose, dose_response_mise
from uplift_forecast.models import DRNet, VCNet


@pytest.fixture
def dose_data():
    rng = np.random.default_rng(0)
    n, p = 300, 6
    x = rng.normal(size=(n, p)).astype('float32')
    dose = rng.uniform(0, 1, size=n).astype('float32')
    y = (x[:, 0] + dose * x[:, 1] + rng.normal(size=n)).astype('float32')
    return x, dose, y


@pytest.mark.parametrize('cls', [VCNet, DRNet])
def test_predict_uplift_vs_reference(cls, dose_data, trainer_kwargs):
    x, dose, y = dose_data
    model = cls(input_size=6, hidden_size=16, batch_size=128, **trainer_kwargs).fit(x, dose, y)
    uplift = model.predict(x)
    assert uplift.shape == (300,)
    assert np.isfinite(uplift).all()


@pytest.mark.parametrize('cls', [VCNet, DRNet])
def test_predict_dose_response(cls, dose_data, trainer_kwargs):
    x, dose, y = dose_data
    model = cls(input_size=6, hidden_size=16, batch_size=128, **trainer_kwargs).fit(x, dose, y)
    grid = np.linspace(0, 1, 5)
    curves = model.predict_dose_response(x, grid)
    assert curves.shape == (300, 5)
    assert np.isfinite(curves).all()


def test_continuous_metrics(dose_data, trainer_kwargs):
    x, dose, y = dose_data
    grid = np.linspace(0, 1, 5)
    curves = VCNet(input_size=6, hidden_size=16, batch_size=128, **trainer_kwargs).fit(
        x, dose, y,
    ).predict_dose_response(x, grid)
    assert np.isfinite(dose_response_mise(np.zeros_like(curves), curves, grid))
    assert best_dose(curves, grid).shape == (300,)


def test_mise_shape_mismatch():
    with pytest.raises(ValueError):
        dose_response_mise(np.zeros((3, 4)), np.zeros((3, 5)), np.linspace(0, 1, 4))
