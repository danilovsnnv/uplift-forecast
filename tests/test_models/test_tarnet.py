import numpy as np
import torch

from uplift_forecast import UpliftForecast
from uplift_forecast.models import TARNet


def _model(trainer_kwargs) -> TARNet:
    return TARNet(input_size=6, hidden_size=16, batch_size=64, learning_rate=1e-3, **trainer_kwargs)


def test_fit_predict_smoke(uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    model = _model(trainer_kwargs).fit(x, treatment, y)

    uplift = model.predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()


def test_predict_components(uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    model = _model(trainer_kwargs).fit(x, treatment, y)

    uplift, y0, y1 = model.predict(x, return_components=True)
    assert y0.shape == y1.shape == uplift.shape == (x.shape[0],)
    assert np.allclose(uplift, y1 - y0)


def test_scalar_heads_by_default(trainer_kwargs):
    model = _model(trainer_kwargs)
    assert model._outcome_size == 1
    out0, out1 = model(torch.randn(4, 6))
    assert out0.shape == (4, 1)
    assert out1.shape == (4, 1)


def test_integration_with_uplift_forecast(uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_model(trainer_kwargs)])
    forecast.fit(x, treatment, y)
    assert 'uplift_TARNet' in forecast.predict(x).columns
