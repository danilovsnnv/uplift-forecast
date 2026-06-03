import numpy as np
import pytest
import torch

from uplift_forecast import UpliftForecast
from uplift_forecast.losses import DragonNetLoss
from uplift_forecast.models import DESCN, EFIN, EUEN, FlexTENet, TwoStageUplift

MODELS = [EFIN, DESCN, FlexTENet, EUEN, TwoStageUplift]


def _model(cls, trainer_kwargs):
    return cls(input_size=6, hidden_size=16, batch_size=64, learning_rate=1e-3, **trainer_kwargs)


@pytest.mark.parametrize('cls', MODELS)
def test_fit_predict_smoke(cls, uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    uplift = _model(cls, trainer_kwargs).fit(x, treatment, y).predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()


@pytest.mark.parametrize('cls', MODELS)
def test_predict_components(cls, uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    uplift, y0, y1 = _model(cls, trainer_kwargs).fit(x, treatment, y).predict(x, return_components=True)
    assert y0.shape == y1.shape == uplift.shape == (x.shape[0],)
    assert np.allclose(uplift, y1 - y0, atol=1e-4)


@pytest.mark.parametrize('cls', MODELS)
def test_integration_with_uplift_forecast(cls, uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_model(cls, trainer_kwargs)]).fit(x, treatment, y)
    assert f'uplift_{cls.__name__}' in forecast.predict(x).columns


def test_point_heads_by_default(trainer_kwargs):
    model = EUEN(input_size=6, hidden_size=16, batch_size=64, **trainer_kwargs)
    assert model._outcome_size == 1
    out0, out1 = model(torch.randn(4, 6))
    assert out0.shape == out1.shape == (4, 1)


def test_loss_retargets_head_width(trainer_kwargs):
    # The loss declares the head width; switching to a ZILN loss must resize heads.
    model = EFIN(input_size=6, hidden_size=16, batch_size=64, **trainer_kwargs)
    model.set_loss(DragonNetLoss())
    assert model._outcome_size == 3
    y0, y1, t_pred = model(torch.randn(4, 6))
    assert y0.shape == y1.shape == (4, 3)
