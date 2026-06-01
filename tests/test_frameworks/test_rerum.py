import numpy as np
import pytest
import torch

from uplift_forecast import RERUM, UpliftForecast
from uplift_forecast.common._base_neural import BaseNeuralUpliftModel
from uplift_forecast.losses import RERUMLoss
from uplift_forecast.models import CFRNet, DragonNet, SLearner, TARNet


def _model(model_cls, trainer_kwargs) -> BaseNeuralUpliftModel:
    return model_cls(input_size=6, hidden_size=16, batch_size=64, learning_rate=1e-3, **trainer_kwargs)


@pytest.mark.parametrize('model_cls', [DragonNet, CFRNet, TARNet])
def test_fit_predict_smoke(model_cls, uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    framework = RERUM(
        model=_model(model_cls, trainer_kwargs),
        within_ranking_weight=1e-3,
        cross_ranking_weight=1e-3,
        listwise_ranking_weight=1.0,
    )
    framework.fit(x, treatment, y)

    uplift = framework.predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()
    assert isinstance(framework.model.loss, RERUMLoss)


@pytest.mark.parametrize('model_cls', [DragonNet, CFRNet, TARNet])
def test_predict_components(model_cls, uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    framework = RERUM(model=_model(model_cls, trainer_kwargs))
    framework.fit(x, treatment, y)

    uplift, y0, y1 = framework.predict(x, return_components=True)
    assert y0.shape == y1.shape == uplift.shape == (x.shape[0],)
    assert np.allclose(uplift, y1 - y0)


def test_integration_with_uplift_forecast(uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    framework = RERUM(model=_model(DragonNet, trainer_kwargs), alias='rerum_dn')
    forecast = UpliftForecast(models=[framework])
    forecast.fit(x, treatment, y)

    preds = forecast.predict(x)
    assert 'uplift_rerum_dn' in preds.columns


def test_cfrnet_switches_to_ziln_heads(trainer_kwargs):
    model = _model(CFRNet, trainer_kwargs)
    assert model._outcome_size == 1
    RERUM(model=model)
    assert model._outcome_size == 3
    out, _, _ = model(torch.randn(4, 6), torch.ones(4, 1))
    assert out.shape == (4, 3)


def test_normalisation_disabled_for_ziln(trainer_kwargs):
    model = _model(DragonNet, trainer_kwargs)
    RERUM(model=model)
    assert model.normalize_y is False


def test_rejects_incompatible_model():
    with pytest.raises((TypeError, ValueError), match='RERUM'):
        RERUM(model=SLearner(model=None))
