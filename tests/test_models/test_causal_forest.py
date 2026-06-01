import numpy as np

from uplift_forecast import UpliftForecast
from uplift_forecast.metrics import auuc_score
from uplift_forecast.models import CausalForest


def _model(**kwargs) -> CausalForest:
    return CausalForest(n_estimators=20, max_depth=4, n_folds=3, random_state=0, **kwargs)


def test_fit_predict_smoke(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model().fit(x, treatment, y).predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()


def test_predict_components(uplift_data):
    x, treatment, y = uplift_data
    uplift, y0, y1 = _model().fit(x, treatment, y).predict(x, return_components=True)
    assert y0.shape == y1.shape == uplift.shape == (x.shape[0],)
    assert np.allclose(y0, 0.0)
    assert np.allclose(uplift, y1 - y0)


def test_predict_variance(uplift_data):
    x, treatment, y = uplift_data
    model = _model().fit(x, treatment, y)
    var = model.predict_variance(x)
    assert var.shape == (x.shape[0],)
    assert np.isfinite(var).all()
    assert (var >= 0.0).all()


def test_ipw_pseudo_outcome(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model(pseudo_outcome='ipw').fit(x, treatment, y).predict(x)
    assert np.isfinite(uplift).all()


def test_finite_auuc(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model().fit(x, treatment, y).predict(x)
    assert np.isfinite(auuc_score(y, uplift, treatment))


def test_integration_with_uplift_forecast(uplift_data):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_model(alias='cf')])
    forecast.fit(x, treatment, y)
    assert 'uplift_cf' in forecast.predict(x).columns
