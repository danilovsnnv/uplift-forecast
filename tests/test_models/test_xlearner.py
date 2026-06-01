import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from uplift_forecast import UpliftForecast
from uplift_forecast.metrics import auuc_score
from uplift_forecast.models import XLearner


def _model(**kwargs) -> XLearner:
    kwargs.setdefault('propensity_model', LogisticRegression(max_iter=1000))
    return XLearner(
        model=GradientBoostingRegressor(random_state=0),
        n_folds=3,
        random_state=0,
        **kwargs,
    )


def test_fit_predict_smoke(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model().fit(x, treatment, y).predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()


def test_predict_components(uplift_data):
    x, treatment, y = uplift_data
    uplift, y0, y1 = _model().fit(x, treatment, y).predict(x, return_components=True)
    assert y0.shape == y1.shape == uplift.shape == (x.shape[0],)
    assert np.allclose(uplift, y1 - y0)
    assert np.isfinite(y0).all() and np.isfinite(y1).all()


def test_global_propensity(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model(propensity_model=None).fit(x, treatment, y).predict(x)
    assert np.isfinite(uplift).all()


def test_finite_auuc(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model().fit(x, treatment, y).predict(x)
    assert np.isfinite(auuc_score(y, uplift, treatment))


def test_integration_with_uplift_forecast(uplift_data):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_model(alias='xl')])
    forecast.fit(x, treatment, y)
    assert 'uplift_xl' in forecast.predict(x).columns
