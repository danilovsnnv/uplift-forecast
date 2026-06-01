import numpy as np

from uplift_forecast import UpliftForecast
from uplift_forecast.metrics import auuc_score
from uplift_forecast.models import PolicyForest


def _model(**kwargs) -> PolicyForest:
    return PolicyForest(n_estimators=20, max_depth=4, n_folds=3, random_state=0, **kwargs)


def test_fit_predict_smoke(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model().fit(x, treatment, y).predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()


def test_predict_components(uplift_data):
    x, treatment, y = uplift_data
    uplift, y0, y1 = _model().fit(x, treatment, y).predict(x, return_components=True)
    assert np.allclose(y0, 0.0)
    assert np.allclose(uplift, y1 - y0)


def test_assign_binary(uplift_data):
    x, treatment, y = uplift_data
    decisions = _model().fit(x, treatment, y).assign(x)
    assert decisions.shape == (x.shape[0],)
    assert set(np.unique(decisions)).issubset({0, 1})


def test_assign_top_k(uplift_data):
    x, treatment, y = uplift_data
    assert _model().fit(x, treatment, y).assign(x, top_k=10).sum() == 10


def test_policy_value_finite(uplift_data):
    x, treatment, y = uplift_data
    value = _model().fit(x, treatment, y).policy_value(x, treatment, y)
    assert np.isfinite(value)


def test_finite_auuc(uplift_data):
    x, treatment, y = uplift_data
    uplift = _model().fit(x, treatment, y).predict(x)
    assert np.isfinite(auuc_score(y, uplift, treatment))


def test_integration_with_uplift_forecast(uplift_data):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_model(alias='pf')])
    forecast.fit(x, treatment, y)
    assert 'uplift_pf' in forecast.predict(x).columns
