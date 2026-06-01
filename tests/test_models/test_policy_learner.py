import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast
from uplift_forecast.metrics import auuc_score
from uplift_forecast.models import PolicyLearner, TLearner


def _model(**kwargs) -> PolicyLearner:
    return PolicyLearner(
        cate_estimator=TLearner(GradientBoostingRegressor(random_state=0)),
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
    assert np.allclose(y0, 0.0)
    assert np.allclose(uplift, y1 - y0)


def test_assign_binary(uplift_data):
    x, treatment, y = uplift_data
    model = _model().fit(x, treatment, y)
    decisions = model.assign(x)
    assert decisions.shape == (x.shape[0],)
    assert set(np.unique(decisions)).issubset({0, 1})


def test_assign_top_k_and_budget(uplift_data):
    x, treatment, y = uplift_data
    model = _model().fit(x, treatment, y)
    assert model.assign(x, top_k=10).sum() == 10
    assert model.assign(x, budget=0.2).sum() == round(0.2 * x.shape[0])


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
    forecast = UpliftForecast(models=[_model(alias='pl')])
    forecast.fit(x, treatment, y)
    assert 'uplift_pl' in forecast.predict(x).columns
