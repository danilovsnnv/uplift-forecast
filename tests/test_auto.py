import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast
from uplift_forecast.auto import AutoUplift
from uplift_forecast.models import SLearner, TLearner


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, p = 400, 6
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    y = x[:, 0] + treatment * np.clip(x[:, 1], 0, None) + rng.normal(size=n)
    return x, treatment, y


def _candidates():
    return [
        TLearner(GradientBoostingRegressor(random_state=0), alias='tl'),
        SLearner(GradientBoostingRegressor(random_state=0), alias='sl'),
    ]


def test_selects_and_ranks(data):
    x, treatment, y = data
    auto = AutoUplift(_candidates(), metric='qini', random_state=0).fit(x, treatment, y)
    assert auto.leaderboard_.shape[0] == 2
    assert set(auto.leaderboard_['model']) == {'tl', 'sl'}
    assert auto.leaderboard_['val_score'].is_monotonic_decreasing
    assert auto.predict(x).shape == (len(y),)
    assert auto.best_model_ is not None


def test_top_k_ensemble(data):
    x, treatment, y = data
    auto = AutoUplift(_candidates(), top_k=2, random_state=0).fit(x, treatment, y)
    assert len(auto._ensemble) == 2
    uplift, y0, y1 = auto.predict(x, return_components=True)
    assert np.allclose(uplift, y1 - y0)


def test_eval_set_used(data):
    x, treatment, y = data
    auto = AutoUplift(_candidates()[:1]).fit(x, treatment, y, eval_set=(x, treatment, y))
    assert auto.leaderboard_.shape[0] == 1


def test_integration_with_uplift_forecast(data):
    x, treatment, y = data
    forecast = UpliftForecast([AutoUplift(_candidates(), alias='auto')]).fit(x, treatment, y)
    assert 'uplift_auto' in forecast.predict(x).columns


def test_bad_metric():
    with pytest.raises(ValueError):
        AutoUplift(_candidates(), metric='nope')


def test_empty_candidates():
    with pytest.raises(ValueError):
        AutoUplift([])
