import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast.models import SurvivalUplift, TLearner
from uplift_forecast.models.survival import ipcw_survival_pseudo_outcome


@pytest.fixture
def survival_data():
    rng = np.random.default_rng(0)
    n, p = 400, 6
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    time = rng.exponential(5, size=n) * (1.0 + 0.5 * treatment)
    event = (rng.random(n) < 0.7).astype(int)
    y = np.column_stack([time, event])
    return x, treatment, y


def test_fit_predict_smoke(survival_data):
    x, treatment, y = survival_data
    model = SurvivalUplift(TLearner(GradientBoostingRegressor(random_state=0)), horizon=3.0).fit(x, treatment, y)
    uplift = model.predict(x)
    assert uplift.shape == (x.shape[0],)
    assert np.isfinite(uplift).all()


def test_predict_components(survival_data):
    x, treatment, y = survival_data
    model = SurvivalUplift(TLearner(GradientBoostingRegressor(random_state=0)), horizon=3.0).fit(x, treatment, y)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert np.allclose(uplift, y1 - y0)


def test_pseudo_outcome_nonnegative():
    rng = np.random.default_rng(0)
    n = 200
    time = rng.exponential(5, size=n)
    event = (rng.random(n) < 0.7).astype(int)
    pseudo = ipcw_survival_pseudo_outcome(time, event, 3.0)
    assert pseudo.shape == (n,)
    assert (pseudo >= 0).all()


def test_requires_two_column_y(survival_data):
    x, treatment, _ = survival_data
    with pytest.raises(ValueError):
        SurvivalUplift(TLearner(GradientBoostingRegressor(random_state=0)), horizon=3.0).fit(
            x, treatment, np.zeros(len(treatment)),
        )
