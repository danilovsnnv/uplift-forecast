import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast
from uplift_forecast import metrics
from uplift_forecast.models import M3TN, MultiSLearner, MultiTLearner


@pytest.fixture
def multi_data():
    rng = np.random.default_rng(0)
    n, p = 300, 6
    x = rng.normal(size=(n, p)).astype('float32')
    treatment = rng.integers(0, 3, size=n)
    y = (x[:, 0] + (treatment == 1) * x[:, 1] + (treatment == 2) * 0.5 * x[:, 2] + rng.normal(size=n)).astype('float32')
    return x, treatment, y


@pytest.mark.parametrize('cls', [MultiTLearner, MultiSLearner])
def test_multi_meta_shapes(cls, multi_data):
    x, treatment, y = multi_data
    model = cls(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    uplift = model.predict(x)
    assert uplift.shape == (300, 2)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert y0.shape == (300,)
    assert y1.shape == (300, 2)


def test_binary_collapses_to_1d(multi_data):
    x, treatment, y = multi_data
    binary = (treatment > 0).astype(int)
    uplift = MultiTLearner(GradientBoostingRegressor(random_state=0)).fit(x, binary, y).predict(x)
    assert uplift.ndim == 1


def test_requires_control_arm(multi_data):
    x, treatment, y = multi_data
    with pytest.raises(ValueError):
        MultiTLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment + 1, y)


def test_multi_arm_metrics(multi_data):
    x, treatment, y = multi_data
    uplift = MultiTLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y).predict(x)
    assert set(metrics.multi_arm_auuc_scores(y, uplift, treatment)) == {1, 2}
    assert set(metrics.multi_arm_qini_scores(y, uplift, treatment)) == {1, 2}
    assignment = metrics.optimal_treatment_assignment(uplift, costs=[0.1, 0.1])
    assert assignment.shape == (300,)
    assert set(np.unique(assignment)).issubset({0, 1, 2})
    x_axis, value = metrics.cost_based_targeting_curve(uplift)
    assert len(x_axis) == 301 == len(value)


def test_forecast_arm_columns(multi_data):
    x, treatment, y = multi_data
    forecast = UpliftForecast([MultiTLearner(GradientBoostingRegressor(random_state=0), alias='mt')]).fit(
        x, treatment, y,
    )
    cols = forecast.predict(x, return_components=True).columns.tolist()
    assert 'uplift_mt_arm1' in cols
    assert 'uplift_mt_arm2' in cols
    assert 'mt_arm2_y1_pred' in cols


def test_m3tn_smoke(multi_data, trainer_kwargs):
    x, treatment, y = multi_data
    model = M3TN(
        input_size=6, n_treatments=3, hidden_size=16, n_experts=2, batch_size=128, **trainer_kwargs,
    ).fit(x, treatment.astype('float32'), y)
    uplift = model.predict(x)
    assert uplift.shape == (300, 2)
    assert np.isfinite(uplift).all()
