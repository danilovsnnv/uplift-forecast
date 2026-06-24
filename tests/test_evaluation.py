import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast, evaluation, metrics, ope
from uplift_forecast.models import SLearner, TLearner


@pytest.fixture
def fitted_forecast():
    rng = np.random.default_rng(0)
    n, p = 300, 6
    x = rng.normal(size=(n, p)).astype('float32')
    t = rng.integers(0, 3, size=n)
    y = (x[:, 0] + (t == 1) * x[:, 1] + (t == 2) * 0.5 * x[:, 2] + rng.normal(size=n)).astype('float32')
    fc = UpliftForecast([
        TLearner(GradientBoostingRegressor(random_state=0), alias='tl'),
        SLearner(GradientBoostingRegressor(random_state=0), alias='sl'),
    ]).fit(x, t, y)
    return fc, x, t, y


def test_compare_models_pandas(fitted_forecast):
    fc, x, t, y = fitted_forecast
    table = evaluation.compare_models(fc, x, y, t)
    assert list(table.columns) == ['model', 'arm', 'auuc', 'qini', 'lift']
    assert len(table) == 4  # 2 models x 2 treated arms
    assert set(table['model']) == {'tl', 'sl'}
    assert set(table['arm']) == {1, 2}
    assert np.isfinite(table['auuc']).all()


def test_compare_models_polars(fitted_forecast):
    pytest.importorskip('polars')
    fc, x, t, y = fitted_forecast
    table = evaluation.compare_models(fc, x, y, t, frame='polars')
    assert type(table).__name__ == 'DataFrame'  # polars.DataFrame
    assert table.shape == (4, 5)
    assert table.columns == ['model', 'arm', 'auuc', 'qini', 'lift']


def test_compare_models_binary_single_arm():
    rng = np.random.default_rng(1)
    n, p = 200, 5
    x = rng.normal(size=(n, p)).astype('float32')
    t = rng.integers(0, 2, size=n)
    y = (x[:, 0] + t * x[:, 1] + rng.normal(size=n)).astype('float32')
    fc = UpliftForecast([TLearner(GradientBoostingRegressor(random_state=0), alias='tl')]).fit(x, t, y)
    table = evaluation.compare_models(fc, x, y, t, metrics=('auuc',))
    assert list(table.columns) == ['model', 'arm', 'auuc']
    assert table['arm'].tolist() == [1]


def test_compare_models_rejects_unknown_metric(fitted_forecast):
    fc, x, t, y = fitted_forecast
    with pytest.raises(ValueError, match='unknown metric'):
        evaluation.compare_models(fc, x, y, t, metrics=('auuc', 'nope'))


def test_compare_models_rejects_bad_frame(fitted_forecast):
    fc, x, t, y = fitted_forecast
    with pytest.raises(ValueError, match='frame must be'):
        evaluation.compare_models(fc, x, y, t, frame='spark')


def test_expected_response_matches_snips(fitted_forecast):
    fc, x, t, y = fitted_forecast
    recommended = metrics.optimal_treatment_assignment(fc.models[0].predict(x))
    p_taken = np.full(len(t), 1 / 3.0)
    value = evaluation.expected_response(y, recommended, t, p_taken)
    assert np.isfinite(value)
    assert value == pytest.approx(ope.snips(y, t, p_taken, recommended))
