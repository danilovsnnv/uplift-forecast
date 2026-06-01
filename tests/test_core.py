import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast
from uplift_forecast.common._uplift_model import UpliftModel
from uplift_forecast.models import SLearner, TLearner


def _t_learner(alias: str | None = None) -> TLearner:
    return TLearner(GradientBoostingRegressor(random_state=0), alias=alias)


def _s_learner(alias: str | None = None) -> SLearner:
    return SLearner(GradientBoostingRegressor(random_state=0), alias=alias)


def test_predict_collects_one_column_per_model(uplift_data):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_t_learner('tl'), _s_learner('sl')])
    forecast.fit(x, treatment, y)

    preds = forecast.predict(x)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == ['uplift_tl', 'uplift_sl']
    assert len(preds) == x.shape[0]
    assert np.isfinite(preds.to_numpy()).all()


def test_predict_return_components_schema(uplift_data):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_t_learner('tl')]).fit(x, treatment, y)

    preds = forecast.predict(x, return_components=True)
    assert list(preds.columns) == ['uplift_tl', 'tl_y0_pred', 'tl_y1_pred']
    # the uplift column must equal y1 - y0 exactly, not just be finite
    assert np.allclose(preds['uplift_tl'], preds['tl_y1_pred'] - preds['tl_y0_pred'])


def test_column_names_use_class_name_without_alias(uplift_data):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_t_learner()]).fit(x, treatment, y)
    assert 'uplift_TLearner' in forecast.predict(x).columns


def test_predictions_match_standalone_model(uplift_data):
    # UpliftForecast must not alter a model's predictions — it only collects them.
    x, treatment, y = uplift_data
    standalone = _t_learner('tl').fit(x, treatment, y).predict(x)
    via_forecast = UpliftForecast(models=[_t_learner('tl')]).fit(x, treatment, y).predict(x)
    assert np.allclose(via_forecast['uplift_tl'].to_numpy(), standalone)


def test_save_load_roundtrip_preserves_predictions(uplift_data, tmp_path):
    x, treatment, y = uplift_data
    forecast = UpliftForecast(models=[_t_learner('tl')]).fit(x, treatment, y)
    before = forecast.predict(x)

    path = tmp_path / 'nested' / 'forecast.pkl'
    forecast.save(path)
    assert path.exists()  # save() creates parent directories

    after = UpliftForecast.load(path).predict(x)
    pd.testing.assert_frame_equal(before, after)


def test_val_df_and_fit_params_are_forwarded(uplift_data):
    # A stub model records what fit() received, proving UpliftForecast forwards
    # val_df as eval_set and passes **fit_params through untouched.
    x, treatment, y = uplift_data

    class _Recorder(UpliftModel):
        alias = 'rec'

        def fit(self, X, treatment, y, eval_set=None, **fit_params):
            self.seen_eval_set = eval_set
            self.seen_fit_params = fit_params
            return self

        def predict(self, X, *, return_components=False):
            return np.zeros(len(X))

    recorder = _Recorder()
    val = (x[:50], treatment[:50], y[:50])
    UpliftForecast(models=[recorder]).fit(x, treatment, y, val_df=val, sample_weight='w')
    assert recorder.seen_eval_set is val
    assert recorder.seen_fit_params == {'sample_weight': 'w'}


def test_non_uplift_model_rejected():
    with pytest.raises(TypeError, match='subclass'):
        UpliftForecast(models=[object()])


def test_reproducible_across_two_fits(uplift_data):
    # Seeded estimators must give identical predictions on independent fits.
    x, treatment, y = uplift_data
    first = _t_learner('tl').fit(x, treatment, y).predict(x)
    second = _t_learner('tl').fit(x, treatment, y).predict(x)
    assert np.array_equal(first, second)
