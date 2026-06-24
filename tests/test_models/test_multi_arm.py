"""Multi-arm behaviour of the canonical meta-learners (SLearner/TLearner/DRLearner).

The binary paths are covered in test_correctness/test_meta_learners.py and test_drlearner.py;
here we exercise the K>2 arm decomposition, the binary-collapse fallback, and the multi-arm
metric dispatch.
"""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from uplift_forecast import UpliftForecast
from uplift_forecast import metrics
from uplift_forecast.losses import CFRLoss
from uplift_forecast.models import (
    CFRNet,
    CausalForest,
    DRLearner,
    DragonNet,
    M3TN,
    PolicyForest,
    PolicyLearner,
    RLearner,
    SLearner,
    TARNet,
    TLearner,
    UpliftTree,
    XLearner,
    ZLearner,
)


def _dr(**kwargs) -> DRLearner:
    kwargs.setdefault('propensity_model', LogisticRegression(max_iter=1000))
    return DRLearner(
        outcome_model=GradientBoostingRegressor(random_state=0),
        effect_model=GradientBoostingRegressor(random_state=0),
        n_folds=3,
        random_state=0,
        **kwargs,
    )


@pytest.fixture
def multi_data():
    rng = np.random.default_rng(0)
    n, p = 300, 6
    x = rng.normal(size=(n, p)).astype('float32')
    treatment = rng.integers(0, 3, size=n)
    y = (x[:, 0] + (treatment == 1) * x[:, 1] + (treatment == 2) * 0.5 * x[:, 2] + rng.normal(size=n)).astype('float32')
    return x, treatment, y


@pytest.mark.parametrize('cls', [TLearner, SLearner])
def test_multi_meta_shapes(cls, multi_data):
    x, treatment, y = multi_data
    model = cls(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    uplift = model.predict(x)
    assert uplift.shape == (300, 2)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert y0.shape == (300,)
    assert y1.shape == (300, 2)


@pytest.mark.parametrize('cls', [TLearner, SLearner])
def test_multi_meta_ipw(cls, multi_data):
    x, treatment, y = multi_data
    model = cls(
        GradientBoostingRegressor(random_state=0),
        propensity_model=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=0,
    ).fit(x, treatment, y)
    uplift = model.predict(x)
    assert uplift.shape == (300, 2)
    assert np.isfinite(uplift).all()
    assert model._propensity_model is not None


def test_multi_meta_ipw_rejects_bad_clip():
    with pytest.raises(ValueError, match='propensity_clip'):
        TLearner(
            GradientBoostingRegressor(random_state=0),
            propensity_model=LogisticRegression(max_iter=1000),
            propensity_clip=0.0,
        )


def test_binary_collapses_to_1d(multi_data):
    x, treatment, y = multi_data
    binary = (treatment > 0).astype(int)
    uplift = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, binary, y).predict(x)
    assert uplift.ndim == 1


def test_requires_control_arm(multi_data):
    x, treatment, y = multi_data
    with pytest.raises(ValueError, match='control arm'):
        TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment + 1, y)


def test_multi_arm_metrics(multi_data):
    x, treatment, y = multi_data
    uplift = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y).predict(x)
    assert set(metrics.auuc_score(y, uplift, treatment)) == {1, 2}
    assert set(metrics.qini_score(y, uplift, treatment)) == {1, 2}
    assignment = metrics.optimal_treatment_assignment(uplift, costs=[0.1, 0.1])
    assert assignment.shape == (300,)
    assert set(np.unique(assignment)).issubset({0, 1, 2})
    x_axis, value = metrics.cost_based_targeting_curve(uplift)
    assert len(x_axis) == 301 == len(value)


def test_forecast_arm_columns(multi_data):
    x, treatment, y = multi_data
    forecast = UpliftForecast([TLearner(GradientBoostingRegressor(random_state=0), alias='mt')]).fit(
        x, treatment, y,
    )
    cols = forecast.predict(x, return_components=True).columns.tolist()
    assert 'uplift_mt_arm1' in cols
    assert 'uplift_mt_arm2' in cols
    assert 'mt_arm2_y1_pred' in cols


def test_dr_shapes(multi_data):
    x, treatment, y = multi_data
    model = _dr().fit(x, treatment, y)
    uplift = model.predict(x)
    assert uplift.shape == (300, 2)
    assert np.isfinite(uplift).all()

    uplift, y0, y1 = model.predict(x, return_components=True)
    assert y0.shape == (300,)
    assert y1.shape == (300, 2)
    assert np.allclose(uplift, y1 - y0[:, None])
    assert np.isfinite(y0).all()
    assert np.isfinite(y1).all()


def test_dr_global_propensity(multi_data):
    x, treatment, y = multi_data
    uplift = _dr(propensity_model=None).fit(x, treatment, y).predict(x)
    assert uplift.shape == (300, 2)
    assert np.isfinite(uplift).all()


def test_dr_binary_collapses(multi_data):
    x, treatment, y = multi_data
    binary = (treatment > 0).astype(int)
    uplift = _dr().fit(x, binary, y).predict(x)
    assert uplift.ndim == 1


def test_dr_requires_control_arm(multi_data):
    x, treatment, y = multi_data
    with pytest.raises(ValueError, match='control arm'):
        _dr().fit(x, treatment + 1, y)


def test_dr_metrics(multi_data):
    x, treatment, y = multi_data
    uplift = _dr().fit(x, treatment, y).predict(x)
    assert set(metrics.auuc_score(y, uplift, treatment)) == {1, 2}


def test_dr_forecast_columns(multi_data):
    x, treatment, y = multi_data
    forecast = UpliftForecast([_dr(alias='mdr')]).fit(x, treatment, y)
    cols = forecast.predict(x, return_components=True).columns.tolist()
    assert 'uplift_mdr_arm1' in cols
    assert 'uplift_mdr_arm2' in cols
    assert 'mdr_arm2_y1_pred' in cols


def test_tlearner_model_treated_multi_arm(multi_data):
    # model_treated must apply to every treated arm (and leave control on `model`).
    x, treatment, y = multi_data
    model = TLearner(
        GradientBoostingRegressor(random_state=0),
        model_treated=GradientBoostingRegressor(random_state=1),
    ).fit(x, treatment, y)
    assert set(model._models) == {0, 1, 2}
    assert np.isfinite(model.predict(x)).all()


def _xrz_factories() -> list:
    return [
        lambda: XLearner(GradientBoostingRegressor(random_state=0)),
        lambda: RLearner(GradientBoostingRegressor(random_state=0), Ridge()),
        lambda: ZLearner(GradientBoostingRegressor(random_state=0)),
    ]


def test_xlearner_multi_arm(multi_data):
    x, treatment, y = multi_data
    model = XLearner(
        GradientBoostingRegressor(random_state=0),
        propensity_model=LogisticRegression(max_iter=1000),
        n_folds=3,
    ).fit(x, treatment, y)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert uplift.shape == (300, 2)
    assert y0.shape == (300,)
    assert y1.shape == (300, 2)
    assert np.allclose(uplift, y1 - y0[:, None])
    assert np.isfinite(uplift).all()
    assert set(metrics.auuc_score(y, uplift, treatment)) == {1, 2}


@pytest.mark.parametrize('cls', [RLearner, ZLearner])
def test_rz_multi_arm_zero_baseline(cls, multi_data):
    x, treatment, y = multi_data
    kwargs = {'propensity_model': LogisticRegression(max_iter=1000), 'n_folds': 3}
    model = (
        cls(GradientBoostingRegressor(random_state=0), Ridge(), **kwargs)
        if cls is RLearner
        else cls(GradientBoostingRegressor(random_state=0), **kwargs)
    ).fit(x, treatment, y)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert uplift.shape == (300, 2)
    assert np.allclose(y0, 0.0)
    assert np.allclose(uplift, y1)
    assert np.isfinite(uplift).all()


@pytest.mark.parametrize('make', _xrz_factories())
def test_xrz_binary_collapses(make, multi_data):
    x, treatment, y = multi_data
    binary = (treatment > 0).astype(int)
    assert make().fit(x, binary, y).predict(x).ndim == 1


@pytest.mark.parametrize('make', _xrz_factories())
def test_xrz_requires_control_arm(make, multi_data):
    x, treatment, y = multi_data
    with pytest.raises(ValueError, match='control arm'):
        make().fit(x, treatment + 1, y)


def test_m3tn_smoke(multi_data, trainer_kwargs):
    x, treatment, y = multi_data
    model = M3TN(
        input_size=6, n_treatments=3, hidden_size=16, n_experts=2, batch_size=128, **trainer_kwargs,
    ).fit(x, treatment.astype('float32'), y)
    uplift = model.predict(x)
    assert uplift.shape == (300, 2)
    assert np.isfinite(uplift).all()


def test_m3tn_binary_collapses(uplift_data, trainer_kwargs):
    x, treatment, y = uplift_data
    model = M3TN(input_size=6, n_treatments=2, hidden_size=16, n_experts=2, batch_size=128, **trainer_kwargs)
    assert model.fit(x, treatment, y).predict(x).ndim == 1


@pytest.mark.parametrize('cls', [TARNet, CFRNet, DragonNet])
def test_neural_multi_arm_smoke(cls, multi_data, trainer_kwargs):
    x, treatment, y = multi_data
    model = cls(input_size=6, n_treatments=3, hidden_size=16, batch_size=128, **trainer_kwargs).fit(
        x, treatment.astype('float32'), y,
    )
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert uplift.shape == (300, 2)
    assert y0.shape == (300,)
    assert y1.shape == (300, 2)
    assert np.allclose(uplift, y1 - y0[:, None])
    assert np.isfinite(uplift).all()


def test_cfrnet_multi_arm_ipm(multi_data, trainer_kwargs):
    # The per-arm-vs-control IPM penalty path (p_alpha>0) must train and predict finitely.
    x, treatment, y = multi_data
    model = CFRNet(
        input_size=6, n_treatments=3, hidden_size=16, batch_size=128,
        loss=CFRLoss(p_alpha=1.0, imb_fun='mmd2_lin'), **trainer_kwargs,
    ).fit(x, treatment.astype('float32'), y)
    assert np.isfinite(model.predict(x)).all()


@pytest.fixture
def multi_binary_data():
    rng = np.random.default_rng(1)
    n, p = 900, 5
    x = rng.normal(size=(n, p))
    t = rng.integers(0, 3, size=n)
    rate = np.clip(0.3 + 0.15 * (t == 1) + 0.25 * (t == 2) + 0.1 * (x[:, 0] > 0), 0.0, 1.0)
    y = (rng.random(n) < rate).astype(float)
    return x, t, y


@pytest.mark.parametrize('mode', ['multi_way', 'per_arm'])
def test_uplift_tree_multi_arm(mode, multi_binary_data):
    x, t, y = multi_binary_data
    model = UpliftTree(
        max_depth=3, min_samples_leaf=40, min_samples_treatment=5, multi_arm_split=mode,
    ).fit(x, t, y)
    uplift = model.predict(x)
    assert uplift.shape == (900, 2)
    assert np.isfinite(uplift).all()
    assert set(metrics.auuc_score(y, uplift, t)) == {1, 2}


def test_uplift_tree_bad_split_mode():
    with pytest.raises(ValueError, match='multi_arm_split'):
        UpliftTree(multi_arm_split='nope')


def test_causal_forest_multi_arm(multi_data):
    x, treatment, y = multi_data
    model = CausalForest(n_estimators=20, max_depth=4, n_folds=3, random_state=0).fit(x, treatment, y)
    uplift, y0, y1 = model.predict(x, return_components=True)
    assert uplift.shape == (300, 2)
    assert np.allclose(y0, 0.0)
    assert model.predict_variance(x).shape == (300, 2)
    lower, upper = model.predict_interval(x, alpha=0.1)
    assert lower.shape == upper.shape == (300, 2)
    assert (lower <= uplift + 1e-9).all()
    assert (uplift <= upper + 1e-9).all()


def test_policy_forest_multi_arm(multi_data):
    x, treatment, y = multi_data
    model = PolicyForest(n_estimators=20, max_depth=4, n_folds=3, random_state=0).fit(x, treatment, y)
    assert model.predict(x).shape == (300, 2)
    assert set(np.unique(model.assign(x))).issubset({0, 1, 2})
    assert (model.assign(x, top_k=10) > 0).sum() <= 10
    assert np.isfinite(model.policy_value(x, treatment, y))


def test_policy_learner_multi_arm(multi_data):
    x, treatment, y = multi_data
    model = PolicyLearner(TLearner(GradientBoostingRegressor(random_state=0))).fit(x, treatment, y)
    assert model.predict(x).shape == (300, 2)
    assert set(np.unique(model.assign(x))).issubset({0, 1, 2})
    assert np.isfinite(model.policy_value(x, treatment, y))
