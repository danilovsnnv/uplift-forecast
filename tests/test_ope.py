import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import ope
from uplift_forecast.models import TLearner

ESTIMATORS = ['ips', 'snips', 'dm', 'dr', 'switch_dr', 'dr_os']


@pytest.fixture
def policy_setup():
    rng = np.random.default_rng(0)
    n, p = 500, 6
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    y = x[:, 0] + treatment * np.clip(x[:, 1], 0, None) + rng.normal(size=n)
    model = TLearner(GradientBoostingRegressor(random_state=0)).fit(x, treatment, y)
    propensity = np.full(n, treatment.mean())
    return model, x, treatment, y, propensity


@pytest.mark.parametrize('estimator', ESTIMATORS)
def test_evaluate_policy_finite(estimator, policy_setup):
    model, x, treatment, y, propensity = policy_setup
    value = ope.evaluate_policy(model, x, treatment, y, propensity, estimator=estimator)
    assert np.isfinite(value)


def test_evaluate_policy_bad_estimator(policy_setup):
    model, x, treatment, y, propensity = policy_setup
    with pytest.raises(ValueError):
        ope.evaluate_policy(model, x, treatment, y, propensity, estimator='nope')


def test_raw_estimators_finite():
    rng = np.random.default_rng(1)
    n = 300
    reward = rng.normal(size=n)
    action = rng.integers(0, 2, size=n)
    policy_action = rng.integers(0, 2, size=n)
    pscore = np.full(n, 0.5)
    q_hat = rng.normal(size=(n, 2))
    values = [
        ope.ips(reward, action, pscore, policy_action),
        ope.snips(reward, action, pscore, policy_action),
        ope.direct_method(q_hat, policy_action),
        ope.doubly_robust(reward, action, pscore, policy_action, q_hat),
        ope.switch_dr(reward, action, pscore, policy_action, q_hat),
        ope.dr_os(reward, action, pscore, policy_action, q_hat),
    ]
    assert all(np.isfinite(v) for v in values)


def test_pscore_must_be_positive():
    with pytest.raises(ValueError):
        ope.ips([1.0], [1], [0.0], [1])
