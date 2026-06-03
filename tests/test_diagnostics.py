import numpy as np
import pytest

from uplift_forecast import diagnostics


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, p = 400, 5
    x = rng.normal(size=(n, p))
    treatment = rng.integers(0, 2, size=n)
    return x, treatment


def test_variance_ratio_shape(data):
    x, treatment = data
    vr = diagnostics.variance_ratio(x, treatment)
    assert vr.shape == (x.shape[1],)
    assert np.isfinite(vr).all()


def test_variance_ratio_requires_both_arms(data):
    x, _ = data
    with pytest.raises(ValueError):
        diagnostics.variance_ratio(x, np.ones(len(x), dtype=int))


def test_positivity_check():
    rng = np.random.default_rng(0)
    n = 300
    propensity = rng.uniform(0, 1, size=n)
    treatment = (rng.random(n) < propensity).astype(int)
    report = diagnostics.positivity_check(propensity, treatment, low=0.1, high=0.9)
    assert 0.0 <= report['share_outside_overlap'] <= 1.0
    assert 'has_overlap' in report
    assert 'min_treated_propensity' in report


def test_overlap_report(data):
    x, treatment = data
    propensity = np.full(len(treatment), treatment.mean())
    report = diagnostics.overlap_report(x, treatment, propensity=propensity)
    assert report['balance'].shape[0] == x.shape[1]
    assert set(report['balance'].columns) == {'smd', 'variance_ratio', 'smd_ok', 'variance_ratio_ok'}
    assert isinstance(report['balanced'], bool)
    assert 'positivity' in report
