import numpy as np
import pandas as pd
import pytest

from uplift_forecast.matching import (
    MahalanobisMatcher,
    covariate_balance,
    match_rate,
    standardized_mean_difference,
)


def test_smd_one_value_per_feature(uplift_data):
    x, treatment, y = uplift_data
    smd = standardized_mean_difference(x, treatment)
    assert smd.shape == (x.shape[1],)
    assert np.isfinite(smd).all()


def test_covariate_balance_table_schema(uplift_data):
    x, treatment, y = uplift_data
    table = covariate_balance(x, treatment)
    assert list(table.columns) == ['mean_treated', 'mean_control', 'smd']
    assert table.shape[0] == x.shape[1]
    assert table.index.name == 'feature'


def test_matching_reduces_imbalance():
    # Treatment assignment depends on feature 0, so the raw arms are imbalanced;
    # matching on the covariates should shrink the average |SMD|.
    rng = np.random.default_rng(0)
    n, n_features = 400, 6
    x = rng.normal(size=(n, n_features))
    treatment = (rng.normal(size=n) + x[:, 0] > 0).astype(int)
    y = rng.normal(size=n)

    matched = MahalanobisMatcher().fit_transform(x, treatment, y)
    feat = [c for c in matched.columns if c.startswith('feature_')]
    before = covariate_balance(x, treatment)
    after = covariate_balance(
        matched[feat].to_numpy(), matched['treatment'].to_numpy(), weight=matched['weight'].to_numpy(),
    )
    assert np.abs(after['smd']).mean() < np.abs(before['smd']).mean()


def test_match_rate_range(uplift_data):
    x, treatment, y = uplift_data
    matched = MahalanobisMatcher().fit_transform(x, treatment, y)
    rate = match_rate(treatment, matched['treatment'].to_numpy())
    assert 0.0 <= rate <= 1.0


def test_match_rate_all_retained():
    t = np.array([1, 1, 0, 0])
    assert match_rate(t, np.array([1, 1, 0])) == pytest.approx(1.0)


def test_match_rate_zero_treated_raises():
    with pytest.raises(ValueError):
        match_rate(np.zeros(5), np.zeros(3))


def test_zero_variance_feature_no_nan():
    x = np.column_stack([np.ones(20), np.arange(20.0)])
    t = np.array([1, 0] * 10)
    smd = standardized_mean_difference(x, t)
    assert np.isfinite(smd).all()
    assert smd[0] == 0.0
