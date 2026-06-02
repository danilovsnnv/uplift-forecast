import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from uplift_forecast.matching import (
    CoarsenedExactMatcher,
    PropensityScoreMatcher,
    covariate_balance,
)


def test_propensity_matching_reduces_smd_for_every_feature():
    # Every covariate drives treatment, so all features start imbalanced; matching on the
    # propensity score must shrink |SMD| for each feature, not merely on average.
    rng = np.random.default_rng(0)
    n, n_features = 2000, 5
    x = rng.normal(size=(n, n_features))
    treatment = (rng.random(n) < 1.0 / (1.0 + np.exp(-x.sum(axis=1)))).astype(int)
    y = rng.normal(size=n)

    matched = PropensityScoreMatcher(
        model=LogisticRegression(max_iter=1000), caliper=0.05,
    ).fit_transform(x, treatment, y)
    feat = [c for c in matched.columns if c.startswith('feature_')]

    before = np.abs(covariate_balance(x, treatment)['smd'].to_numpy())
    after = np.abs(covariate_balance(
        matched[feat].to_numpy(), matched['treatment'].to_numpy(), weight=matched['weight'].to_numpy(),
    )['smd'].to_numpy())
    assert (after < before).all()


def test_cem_match_rate_one_on_separable_strata():
    # Each coarsened stratum contains both arms by construction, so CEM drops nothing
    # and the match rate is exactly 1.0.
    feature = np.repeat(np.arange(4.0), 50).reshape(-1, 1)
    treatment = np.tile([1, 0], feature.shape[0] // 2)
    y = np.zeros(feature.shape[0])

    matcher = CoarsenedExactMatcher(n_bins=4).fit(feature, treatment)
    matcher.transform(feature, treatment, y)
    assert matcher.match_rate_ == pytest.approx(1.0)
