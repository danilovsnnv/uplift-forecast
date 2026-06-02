import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from uplift_forecast.matching import (
    EmbeddingMatcher,
    KernelMatcher,
    MahalanobisMatcher,
    MahalanobisPSCaliperMatcher,
    NearestNeighborMatcher,
    PropensityScoreMatcher,
)
from uplift_forecast.models import (
    CausalForest,
    DRLearner,
    PolicyForest,
    PolicyLearner,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
    ZLearner,
)


def _gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=0)


def _lr() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


# Every meta-learner, in light configurations: edge-case tests check that degenerate inputs
# yield finite predictions rather than crashing, not that the estimates are accurate.
_META_BUILDERS = [
    pytest.param(lambda: SLearner(_gbr()), id='SLearner'),
    pytest.param(lambda: TLearner(_gbr()), id='TLearner'),
    pytest.param(lambda: XLearner(model=_gbr(), propensity_model=_lr(), n_folds=3, random_state=0), id='XLearner'),
    pytest.param(
        lambda: RLearner(
            outcome_model=_gbr(), effect_model=Ridge(),
            propensity_model=_lr(), n_folds=3, random_state=0,
        ),
        id='RLearner',
    ),
    pytest.param(
        lambda: DRLearner(
            outcome_model=_gbr(), effect_model=_gbr(),
            propensity_model=_lr(), n_folds=3, random_state=0,
        ),
        id='DRLearner',
    ),
    pytest.param(
        lambda: ZLearner(effect_model=_gbr(), propensity_model=_lr(), n_folds=3, random_state=0),
        id='ZLearner',
    ),
    pytest.param(lambda: CausalForest(n_estimators=20, max_depth=4, n_folds=3, random_state=0), id='CausalForest'),
    pytest.param(lambda: PolicyForest(n_estimators=20, max_depth=4, n_folds=3, random_state=0), id='PolicyForest'),
    pytest.param(lambda: PolicyLearner(cate_estimator=TLearner(_gbr())), id='PolicyLearner'),
]


def _imbalanced_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 95% control / 5% treated, with the treated count fixed so cross-fitting folds stay populated.
    rng = np.random.default_rng(0)
    n, n_features = 800, 5
    x = rng.normal(size=(n, n_features))
    treatment = np.zeros(n, dtype=float)
    treatment[rng.choice(n, size=int(0.05 * n), replace=False)] = 1.0
    y = x[:, 0] + treatment * (1.0 + x[:, 1]) + rng.normal(scale=0.5, size=n)
    return x, treatment, y


def _single_feature_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, 1))
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    y = x[:, 0] + treatment * 1.5 + rng.normal(scale=0.5, size=n)
    return x, treatment, y


def _dataframe_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, n_features = 600, 4
    x = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f'f{i}' for i in range(n_features)])
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    y = x['f0'].to_numpy() + treatment * 1.5 + rng.normal(scale=0.5, size=n)
    return x, treatment, y


def _zero_and_negative_y_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Outcome spans negatives (relevant for the ZILN-based neural path) and carries injected
    # zeros, so both signs and the zero mass are exercised at once.
    rng = np.random.default_rng(0)
    n, n_features = 600, 4
    x = rng.normal(size=(n, n_features))
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    y = x[:, 0] + treatment * 1.0 + rng.normal(scale=0.5, size=n)
    y[rng.random(n) < 0.3] = 0.0
    return x, treatment, y


@pytest.mark.parametrize('build_model', _META_BUILDERS)
@pytest.mark.parametrize(
    'make_data',
    [_imbalanced_data, _single_feature_data, _dataframe_data, _zero_and_negative_y_data],
    ids=['imbalanced', 'single_feature', 'dataframe', 'zero_and_negative_y'],
)
def test_meta_learner_handles_degenerate_inputs(build_model, make_data):
    x, treatment, y = make_data()
    uplift = build_model().fit(x, treatment, y).predict(x)
    assert uplift.shape == (len(treatment),)
    assert np.isfinite(uplift).all()


# k-NN matchers asked for more neighbours than there are controls; the kernel matcher only
# uses n_neighbors in 'knn' candidate mode, so it is built that way to make the check apply.
_KNN_MATCHERS = [
    pytest.param(lambda: MahalanobisMatcher(n_neighbors=999), id='Mahalanobis'),
    pytest.param(lambda: PropensityScoreMatcher(model=_lr(), n_neighbors=999), id='PropensityScore'),
    pytest.param(lambda: EmbeddingMatcher(n_neighbors=999), id='Embedding'),
    pytest.param(lambda: NearestNeighborMatcher(n_neighbors=999), id='NearestNeighbor'),
    pytest.param(
        lambda: MahalanobisPSCaliperMatcher(model=_lr(), caliper=10.0, n_neighbors=999),
        id='MahalanobisPSCaliper',
    ),
    pytest.param(lambda: KernelMatcher(candidate_mode='knn', n_neighbors=999), id='Kernel'),
]


@pytest.mark.parametrize('build_matcher', _KNN_MATCHERS)
def test_matcher_n_neighbors_exceeds_controls(uplift_data, build_matcher):
    x, treatment, y = uplift_data  # ~300 rows, far fewer than the 999 neighbours requested
    with pytest.raises(ValueError, match='n_neighbors'):
        build_matcher().fit_transform(x, treatment, y)
