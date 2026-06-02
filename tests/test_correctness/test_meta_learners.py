import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

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


def _propensity() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation (Spearman) without a SciPy dependency."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# (model builder, MAE tolerance) per CATE-estimating meta-learner. Tolerances sit comfortably
# above the observed error (see the handoff note for measured MAE per model) so the suite
# catches recovery regressions without flaking across sklearn / Python versions.
#
# CausalForest uses max_features=None here: with only one of five features informative, the
# default 'sqrt' subsampling starves most trees of the signal (MAE ~1.0); given every feature
# it recovers tau cleanly (MAE ~0.13). PolicyForest is validated separately below -- its
# predict returns a rankable gain score, not a calibrated CATE, so MAE is the wrong check.
_LEARNERS = [
    pytest.param(lambda: SLearner(_gbr()), 0.4, id='SLearner'),
    pytest.param(lambda: TLearner(_gbr()), 0.3, id='TLearner'),
    pytest.param(
        lambda: XLearner(model=_gbr(), propensity_model=_propensity(), n_folds=5, random_state=0),
        0.2, id='XLearner',
    ),
    pytest.param(
        lambda: RLearner(
            outcome_model=_gbr(), effect_model=Ridge(),
            propensity_model=_propensity(), n_folds=5, random_state=0,
        ),
        0.1, id='RLearner',
    ),
    pytest.param(
        lambda: DRLearner(
            outcome_model=_gbr(), effect_model=_gbr(),
            propensity_model=_propensity(), n_folds=5, random_state=0,
        ),
        0.2, id='DRLearner',
    ),
    pytest.param(
        lambda: ZLearner(effect_model=_gbr(), propensity_model=_propensity(), n_folds=5, random_state=0),
        0.5, id='ZLearner',
    ),
    pytest.param(
        lambda: PolicyLearner(cate_estimator=TLearner(_gbr())),
        0.3, id='PolicyLearner',
    ),
    pytest.param(
        lambda: CausalForest(n_estimators=200, max_features=None, n_folds=5, random_state=0),
        0.3, id='CausalForest',
    ),
]


@pytest.mark.parametrize(('build_model', 'tol'), _LEARNERS)
def test_recovers_known_cate(known_cate_data, build_model, tol):
    x, treatment, y, true_cate = known_cate_data
    uplift = build_model().fit(x, treatment, y).predict(x)
    mae = float(np.mean(np.abs(uplift - true_cate)))
    assert mae < tol, f'CATE MAE {mae:.3f} exceeds tolerance {tol}'


def test_policy_forest_ranks_by_cate(known_cate_data):
    # PolicyForest reports a doubly-robust gain score, not a CATE magnitude, so correctness
    # means ranking units by their true effect -- not matching tau in absolute units.
    x, treatment, y, true_cate = known_cate_data
    model = PolicyForest(n_estimators=200, max_features=None, n_folds=5, random_state=0)
    score = model.fit(x, treatment, y).predict(x)
    assert _spearman(score, true_cate) > 0.9
    # ...and it should recommend treating units whose true effect is positive.
    assign_accuracy = float(np.mean(model.assign(x) == (true_cate > 0).astype(int)))
    assert assign_accuracy > 0.9
