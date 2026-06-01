import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from uplift_forecast.matching import MahalanobisPSCaliperMatcher


def _model() -> LogisticRegression:
    return LogisticRegression(max_iter=200)


def test_fit_transform_smoke(uplift_data):
    x, treatment, y = uplift_data
    matched = MahalanobisPSCaliperMatcher(model=_model(), caliper=0.2).fit_transform(x, treatment, y)
    assert isinstance(matched, pd.DataFrame)
    assert {'treatment', 'y', 'weight'} <= set(matched.columns)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()


def test_return_unmatched_tuple(uplift_data):
    x, treatment, y = uplift_data
    m = MahalanobisPSCaliperMatcher(model=_model(), caliper=0.2, return_unmatched=True)
    out = m.fit_transform(x, treatment, y)
    assert isinstance(out, tuple) and len(out) == 2
    matched, unmatched = out
    assert isinstance(matched, pd.DataFrame)
    assert isinstance(unmatched, np.ndarray)


def test_tight_caliper_leaves_more_unmatched(uplift_data):
    x, treatment, y = uplift_data
    loose = MahalanobisPSCaliperMatcher(model=_model(), caliper=0.5).fit(x, treatment)
    loose.transform(x, treatment, y)
    tight = MahalanobisPSCaliperMatcher(model=_model(), caliper=0.01).fit(x, treatment)
    tight.transform(x, treatment, y)
    assert tight.unmatched_treated_.size >= loose.unmatched_treated_.size


def test_n_neighbors_and_no_replacement(uplift_data):
    x, treatment, y = uplift_data
    matched = MahalanobisPSCaliperMatcher(
        model=_model(), caliper=0.3, n_neighbors=2, replace=False,
    ).fit_transform(x, treatment, y)
    controls = matched.loc[matched['treatment'] == 0]
    # without replacement each control is used once → weight 1/n_neighbors
    assert np.allclose(controls['weight'].to_numpy(), 0.5)


def test_requires_predict_proba():
    with pytest.raises(TypeError):
        MahalanobisPSCaliperMatcher(model=object(), caliper=0.1)
