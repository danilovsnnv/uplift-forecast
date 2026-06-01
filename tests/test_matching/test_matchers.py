import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from uplift_forecast.matching import MahalanobisMatcher, PropensityScoreMatcher


def _matchers():
    return [
        PropensityScoreMatcher(model=LogisticRegression(max_iter=200)),
        MahalanobisMatcher(),
    ]


@pytest.mark.parametrize('matcher', _matchers())
def test_fit_transform_smoke(uplift_data, matcher):
    x, treatment, y = uplift_data
    matched = matcher.fit_transform(x, treatment, y)

    assert isinstance(matched, pd.DataFrame)
    assert {'treatment', 'y', 'weight'} <= set(matched.columns)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()
    # every kept treated unit balanced by an equal control weight mass
    treated = matched.loc[matched['treatment'] == 1, 'weight'].sum()
    control = matched.loc[matched['treatment'] == 0, 'weight'].sum()
    assert treated == pytest.approx(control)


@pytest.mark.parametrize('matcher', _matchers())
def test_transform_after_fit(uplift_data, matcher):
    x, treatment, y = uplift_data
    matcher.fit(x, treatment)
    out = matcher.transform(x, treatment)
    assert 'y' not in out.columns
    assert 'weight' in out.columns


def test_preserves_dataframe_columns(uplift_data):
    x, treatment, y = uplift_data
    df = pd.DataFrame(x, columns=[f'col_{i}' for i in range(x.shape[1])])
    matched = MahalanobisMatcher().fit_transform(df, treatment, y)
    assert [c for c in matched.columns if c.startswith('col_')] == list(df.columns)


def test_without_replacement_uses_each_control_once(uplift_data):
    x, treatment, y = uplift_data
    matched = MahalanobisMatcher(replace=False).fit_transform(x, treatment, y)
    controls = matched.loc[matched['treatment'] == 0]
    assert (controls['weight'] == 1.0).all()


def test_caliper_too_tight_raises(uplift_data):
    x, treatment, y = uplift_data
    with pytest.raises(ValueError):
        MahalanobisMatcher(caliper=1e-9).fit_transform(x, treatment, y)


def test_propensity_requires_predict_proba():
    with pytest.raises(TypeError):
        PropensityScoreMatcher(model=object())
