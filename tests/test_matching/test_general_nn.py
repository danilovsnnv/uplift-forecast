import numpy as np
import pandas as pd
import pytest

from uplift_forecast.matching import NearestNeighborMatcher


@pytest.mark.parametrize('metric', ['euclidean', 'manhattan', 'cosine', 'mahalanobis'])
def test_metrics_smoke(uplift_data, metric):
    x, treatment, y = uplift_data
    matched = NearestNeighborMatcher(metric=metric).fit_transform(x, treatment, y)
    assert isinstance(matched, pd.DataFrame)
    assert {'treatment', 'y', 'weight'} <= set(matched.columns)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()


def test_knn_multiple_neighbors(uplift_data):
    x, treatment, y = uplift_data
    matched = NearestNeighborMatcher(n_neighbors=3).fit_transform(x, treatment, y)
    assert (matched.loc[matched['treatment'] == 1, 'weight'] == 1.0).all()


def test_without_replacement_uses_each_control_once(uplift_data):
    x, treatment, y = uplift_data
    matched = NearestNeighborMatcher(replace=False).fit_transform(x, treatment, y)
    controls = matched.loc[matched['treatment'] == 0]
    assert (controls['weight'] == 1.0).all()


def test_radius_mode(uplift_data):
    x, treatment, y = uplift_data
    matched = NearestNeighborMatcher(radius=3.0).fit_transform(x, treatment, y)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()


def test_custom_callable_metric(uplift_data):
    x, treatment, y = uplift_data

    def l1(a, b):
        return float(np.abs(a - b).sum())

    matched = NearestNeighborMatcher(metric=l1).fit_transform(x, treatment, y)
    assert len(matched) > 0


def test_custom_backend(uplift_data):
    x, treatment, y = uplift_data
    from uplift_forecast.matching._backends import SklearnNeighborBackend

    matched = NearestNeighborMatcher(backend=SklearnNeighborBackend(metric='euclidean')).fit_transform(x, treatment, y)
    assert len(matched) > 0


def test_radius_and_caliper_conflict():
    with pytest.raises(ValueError):
        NearestNeighborMatcher(radius=1.0, caliper=0.5)
