import numpy as np
import pandas as pd
import pytest

from uplift_forecast.matching import KernelMatcher


@pytest.mark.parametrize('kernel', ['gaussian', 'epanechnikov', 'triangular', 'uniform'])
def test_kernels_smoke(uplift_data, kernel):
    x, treatment, y = uplift_data
    matched = KernelMatcher(kernel=kernel, candidate_mode='knn', n_neighbors=20).fit_transform(x, treatment, y)
    assert isinstance(matched, pd.DataFrame)
    assert {'treatment', 'y', 'weight'} <= set(matched.columns)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()


@pytest.mark.parametrize('mode,kw', [
    ('all', {}),
    ('knn', {'n_neighbors': 15}),
    ('radius', {'radius': 3.0}),
])
def test_candidate_modes(uplift_data, mode, kw):
    x, treatment, y = uplift_data
    matched = KernelMatcher(candidate_mode=mode, **kw).fit_transform(x, treatment, y)
    assert len(matched) > 0


def test_explicit_and_auto_bandwidth(uplift_data):
    x, treatment, y = uplift_data
    assert len(KernelMatcher(bandwidth=1.5).fit_transform(x, treatment, y)) > 0
    assert len(KernelMatcher(bandwidth='auto').fit_transform(x, treatment, y)) > 0


def test_normalized_weights_sum_to_one_per_treated(uplift_data):
    x, treatment, y = uplift_data
    m = KernelMatcher(candidate_mode='knn', n_neighbors=10, normalize=True, return_weight_matrix=True)
    _, matrix = m.fit_transform(x, treatment, y)
    per_treated: dict[int, float] = {}
    for ti, _, w in matrix:
        per_treated[ti] = per_treated.get(ti, 0.0) + w
    assert all(abs(total - 1.0) < 1e-6 for total in per_treated.values())


def test_return_weight_matrix(uplift_data):
    x, treatment, y = uplift_data
    out = KernelMatcher(candidate_mode='knn', n_neighbors=5, return_weight_matrix=True).fit_transform(x, treatment, y)
    assert isinstance(out, tuple) and len(out) == 2
    matched, matrix = out
    assert isinstance(matched, pd.DataFrame)
    assert isinstance(matrix, list) and len(matrix) > 0 and len(matrix[0]) == 3


def test_unknown_kernel_raises():
    with pytest.raises(ValueError):
        KernelMatcher(kernel='nope')
