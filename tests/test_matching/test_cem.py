import numpy as np
import pandas as pd
import pytest

from uplift_forecast.matching import CoarsenedExactMatcher


def test_fit_transform_smoke(uplift_data):
    x, treatment, y = uplift_data
    matched = CoarsenedExactMatcher(n_bins=3).fit_transform(x, treatment, y)

    assert isinstance(matched, pd.DataFrame)
    assert {'treatment', 'y', 'weight', 'stratum'} <= set(matched.columns)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()


def test_each_stratum_has_both_arms(uplift_data):
    x, treatment, y = uplift_data
    matched = CoarsenedExactMatcher(n_bins=3).fit_transform(x, treatment, y)
    for _, group in matched.groupby('stratum'):
        assert (group['treatment'] == 1).any()
        assert (group['treatment'] == 0).any()


def test_att_weights_balance_per_stratum(uplift_data):
    x, treatment, y = uplift_data
    matched = CoarsenedExactMatcher(n_bins=3).fit_transform(x, treatment, y)
    for _, group in matched.groupby('stratum'):
        treated = group.loc[group['treatment'] == 1, 'weight'].sum()
        control = group.loc[group['treatment'] == 0, 'weight'].sum()
        assert treated == pytest.approx(control)


def test_diagnostics_populated(uplift_data):
    x, treatment, y = uplift_data
    m = CoarsenedExactMatcher(n_bins=4).fit(x, treatment)
    m.transform(x, treatment, y)
    assert m.n_strata_ is not None and m.n_strata_ > 0
    assert m.n_strata_dropped_ is not None
    assert 0.0 <= m.match_rate_ <= 1.0


def test_user_bin_edges(uplift_data):
    x, treatment, y = uplift_data
    edges = {0: np.array([-10.0, 0.0, 10.0])}
    matched = CoarsenedExactMatcher(bin_edges=edges).fit_transform(x, treatment, y)
    assert len(matched) > 0


def test_categorical_dataframe_column():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        'num': rng.normal(size=n),
        'cat': rng.choice(['a', 'b', 'c'], size=n),
    })
    t = rng.integers(0, 2, n)
    y = rng.normal(size=n)
    matched = CoarsenedExactMatcher(n_bins=3).fit_transform(df, t, y)
    assert {'num', 'cat'} <= set(matched.columns)
    assert len(matched) > 0

    # pandas 'category' dtype must be inferred as categorical without raising
    df_cat = df.assign(cat=df['cat'].astype('category'))
    assert len(CoarsenedExactMatcher(n_bins=3).fit_transform(df_cat, t, y)) > 0


def test_rare_category_grouping():
    rng = np.random.default_rng(1)
    n = 500
    cats = np.array(['common'] * (n - 10) + ['rare'] * 10)
    rng.shuffle(cats)
    df = pd.DataFrame({'num': rng.normal(size=n), 'cat': cats})
    t = rng.integers(0, 2, n)
    m = CoarsenedExactMatcher(n_bins=2, rare_threshold=0.1).fit(df, t)
    assert 'rare' not in m._kept_categories['cat']
