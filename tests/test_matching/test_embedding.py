import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA

from uplift_forecast.matching import EmbeddingMatcher


def test_identity_encoder(uplift_data):
    x, treatment, y = uplift_data
    matched = EmbeddingMatcher(encoder=None, n_neighbors=2).fit_transform(x, treatment, y)
    assert isinstance(matched, pd.DataFrame)
    assert {'treatment', 'y', 'weight'} <= set(matched.columns)
    assert len(matched) > 0
    assert (matched['weight'] > 0).all()


def test_pca_encoder(uplift_data):
    x, treatment, y = uplift_data
    m = EmbeddingMatcher(encoder=PCA(n_components=3), n_neighbors=2)
    matched = m.fit_transform(x, treatment, y)
    assert m._embed(x).shape[1] == 3
    assert len(matched) > 0


def test_callable_encoder(uplift_data):
    x, treatment, y = uplift_data
    matched = EmbeddingMatcher(encoder=lambda a: np.asarray(a)[:, :2]).fit_transform(x, treatment, y)
    assert len(matched) > 0


def test_torch_module_encoder(uplift_data):
    x, treatment, y = uplift_data
    encoder = torch.nn.Linear(6, 4)
    matched = EmbeddingMatcher(encoder=encoder, n_neighbors=2).fit_transform(x, treatment, y)
    assert len(matched) > 0


def test_unsupported_encoder_raises(uplift_data):
    x, treatment, y = uplift_data
    m = EmbeddingMatcher(encoder=12345)
    with pytest.raises(TypeError):
        m.fit_transform(x, treatment, y)
