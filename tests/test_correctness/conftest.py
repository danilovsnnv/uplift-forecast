import numpy as np
import pytest


@pytest.fixture
def known_cate_data():
    """Synthetic RCT with an analytically known CATE tau(x) = 2 * x[:, 0].

    Treatment is randomised (constant propensity 0.5) and the outcome noise is small,
    so a correct estimator should recover tau closely. The large n suppresses sampling
    variance, which keeps the per-model MAE checks stable across runs and Python versions.

    Returns:
        Tuple of (X, treatment, y, true_cate), where true_cate is the ground-truth CATE.
    """
    rng = np.random.default_rng(0)
    n, n_features = 4000, 5
    x = rng.normal(size=(n, n_features))
    true_cate = 2.0 * x[:, 0]
    prognostic = x[:, 1] + 0.5 * x[:, 2]  # baseline outcome shared by both arms
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    y = prognostic + treatment * true_cate + rng.normal(scale=0.1, size=n)
    return x, treatment, y, true_cate
