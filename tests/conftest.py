import warnings

import numpy as np
import pytest


@pytest.fixture
def uplift_data():
    """Synthetic zero-inflated revenue uplift data: (X, treatment, y)."""
    rng = np.random.default_rng(0)
    n, n_features = 300, 6
    x = rng.normal(size=(n, n_features)).astype('float32')
    treatment = rng.integers(0, 2, size=n).astype('float32')
    base = np.clip(x[:, 0] * 2.0 + rng.normal(size=n), 0.0, None)
    y = (base + treatment * np.clip(x[:, 1], 0.0, None)).astype('float32')
    y[rng.random(n) < 0.3] = 0.0
    return x, treatment, y


@pytest.fixture
def trainer_kwargs():
    """Fast, deterministic, CPU-only Lightning settings for smoke tests.

    Gradient clipping keeps the ZILN + ranking objective numerically stable.
    """
    return {
        'max_epochs': 3,
        'accelerator': 'cpu',
        'logger': False,
        'enable_progress_bar': False,
        'enable_model_summary': False,
        'enable_checkpointing': False,
        'gradient_clip_val': 1.0,
    }


@pytest.fixture(autouse=True)
def _quiet() -> None:
    warnings.filterwarnings('ignore')
