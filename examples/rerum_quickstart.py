"""Train RERUM on synthetic revenue-uplift data and report ranking metrics.

Mirrors the train/test + AUUC/Qini flow of the legacy ``main.py`` through the
new ``UpliftForecast`` / ``RERUM`` API, on a self-contained synthetic dataset
(no project parquet files or Hydra config).

Run:
    python examples/rerum_quickstart.py
"""

import numpy as np

from uplift_forecast import RERUM, UpliftForecast
from uplift_forecast.metrics import auuc_score, qini_score
from uplift_forecast.models import CFRNet, DragonNet, TARNet


def make_dataset(n: int = 4000, n_features: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-inflated, long-tailed revenue response with a heterogeneous uplift."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype('float32')
    treatment = rng.integers(0, 2, size=n).astype('float32')
    # baseline spend (lognormal-ish) and a treatment effect that grows with x[:, 1]
    base = np.exp(0.5 * x[:, 0] + rng.normal(scale=0.5, size=n))
    effect = treatment * np.clip(x[:, 1], 0.0, None) * 2.0
    y = (base + effect).astype('float32')
    y[rng.random(n) < 0.4] = 0.0  # non-payers
    return x, treatment, y


def train_test_split(
    x: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    *,
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[tuple, tuple]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return (x[tr], treatment[tr], y[tr]), (x[te], treatment[te], y[te])


def main() -> None:
    x, treatment, y = make_dataset()
    (x_tr, t_tr, y_tr), (x_te, t_te, y_te) = train_test_split(x, treatment, y)

    # Gradient clipping keeps the ZILN + ranking objective numerically stable.
    trainer_kwargs = {
        'max_epochs': 15,
        'accelerator': 'cpu',
        'gradient_clip_val': 1.0,
        'logger': False,
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }

    forecast = UpliftForecast(
        models=[
            RERUM(model=DragonNet(input_size=x.shape[1], batch_size=512, **trainer_kwargs), alias='rerum_dragonnet'),
            RERUM(model=CFRNet(input_size=x.shape[1], batch_size=512, **trainer_kwargs), alias='rerum_cfrnet'),
            RERUM(model=TARNet(input_size=x.shape[1], batch_size=512, **trainer_kwargs), alias='rerum_tarnet'),
        ],
    )
    forecast.fit(x_tr, t_tr, y_tr)

    preds = forecast.predict(x_te)
    for col in preds.columns:
        uplift = preds[col].to_numpy()
        auuc = auuc_score(y_te, uplift, t_te)
        qini = qini_score(y_te, uplift, t_te)
        print(f'{col:>28s}  AUUC={auuc:.4f}  Qini={qini:.4f}')


if __name__ == '__main__':
    main()
