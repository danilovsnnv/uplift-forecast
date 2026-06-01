"""A first look at uplift-forecast: meta-learners and neural models side by side.

Written from the point of view of someone new to uplift modeling. We generate a
small synthetic dataset with a known heterogeneous treatment effect, fit a few
models through the single ``UpliftForecast`` entry point, and compare them with
AUUC / Qini. Finally we round-trip the fitted models through ``save`` / ``load``.

Run:
    PYTHONPATH=. python examples/quickstart.py
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast
from uplift_forecast.metrics import auuc_score, qini_score
from uplift_forecast.models import DragonNet, SLearner, TARNet, TLearner


def make_dataset(n: int = 6000, n_features: int = 6, seed: int = 0):
    """Continuous outcome with a treatment effect that grows with feature 1.

    The true individual uplift is ``tau(x) = 3 * relu(x[:, 1])`` so we know which
    units should be ranked highest by a good model.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype('float32')
    treatment = rng.integers(0, 2, size=n).astype('float32')
    baseline = 2.0 * x[:, 0] + x[:, 2]
    tau = 3.0 * np.clip(x[:, 1], 0.0, None)
    noise = rng.normal(scale=1.0, size=n)
    y = (baseline + treatment * tau + noise).astype('float32')
    return x, treatment, y, tau


def split(x, treatment, y, *, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return tr, te


def main() -> None:
    x, treatment, y, tau = make_dataset()
    tr, te = split(x, treatment, y)
    x_tr, t_tr, y_tr = x[tr], treatment[tr], y[tr]
    x_te, t_te, y_te = x[te], treatment[te], y[te]

    trainer_kwargs = {
        'max_epochs': 20,
        'accelerator': 'cpu',
        'gradient_clip_val': 1.0,
        'logger': False,
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }

    forecast = UpliftForecast(
        models=[
            SLearner(GradientBoostingRegressor(random_state=0), alias='slearner_gbm'),
            TLearner(GradientBoostingRegressor(random_state=0), alias='tlearner_gbm'),
            TARNet(input_size=x.shape[1], batch_size=512, alias='tarnet', **trainer_kwargs),
            DragonNet(input_size=x.shape[1], batch_size=512, alias='dragonnet', **trainer_kwargs),
        ],
    )
    forecast.fit(x_tr, t_tr, y_tr)

    preds = forecast.predict(x_te, return_components=True)

    print(f'{"model":>16s}  {"AUUC":>7s}  {"Qini":>7s}  {"corr(tau)":>9s}')
    for model in forecast.models:
        name = model.display_name
        uplift = preds[f'uplift_{name}'].to_numpy()
        auuc = auuc_score(y_te, uplift, t_te)
        qini = qini_score(y_te, uplift, t_te)
        corr = np.corrcoef(uplift, tau[te])[0, 1]
        print(f'{name:>16s}  {auuc:7.4f}  {qini:7.4f}  {corr:9.4f}')

    # save / load round-trip
    forecast.save('checkpoints/quickstart.pkl')
    reloaded = UpliftForecast.load('checkpoints/quickstart.pkl')
    reloaded_preds = reloaded.predict(x_te)
    print('\nsave/load OK — reloaded columns:', list(reloaded_preds.columns))


if __name__ == '__main__':
    main()
