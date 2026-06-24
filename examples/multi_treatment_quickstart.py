"""Multi-treatment uplift end to end: fit, per-arm metrics, compare, assign, evaluate.

Each unit can receive one of K discrete treatments (arm 0 = control). We generate a
3-arm dataset with known per-arm effects, fit meta-learners and a deep model that all
return one uplift column per treated arm, compare them per arm, decompose the predicted
outcomes, pick the best arm per unit, and score the induced policy off-policy.

Run:
    PYTHONPATH=. python examples/multi_treatment_quickstart.py
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from uplift_forecast import UpliftForecast
from uplift_forecast import evaluation, metrics, ope
from uplift_forecast.models import M3TN, TLearner, XLearner


def make_dataset(n: int = 6000, n_features: int = 6, n_arms: int = 3, seed: int = 0):
    """Continuous outcome where each treated arm has its own heterogeneous effect.

    Arm 1's effect grows with feature 1; arm 2's with feature 2 (half as strong). The
    returned ``tau`` is the ``[n, n_arms-1]`` true per-arm uplift vs control.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype('float32')
    treatment = rng.integers(0, n_arms, size=n)
    baseline = 2.0 * x[:, 0] + x[:, 2]
    arm_effect = np.column_stack([3.0 * np.clip(x[:, 1], 0.0, None), 1.5 * np.clip(x[:, 2], 0.0, None)])
    realised = np.where(treatment == 0, 0.0, arm_effect[np.arange(n), np.clip(treatment - 1, 0, None)])
    y = (baseline + realised + rng.normal(size=n)).astype('float32')
    return x, treatment, y, arm_effect


def main() -> None:
    x, treatment, y, tau = make_dataset()
    cut = int(0.7 * len(x))
    x_tr, t_tr, y_tr = x[:cut], treatment[:cut], y[:cut]
    x_te, t_te, y_te, tau_te = x[cut:], treatment[cut:], y[cut:], tau[cut:]

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
            TLearner(GradientBoostingRegressor(random_state=0), alias='tlearner'),
            XLearner(
                GradientBoostingRegressor(random_state=0),
                propensity_model=LogisticRegression(max_iter=1000),
                alias='xlearner',
            ),
            M3TN(input_size=x.shape[1], n_treatments=3, batch_size=512, alias='m3tn', **trainer_kwargs),
        ],
    )
    forecast.fit(x_tr, t_tr.astype('float32'), y_tr)

    # 1) Decompose: per-arm uplift columns plus the predicted outcome components.
    preds = forecast.predict(x_te, return_components=True)
    print('prediction columns:', list(preds.columns))

    # 2) Compare every model per treatment arm in one table.
    print('\nper-model, per-arm ranking metrics:')
    print(evaluation.compare_models(forecast, x_te, y_te, t_te).to_string(index=False))

    # 3) PEHE against the known effect, aggregated across arms (mPEHE / sdPEHE).
    print('\nmPEHE / sdPEHE (lower is better):')
    for model in forecast.models:
        uplift = model.predict(x_te)
        m_pehe, sd_pehe = metrics.arm_score_summary(metrics.pehe(tau_te, uplift))
        print(f'  {model.display_name:>10s}  mPEHE={m_pehe:6.3f}  sdPEHE={sd_pehe:6.3f}')

    # 4) Best arm per unit (argmax uplift) and the induced policy's off-policy value.
    best = forecast.models[0]
    uplift = best.predict(x_te)
    assignment = metrics.optimal_treatment_assignment(uplift)
    print(f'\n{best.display_name}: assigned arm counts =', dict(zip(*np.unique(assignment, return_counts=True), strict=True)))

    gps = np.full((len(t_te), 3), 1 / 3.0)  # uniform RCT generalized propensity
    p_taken = gps[np.arange(len(t_te)), t_te]
    print(f'  expected response of best-arm policy = {evaluation.expected_response(y_te, assignment, t_te, p_taken):.4f}')
    print(f'  doubly-robust policy value           = {ope.evaluate_policy(best, x_te, t_te, y_te, gps, estimator="dr"):.4f}')


if __name__ == '__main__':
    main()
