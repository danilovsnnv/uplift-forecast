"""Tour of the roadmap additions: diagnostics, OPE, explainability, model selection,
modern neural models, multi-treatment, and continuous-dose response.

Everything runs through the same ``UpliftModel`` contract, so the new pieces slot
into the existing workflow. Kept small and CPU-only so it runs in a few seconds.

Run:
    PYTHONPATH=. python examples/roadmap_extras.py
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from uplift_forecast import UpliftForecast, diagnostics, explain, ope
from uplift_forecast.auto import AutoUplift
from uplift_forecast.metrics import (
    auuc_score,
    best_dose,
    multi_arm_auuc_scores,
    optimal_treatment_assignment,
)
from uplift_forecast.models import (
    EFIN,
    VCNet,
    CausalForest,
    MultiTLearner,
    SLearner,
    TLearner,
)

TRAINER_KWARGS = {
    'max_epochs': 15,
    'accelerator': 'cpu',
    'gradient_clip_val': 1.0,
    'logger': False,
    'enable_progress_bar': False,
    'enable_model_summary': False,
    'enable_checkpointing': False,
}


def binary_dataset(n: int = 4000, n_features: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype('float32')
    treatment = rng.integers(0, 2, size=n).astype('float32')
    tau = 3.0 * np.clip(x[:, 1], 0.0, None)
    y = (2.0 * x[:, 0] + treatment * tau + rng.normal(size=n)).astype('float32')
    return x, treatment, y


def main() -> None:
    x, treatment, y = binary_dataset()
    propensity = np.full(len(treatment), treatment.mean())

    print('== Topic 7: design diagnostics ==')
    report = diagnostics.overlap_report(x, treatment, propensity=propensity)
    print(f'balanced={report["balanced"]}  max|SMD|={report["max_abs_smd"]:.3f}  '
          f'share_outside_overlap={report["positivity"]["share_outside_overlap"]:.3f}')

    print('\n== Topic 9: AutoUplift model selection ==')
    auto = AutoUplift(
        candidates=[
            TLearner(GradientBoostingRegressor(random_state=0), alias='tlearner'),
            SLearner(GradientBoostingRegressor(random_state=0), alias='slearner'),
            CausalForest(n_estimators=50, random_state=0, alias='causal_forest'),
            EFIN(input_size=x.shape[1], batch_size=512, alias='efin', **TRAINER_KWARGS),
        ],
        metric='qini',
        random_state=0,
    ).fit(x, treatment, y)
    print(auto.leaderboard_.to_string(index=False))

    print('\n== Topic 6: off-policy evaluation of the selected policy ==')
    for estimator in ('ips', 'snips', 'dr'):
        value = ope.evaluate_policy(auto.best_model_, x, treatment, y, propensity, estimator=estimator)
        print(f'{estimator:>6s} policy value = {value:.4f}')

    print('\n== Topic 8: AUUC permutation importance for the uplift signal ==')
    importance = explain.permutation_importance(auto.best_model_, x, treatment, y, metric='auuc', n_repeats=3)
    print(importance.head(3).to_string())

    print('\n== Topic 3: multi-treatment (3 arms) ==')
    rng = np.random.default_rng(1)
    arm = rng.integers(0, 3, size=len(x))
    y_multi = (2.0 * x[:, 0] + (arm == 1) * x[:, 1] + (arm == 2) * 2.0 * np.clip(x[:, 2], 0, None)
               + rng.normal(size=len(x))).astype('float32')
    multi = MultiTLearner(GradientBoostingRegressor(random_state=0)).fit(x, arm, y_multi)
    uplift_multi = multi.predict(x)
    print('per-arm AUUC:', {k: round(v, 4) for k, v in multi_arm_auuc_scores(y_multi, uplift_multi, arm).items()})
    assignment = optimal_treatment_assignment(uplift_multi, costs=[0.2, 0.5])
    print('cost-aware assignment counts:', dict(zip(*np.unique(assignment, return_counts=True), strict=True)))

    print('\n== Topic 4: continuous dose-response (VCNet) ==')
    dose = rng.uniform(0, 1, size=len(x)).astype('float32')
    y_dose = (2.0 * x[:, 0] + dose * (2.0 + x[:, 1]) + rng.normal(size=len(x))).astype('float32')
    vcnet = VCNet(input_size=x.shape[1], batch_size=512, **TRAINER_KWARGS).fit(x, dose, y_dose)
    grid = np.linspace(0, 1, 11)
    curves = vcnet.predict_dose_response(x[:200], grid)
    print('mean predicted best dose:', round(float(best_dose(curves, grid).mean()), 3))

    print('\n== single AUUC sanity check on the best model ==')
    forecast = UpliftForecast([auto])
    best_uplift = forecast.predict(x)['uplift_AutoUplift'].to_numpy()
    print('AUUC(best):', round(auuc_score(y, best_uplift, treatment), 4))


if __name__ == '__main__':
    main()
