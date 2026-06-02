# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - unreleased

Initial public release.

### Added

- **`UpliftForecast`** orchestrator — fits a list of models on shared data, collects
  predictions into one DataFrame, with `save` / `load`.
- **`UpliftModel`** unified interface (`fit`, `predict`, `display_name`) implemented by
  every model and framework.
- **Neural models** (subclass `BaseNeuralUpliftModel`, PyTorch Lightning): `DragonNet`,
  `CFRNet`, `TARNet`.
- **Meta-learners** (subclass `BaseMetaUpliftModel`, sklearn-style): `SLearner`, `TLearner`,
  `XLearner`, `RLearner`, `DRLearner`, `ZLearner`, `CausalForest`, `PolicyForest`,
  `PolicyLearner`.
- **Covariate matching** module — seven matchers with balance diagnostics.
- **`RERUM`** framework — rankability-enhanced revenue uplift, retargeting a base neural
  model's objective.
- **Losses** (`uplift_forecast.losses`): `MSELoss`, `MAELoss`, `PseudoPEHE`, `DragonNetLoss`,
  `RERUMLoss`, `CFRLoss`, plus ZILN / IPM / ranking helpers.
- **Metrics** (`uplift_forecast.metrics`): `auuc_score`, `qini_score`, `cumulative_gain_curve`,
  `qini_curve`, `uplift_component_{mae,mse,mape}`.
- **Hyperparameter search** (`uplift_forecast.auto`): Optuna-based `OptunaHyperparameterTuner`
  with `Suggest*` helpers (lazy optuna import).
- Supports Python 3.10–3.12.
