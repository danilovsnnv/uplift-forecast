"""Optuna-based hyperparameter search helpers.

`Suggest*` classes describe per-parameter search spaces in a declarative way:

    space = {
        'hidden_size': SuggestCategorical([64, 128, 256]),
        'learning_rate': SuggestFloat(1e-4, 1e-2, log=True),
        'max_epochs': SuggestInt(20, 200, step=10),
    }

Anything that isn't a `Suggest` is treated as a constant for every trial.

`OptunaHyperparameterTuner` wires this declarative space into an Optuna study.
The objective callable receives a list of param dicts (one per search space).

Optuna is loaded lazily so the rest of `uplift_forecast` does not pay the
import cost — install the optional dependency `pip install uplift-forecast[auto]`
to use this module.
"""

from collections.abc import Sequence
from typing import Any

__all__ = [
    'OptunaHyperparameterTuner',
    'Suggest',
    'SuggestCategorical',
    'SuggestFloat',
    'SuggestInt',
    'const_trial_params_split',
]


def _require_optuna():
    try:
        import optuna
    except ImportError as err:
        raise ImportError(
            'optuna is required for uplift_forecast.auto. '
            'Install with `pip install uplift-forecast[auto]`.',
        ) from err
    return optuna


class Suggest:
    """Base wrapper around an `optuna.Trial.suggest_*` call."""

    def __init__(self, func, **params):
        self.func = func
        self.params = params

    def __call__(self, trial, name: str):
        return self.func(trial, name=name, **self.params)


class SuggestCategorical(Suggest):
    def __init__(self, choices: Sequence):
        optuna = _require_optuna()
        super().__init__(func=optuna.Trial.suggest_categorical, choices=choices)


class SuggestFloat(Suggest):
    def __init__(self, low: float, high: float, *, step: float | None = None, log: bool = False):
        optuna = _require_optuna()
        super().__init__(func=optuna.Trial.suggest_float, low=low, high=high, step=step, log=log)


class SuggestInt(Suggest):
    def __init__(self, low: int, high: int, *, step: int = 1, log: bool = False):
        optuna = _require_optuna()
        super().__init__(func=optuna.Trial.suggest_int, low=low, high=high, step=step, log=log)


def const_trial_params_split(*params: dict) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Split each dict into `(constants, suggest_spaces)`."""

    def split(params):
        for param in params:
            const = {k: v for k, v in param.items() if not isinstance(v, Suggest)}
            space = {k: v for k, v in param.items() if isinstance(v, Suggest)}
            yield const, space

    const_params, params_spaces = zip(*split(params), strict=False)
    return const_params, params_spaces


class OptunaHyperparameterTuner:
    """Optuna sweep over one or more search spaces.

    Args:
        func: Objective callable. Receives one or more parameter dicts (one
            per `params_space`) and returns the score(s) Optuna should
            optimise.
        params_space: First search space. Mix `Suggest*` entries with constants.
        *other_params_spaces: Additional spaces (named `0_`, `1_`, ... inside
            Optuna).
        n_trials: Number of trials.
        direction: `'maximize'` or `'minimize'`.
        verbose: Compatibility shim — Optuna's own logging level is more
            granular; ignored here.
    """

    def __init__(
        self,
        func,
        params_space: dict,
        *other_params_spaces: dict,
        n_trials: int = 100,
        direction: str = 'maximize',
        verbose: int = 1,
    ):
        self.func = func
        self.params_space = params_space
        self.other_params_spaces = other_params_spaces

        self.const_params, self.params_spaces = const_trial_params_split(
            params_space, *other_params_spaces,
        )
        self.n_spaces = len(self.params_spaces)

        self.study = None
        self.n_trials = n_trials
        self.direction = direction
        self.verbose = verbose

    def _get_single_trial_params(self, trial):
        trial_params = {name: fn(trial, name=name) for name, fn in self.params_spaces[0].items()}
        yield self.const_params[0] | trial_params

    def _get_multiple_trial_params(self, trial):
        for i, (const_params, space) in enumerate(
            zip(self.const_params, self.params_spaces, strict=False),
        ):
            trial_params = {
                name: fn(trial, name=f'{i}_{name}') for name, fn in space.items()
            }
            yield const_params | trial_params

    def optimize(self, study=None, **kwargs) -> 'OptunaHyperparameterTuner':
        optuna = _require_optuna()

        def objective(trial):
            params_getter = (
                self._get_multiple_trial_params if self.n_spaces > 1 else self._get_single_trial_params
            )
            return self.func(*params_getter(trial), **kwargs)

        self.study = study or optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials)
        return self

    @property
    def best_params(self) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        if self.study is None:
            raise RuntimeError('Call .optimize() before reading best_params.')
        params = self.study.best_params
        if self.n_spaces == 1:
            return params
        prefixes = [f'{i}_' for i in range(self.n_spaces)]
        return tuple(
            {k.removeprefix(prefix): v for k, v in params.items() if k.startswith(prefix)}
            for prefix in prefixes
        )

    @property
    def has_completed_trials(self) -> bool:
        if self.study is None:
            return False
        optuna = _require_optuna()
        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        return len(completed) > 0
