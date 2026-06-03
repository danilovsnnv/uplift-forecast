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

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .common._uplift_model import UpliftModel, _row_subset, _to_array, _to_numpy_1d
from .metrics import auuc_score, qini_score

__all__ = [
    'AutoUplift',
    'OptunaHyperparameterTuner',
    'Suggest',
    'SuggestCategorical',
    'SuggestFloat',
    'SuggestInt',
    'const_trial_params_split',
]

_SELECTION_METRICS = {'auuc': auuc_score, 'qini': qini_score}


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


# Candidate spec: a ready UpliftModel instance (used as-is), or a
# (factory, space) pair where `factory(**params) -> UpliftModel` and `space`
# mixes constants with `Suggest*` entries to tune (same convention as the tuner).
Candidate = UpliftModel | tuple[Callable[..., UpliftModel], dict]


class AutoUplift(UpliftModel):
    """Tune and select across a candidate set of uplift models on a uplift metric.

    Each candidate is fitted on a training split and scored on a validation split
    by AUUC or Qini; candidates carrying a `Suggest*` space are first tuned with
    `OptunaHyperparameterTuner` over that space. The leaderboard ranks them, and
    the best (or a top-k average ensemble) is refitted on the full data and used
    for prediction. Implements `UpliftModel`, so it drops into `UpliftForecast`
    and supports `save`/`load`.

    Per the "no stable SOTA" benchmark caveat, prefer a diverse candidate set
    (meta-learner + tree/forest + 1-2 neural) and read the full `leaderboard_`,
    not just the winner.

    Args:
        candidates: List of `UpliftModel` instances and/or `(factory, space)` pairs.
            A bare instance is fitted as-is; a `(factory, space)` pair is tuned by
            building `factory(**params)` over `space` (Optuna, loaded lazily).
        metric: Selection metric — `'auuc'` or `'qini'` (validation, normalized).
        n_trials: Optuna trials per tuned candidate.
        validation_size: Held-out fraction when no `eval_set` is passed to `fit`.
        top_k: Average the top-k models' predictions (1 = pick the single best).
        random_state: Seed for the train/validation split.
        alias: Display name used by `UpliftForecast`.
    """

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        metric: str = 'qini',
        n_trials: int = 50,
        validation_size: float = 0.25,
        top_k: int = 1,
        random_state: int = 0,
        alias: str | None = None,
    ):
        if metric not in _SELECTION_METRICS:
            raise ValueError(f'metric must be one of {sorted(_SELECTION_METRICS)}; got {metric!r}.')
        if not candidates:
            raise ValueError('AutoUplift needs at least one candidate.')
        if top_k < 1:
            raise ValueError('top_k must be >= 1.')
        self.candidates = list(candidates)
        self.metric = metric
        self.n_trials = n_trials
        self.validation_size = validation_size
        self.top_k = top_k
        self.random_state = random_state
        self.alias = alias

        self._score_fn = _SELECTION_METRICS[metric]
        self.leaderboard_: pd.DataFrame | None = None
        self.best_model_: UpliftModel | None = None
        self._ensemble: list[UpliftModel] = []

    @staticmethod
    def _as_factory_space(candidate: Candidate) -> tuple[Callable[..., UpliftModel], dict]:
        if isinstance(candidate, UpliftModel):
            return (lambda c=candidate, **_: deepcopy(c)), {}
        factory, space = candidate
        return factory, dict(space)

    def _make_split(self, X: Any, t: np.ndarray, y: np.ndarray, eval_set: tuple | None):
        if eval_set is not None:
            xv, tv, yv = eval_set
            return X, t, y, _to_array(xv), _to_numpy_1d(tv), _to_numpy_1d(yv)
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(t))
        train_idx, val_idx = train_test_split(
            idx, test_size=self.validation_size, random_state=self.random_state, stratify=t,
        )
        return (
            _row_subset(X, train_idx), t[train_idx], y[train_idx],
            _row_subset(X, val_idx), t[val_idx], y[val_idx],
        )

    def _fit_score(self, model: UpliftModel, train: tuple, val: tuple) -> float:
        xt, tt, yt = train
        xv, tv, yv = val
        model.fit(xt, tt, yt)
        uplift = np.asarray(model.predict(xv)).reshape(-1)
        return float(self._score_fn(yv, uplift, tv))

    def _tune_candidate(self, factory: Callable[..., UpliftModel], space: dict, train: tuple, val: tuple) -> dict:
        const = {k: v for k, v in space.items() if not isinstance(v, Suggest)}
        if not any(isinstance(v, Suggest) for v in space.values()):
            return const
        tuner = OptunaHyperparameterTuner(
            func=lambda params: self._fit_score(factory(**params), train, val),
            params_space=space,
            n_trials=self.n_trials,
            direction='maximize',
        )
        tuner.optimize()
        return const | tuner.best_params

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **_: Any,
    ) -> 'AutoUplift':
        """Tune/score every candidate, build the leaderboard, refit the winner(s)."""
        x_arr = _to_array(X)
        t = _to_numpy_1d(treatment)
        y_arr = _to_numpy_1d(y)
        xt, tt, yt, xv, tv, yv = self._make_split(x_arr, t, y_arr, eval_set)
        train, val = (xt, tt, yt), (xv, tv, yv)

        rows = []
        seen: dict[str, int] = {}
        for candidate in self.candidates:
            factory, space = self._as_factory_space(candidate)
            best_params = self._tune_candidate(factory, space, train, val)
            scored = factory(**best_params)
            score = self._fit_score(scored, train, val)
            name = scored.display_name
            if name in seen:
                seen[name] += 1
                name = f'{name}_{seen[name]}'
            else:
                seen[name] = 0
            rows.append({'model': name, 'val_score': score, 'best_params': best_params, '_factory': factory})

        leaderboard = pd.DataFrame(rows).sort_values('val_score', ascending=False).reset_index(drop=True)
        self.leaderboard_ = leaderboard.drop(columns='_factory')

        k = min(self.top_k, len(leaderboard))
        self._ensemble = []
        for i in range(k):
            model = leaderboard.loc[i, '_factory'](**leaderboard.loc[i, 'best_params'])
            model.fit(x_arr, t, y_arr)
            self._ensemble.append(model)
        self.best_model_ = self._ensemble[0]
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with the best model, or the top-k average when `top_k > 1`."""
        if not self._ensemble:
            raise RuntimeError('AutoUplift has not been fitted yet. Call .fit() first.')
        if return_components:
            parts = [m.predict(X, return_components=True) for m in self._ensemble]
            uplift = np.mean([p[0] for p in parts], axis=0)
            y0 = np.mean([p[1] for p in parts], axis=0)
            y1 = np.mean([p[2] for p in parts], axis=0)
            return uplift, y0, y1
        return np.mean([np.asarray(m.predict(X)).reshape(-1) for m in self._ensemble], axis=0)
