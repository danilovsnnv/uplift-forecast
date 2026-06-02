# Contributing

Thank you for your interest in contributing to uplift-forecast.

## Getting started

```bash
git clone https://github.com/danilovsnnv/uplift-forecast.git
cd uplift-forecast
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests -q
```

All tests must pass before submitting a pull request.

## Code style

- Python 3.10+. Use modern type hints (`X | None`, `list[X]`) without `from __future__ import annotations`.
- Single quotes by default; triple double quotes for docstrings.
- Line length 120. Ruff is the linter and formatter.

```bash
ruff check uplift_forecast/
```

Fix only the violations you introduce — do not reformat unrelated files.

## Adding a model

1. Create one file under `uplift_forecast/models/` named after the class (lowercase, underscored).
2. Subclass `BaseNeuralUpliftModel` (neural) or `BaseMetaUpliftModel` (sklearn-style).
3. Register the class in `uplift_forecast/models/__init__.py`.
4. Add tests under `tests/test_models/`.

See existing models for the expected structure and constructor forwarding pattern.

## Adding a matcher

1. Create one file under `uplift_forecast/matching/`.
2. Subclass `BaseMatcher` and implement `_fit_embedding` and `_embed`.
3. Register the class in `uplift_forecast/matching/__init__.py`.
4. Add tests under `tests/test_matching/`.

## Pull requests

- Keep changes focused. One concern per PR.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Ensure `pytest tests -q` and `ruff check uplift_forecast/` both pass.
