__all__ = ['PropensityScoreMatcher']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ._base_matcher import BaseMatcher


class PropensityScoreMatcher(BaseMatcher):
    """Propensity score matching (PSM).

    Fits a caller-supplied probabilistic classifier to predict treatment from the
    covariates, then matches treated and control units on the estimated propensity
    score (the predicted probability of treatment).

    Args:
        model: Classifier with sklearn-style ``fit`` / ``predict_proba`` (CatBoost,
            lgbm, sklearn). Trained to predict treatment.
        n_neighbors (int): Number of control units matched to each treated unit.
        caliper (float): Maximum propensity-score difference for a valid match;
            treated units with no control within it are dropped. None disables it.
        replace (bool): Whether a control may match several treated units.
        alias (str): Optional display name.
    """

    def __init__(
        self,
        model: Any,
        n_neighbors: int = 1,
        caliper: float | None = None,
        replace: bool = True,
        alias: str | None = None,
    ):
        super(PropensityScoreMatcher, self).__init__(
            n_neighbors=n_neighbors, caliper=caliper, replace=replace, alias=alias,
        )
        if not hasattr(model, 'predict_proba'):
            raise TypeError('model must expose predict_proba for propensity score matching.')
        self.model = model
        self._fitted_model = None

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        self._fitted_model = deepcopy(self.model)
        self._fitted_model.fit(X, treatment)

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        propensity = np.asarray(self._fitted_model.predict_proba(X))[:, 1]
        return propensity.reshape(-1, 1)
