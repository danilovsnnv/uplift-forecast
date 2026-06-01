__all__ = ['RERUM']


from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from torch import nn

from ..common._base_neural import BaseNeuralUpliftModel
from ..common._uplift_model import UpliftModel
from ._loss_builder import build_rerum_loss


class RERUM(UpliftModel):
    """Rankability-enhanced Revenue Uplift Modeling framework (He et al., KDD'24).

    RERUM is model-agnostic: it wraps any `BaseNeuralUpliftModel` and swaps in a
    `RERUMLoss` to optimise the *ranking* of predicted uplift rather than only its
    accuracy:

    1. `RERUMLoss` declares `outputsize = 3`, so `model.set_loss(...)` rebuilds the
       outcome heads to emit zero-inflated-lognormal (ZILN) parameters, which fit
       the continuous, zero-inflated, long-tailed revenue response.
    2. The objective adds within-group, cross-group, and listwise uplift-ranking
       terms to the ZILN regression (paper Eqs. 10-21), plus any model-specific
       structural term (DragonNet's treatment classifier, CFRNet's IPM).

    The model needs no RERUM-specific code: the per-model loss configuration lives
    in `build_rerum_loss`. Target normalisation is disabled because ZILN models the
    response scale itself; the paper's `λ‖θ‖²` term is applied via the optimizer's
    `weight_decay`.

    `RERUM` implements the `UpliftModel` contract, so it can be used directly or
    inside `UpliftForecast(models=[RERUM(...)])`.

    Args:
        model: A `BaseNeuralUpliftModel` instance (e.g. `DragonNet`, `CFRNet`, `TARNet`).
        within_ranking_weight: Weight on the within-group response ranking (Eq. 10-11).
        cross_ranking_weight: Weight on the cross-group response ranking (Eq. 12-13).
            Defaults to 0 (the term the authors' reference code omits) but is available.
        listwise_ranking_weight: Weight on the listwise uplift ranking (Eq. 20).
        ranking_sample_size: Individuals sampled per group for the pairwise terms
            (Algorithm 1); `None` uses every pair in the batch.
        l2_lambda: L2 weight-decay coefficient (paper's `λ`); applied via the optimizer.
        valid_loss: Validation loss; defaults to the training `RERUMLoss`.
        alias: Display name; defaults to `f'RERUM_{model.display_name}'`.

    Example:
        >>> from uplift_forecast.frameworks import RERUM
        >>> from uplift_forecast.models import DragonNet
        >>> framework = RERUM(model=DragonNet(input_size=10, max_epochs=5))
        >>> framework.fit(X_train, t_train, y_train)
        >>> uplift = framework.predict(X_test)
    """

    def __init__(
        self,
        model: BaseNeuralUpliftModel,
        *,
        within_ranking_weight: float = 1e-4,
        cross_ranking_weight: float = 0.0,
        listwise_ranking_weight: float = 10.0,
        ranking_sample_size: int | None = None,
        l2_lambda: float = 0.0,
        valid_loss: nn.Module | None = None,
        alias: str | None = None,
    ):
        if not isinstance(model, BaseNeuralUpliftModel):
            raise TypeError(
                f'RERUM wraps a neural uplift model (BaseNeuralUpliftModel), '
                f'got {type(model).__name__}.'
            )

        self.model = model
        self.alias = alias

        loss = build_rerum_loss(
            model,
            within_ranking_weight=within_ranking_weight,
            cross_ranking_weight=cross_ranking_weight,
            listwise_ranking_weight=listwise_ranking_weight,
            ranking_sample_size=ranking_sample_size,
        )
        model.set_loss(loss, valid_loss=valid_loss)
        model.normalize_y = False

        if l2_lambda > 0:
            model.optimizer_kwargs = {**model.optimizer_kwargs, 'weight_decay': l2_lambda}

    @property
    def display_name(self) -> str:
        return self.alias or f'RERUM_{self.model.display_name}'

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **fit_params: Any,
    ) -> 'RERUM':
        """Train the wrapped model with the RERUM objective."""
        self.model.fit(X, treatment, y, eval_set=eval_set, **fit_params)
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict uplift. If `return_components=True`, also return `(uplift, y0, y1)`."""
        return self.model.predict(X, return_components=return_components)
