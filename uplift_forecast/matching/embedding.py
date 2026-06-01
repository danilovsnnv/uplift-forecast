__all__ = ['EmbeddingMatcher']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch

from ._base_matcher import BaseMatcher


class EmbeddingMatcher(BaseMatcher):
    """Matching in a learned or provided representation space.

    Transforms covariates ``X`` into embeddings ``Z`` with an arbitrary encoder,
    then runs nearest-neighbor matching in ``Z``. Treated and control units are
    embedded by the same encoder, so they live in one comparable space. Useful
    for high-dimensional data (text/user/item embeddings, deep uplift models).

    For radius or kernel matching in the embedding space, precompute ``Z`` and pass
    it to ``NearestNeighborMatcher`` / ``KernelMatcher`` (they accept any numeric
    input).

    Args:
        encoder: How to embed ``X``. One of: an object with ``.transform`` (e.g.
            sklearn PCA), a ``torch.nn.Module``, a plain callable ``f(X) -> Z``, or
            ``None`` to match on ``X`` as-is (precomputed embeddings / identity).
        n_neighbors (int): Number of control units matched to each treated unit.
        caliper (float): Maximum distance (in embedding space) for a valid match.
        replace (bool): Whether a control may match several treated units.
        alias (str): Optional display name.
    """

    def __init__(
        self,
        encoder: Any = None,
        n_neighbors: int = 1,
        caliper: float | None = None,
        replace: bool = True,
        alias: str | None = None,
    ):
        super(EmbeddingMatcher, self).__init__(
            n_neighbors=n_neighbors, caliper=caliper, replace=replace, alias=alias,
        )
        self.encoder = encoder
        self._encoder: Any = None

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        encoder = self.encoder
        # Fit unsupervised encoders (e.g. PCA) on a copy so self.encoder stays unfitted.
        # torch modules are assumed pretrained and are not copied.
        if encoder is not None and not isinstance(encoder, torch.nn.Module) and hasattr(encoder, 'fit'):
            encoder = deepcopy(encoder)
            encoder.fit(X)
        self._encoder = encoder

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        encoder = self._encoder
        if encoder is None:
            z = np.asarray(X, dtype=float)
        elif isinstance(encoder, torch.nn.Module):
            encoder.eval()
            with torch.no_grad():
                z = encoder(torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float32)).cpu().numpy()
        elif hasattr(encoder, 'transform'):
            z = encoder.transform(X)
        elif callable(encoder):
            z = encoder(X)
        else:
            raise TypeError(
                'encoder must be None, a torch.nn.Module, an object with .transform, or a callable; '
                f'got {type(encoder).__name__}.'
            )
        z = np.asarray(z, dtype=float)
        return z.reshape(-1, 1) if z.ndim == 1 else z
