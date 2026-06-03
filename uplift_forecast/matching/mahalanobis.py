__all__ = ['MahalanobisMatcher']


import numpy as np
import pandas as pd

from ._base_matcher import BaseMatcher


def _fit_whitener(X: np.ndarray | pd.DataFrame, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Learn the mean and whitening matrix for Mahalanobis distance.

    Whitening with the returned matrix turns the pooled covariance into the
    identity, so ordinary Euclidean distance in the whitened space equals the
    Mahalanobis distance.

    Args:
        X: Feature matrix.
        reg: Ridge added to the covariance diagonal before factorisation.

    Returns:
        ``(mean, whitener)`` where ``z = (x - mean) @ whitener``.
    """
    x = np.asarray(X, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1])
    cov += reg * np.eye(x.shape[1])
    whitener = np.linalg.inv(np.linalg.cholesky(cov)).T
    return mean, whitener


def _whiten(X: np.ndarray | pd.DataFrame, mean: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    """Map covariates into the whitened space defined by ``_fit_whitener``."""
    return (np.asarray(X, dtype=float) - mean) @ whitener


class MahalanobisMatcher(BaseMatcher):
    """Mahalanobis-distance matching.

    Whitens the covariates with the (pooled) inverse covariance learned on `fit`,
    so that ordinary Euclidean distance in the transformed space equals the
    Mahalanobis distance. Matching then runs in that whitened space.

    Args:
        n_neighbors (int): Number of control units matched to each treated unit.
        caliper (float): Maximum Mahalanobis distance for a valid match; treated
            units with no control within it are dropped. None disables it.
        replace (bool): Whether a control may match several treated units.
        reg (float): Ridge added to the covariance diagonal before factorisation,
            guarding against singular / ill-conditioned covariance matrices.
        alias (str): Optional display name.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: float | None = None,
        replace: bool = True,
        reg: float = 1e-6,
        alias: str | None = None,
    ):
        super().__init__(
            n_neighbors=n_neighbors, caliper=caliper, replace=replace, alias=alias,
        )
        self.reg = reg
        self._mean: np.ndarray | None = None
        self._whitener: np.ndarray | None = None

    def _fit_embedding(self, X: np.ndarray | pd.DataFrame, treatment: np.ndarray) -> None:
        self._mean, self._whitener = _fit_whitener(X, self.reg)

    def _embed(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return _whiten(X, self._mean, self._whitener)
