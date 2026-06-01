__all__ = ['KERNELS', 'epanechnikov', 'gaussian', 'triangular', 'uniform']


import numpy as np


def gaussian(u: np.ndarray) -> np.ndarray:
    """Gaussian kernel of the normalised distance ``u = d / bandwidth``."""
    return np.exp(-0.5 * np.square(u))


def epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel; zero outside ``u <= 1``."""
    return np.where(u <= 1.0, 0.75 * (1.0 - np.square(u)), 0.0)


def triangular(u: np.ndarray) -> np.ndarray:
    """Triangular kernel; zero outside ``u <= 1``."""
    return np.where(u <= 1.0, 1.0 - u, 0.0)


def uniform(u: np.ndarray) -> np.ndarray:
    """Uniform (boxcar) kernel; 1 inside ``u <= 1`` else 0."""
    return np.where(u <= 1.0, 1.0, 0.0)


KERNELS = {
    'gaussian': gaussian,
    'epanechnikov': epanechnikov,
    'triangular': triangular,
    'uniform': uniform,
}
