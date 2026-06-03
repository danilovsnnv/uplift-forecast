__all__ = ['FeatureScaler']


import torch
from torch import nn


def _identity_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype), \
           torch.ones(1, x.shape[-1], device=x.device, dtype=x.dtype)


def _standard_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True, unbiased=False) + eps


def _robust_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    median = x.median(dim=0, keepdim=True).values
    mad = (x - median).abs().median(dim=0, keepdim=True).values
    return median, mad + eps


def _minmax_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    return x_min, (x_max - x_min) + eps


_STATS_FNS = {
    None: _identity_stats,
    'identity': _identity_stats,
    'standard': _standard_stats,
    'robust': _robust_stats,
    'minmax': _minmax_stats,
}


class FeatureScaler(nn.Module):
    """Per-feature affine scaler for tabular data.

    Statistics are computed once on `fit` (the training data) and cached, so
    `transform` and `inverse_transform` are deterministic and independent of the
    batch they are called on. As a fallback, the first `transform` on an unfitted
    scaler fits it lazily.

    Args:
        scaler_type (str): One of 'identity', 'standard', 'robust', 'minmax'.
        eps (float): Added to the scale denominator to guard against zero-variance columns.
    """

    def __init__(self, scaler_type: str | None = 'robust', eps: float = 1e-6):
        super().__init__()
        if scaler_type not in _STATS_FNS:
            raise ValueError(
                f"Unknown scaler_type='{scaler_type}'. "
                f"Valid options: {[k for k in _STATS_FNS if k is not None]}."
            )
        self.scaler_type = scaler_type
        self.eps = eps
        self.x_shift: torch.Tensor | None = None
        self.x_scale: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> 'FeatureScaler':
        """Compute and cache the shift/scale statistics from `x`."""
        self.x_shift, self.x_scale = _STATS_FNS[self.scaler_type](x, self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_shift is None:
            self.fit(x)
        return (x - self.x_shift) / self.x_scale

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        if self.x_shift is None:
            raise RuntimeError('Call transform() before inverse_transform().')
        return z * self.x_scale + self.x_shift
