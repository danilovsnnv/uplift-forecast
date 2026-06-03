__all__ = ['VCNet']


from functools import partial
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel, _to_float32
from ..common._modules import _FcnBlock
from ..losses import MSELoss


def _truncated_power_basis(t: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Truncated-power spline basis ``[1, t, ..., t^degree, relu(t-k)^degree, ...]``."""
    t = t.reshape(-1, 1)
    powers = [t**d for d in range(degree + 1)]
    truncated = [torch.relu(t - k) ** degree for k in knots]
    return torch.cat(powers + truncated, dim=1)


class _VaryingCoefficientHead(nn.Module):
    """Linear head whose weights vary smoothly with the dose via a spline basis.

    The effective weight matrix is ``W(t) = sum_b b_b(t) * W_b`` (and similarly for
    the bias), so the head's response is a smooth function of the dose ``t``.
    """

    def __init__(self, in_dim: int, out_dim: int, n_knots: int, degree: int):
        super().__init__()
        knots = torch.linspace(0.0, 1.0, n_knots + 2)[1:-1]
        self.register_buffer('knots', knots)
        self.degree = degree
        n_basis = (degree + 1) + n_knots
        self.weight = nn.Parameter(torch.empty(n_basis, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_basis, out_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        basis = _truncated_power_basis(t, self.knots, self.degree)
        weight_t = torch.einsum('nb,bio->nio', basis, self.weight)
        out = torch.bmm(z.unsqueeze(1), weight_t).squeeze(1)
        return out + basis @ self.bias


class VCNet(BaseNeuralUpliftModel):
    """Varying-Coefficient Network for continuous-dose response (Nie et al., ICLR 2021).

    A shared representation ``z(x)`` feeds a varying-coefficient head whose weights
    are smooth spline functions of the (continuous) dose ``t``, so the average
    dose-response function ``mu(x, t)`` stays smooth in ``t`` (arXiv:2103.07861).
    Doses are expected in ``[0, 1]`` (scale your treatment accordingly).

    ``predict`` reports the uplift of a target dose over a reference dose,
    ``mu(x, target_dose) - mu(x, reference_dose)``; ``predict_dose_response`` returns
    the full curve over a dose grid. VCNet trains on the factual MSE (point outcome).

    Args:
        input_size: Number of input features.
        hidden_size: Width of the shared representation.
        activation: Non-linearity, see ``get_activation_fn``.
        loss: Point outcome loss (sizes the head); defaults to ``MSELoss()``.
        valid_loss: Validation loss; defaults to ``loss``.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        valid_batch_size: Validation batch size (defaults to ``batch_size``).
        scaler_type: Feature/target scaler — ``identity``, ``standard``, ``robust``, ``minmax``.
        normalize_y: Whether to scale the target.
        random_seed: Seed used in ``on_fit_start``.
        alias: Display name for this instance.
        optimizer: ``partial(SomeOptimizer, ...)`` (optional).
        optimizer_kwargs: Extra kwargs for the optimizer.
        scheduler: ``partial(SomeScheduler, ...)`` (optional).
        scheduler_kwargs: Extra kwargs for the scheduler.
        dataloader_kwargs: Extra kwargs forwarded to every ``DataLoader``.
        rep_n_layers: Number of hidden layers in the shared representation.
        n_knots: Number of interior spline knots for the varying-coefficient head.
        spline_degree: Degree of the truncated-power spline basis.
        reference_dose: Baseline dose ``t0`` subtracted in ``predict``.
        target_dose: Dose whose effect ``predict`` reports.
        **trainer_kwargs: Forwarded to ``pytorch_lightning.Trainer``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 200,
        activation: str = 'ELU',
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        valid_batch_size: int | None = None,
        scaler_type: str = 'robust',
        normalize_y: bool = True,
        random_seed: int | None = 1,
        alias: str | None = None,
        optimizer: partial[Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: partial[LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        rep_n_layers: int = 3,
        n_knots: int = 2,
        spline_degree: int = 2,
        reference_dose: float = 0.0,
        target_dose: float = 1.0,
        **trainer_kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            loss=loss if loss is not None else MSELoss(),
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            scaler_type=scaler_type,
            normalize_y=normalize_y,
            random_seed=random_seed,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )
        self.rep_n_layers = rep_n_layers
        self.n_knots = n_knots
        self.spline_degree = spline_degree
        self.reference_dose = reference_dose
        self.target_dose = target_dose

        self.representation = _FcnBlock(input_size, hidden_size, rep_n_layers, activation)
        self._build_output_heads()

    def _build_output_heads(self) -> None:
        self.head = _VaryingCoefficientHead(self.hidden_size, self._outcome_size, self.n_knots, self.spline_degree)

    def forward(self, features: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        return self.head(self.representation(features), dose.reshape(-1).clamp(0.0, 1.0))

    def _step(self, batch, loss_fn: nn.Module) -> dict:
        del loss_fn
        x, dose, y_true = batch
        x, dose, y_true = self._normalization(x, dose, y_true)
        pred = self._decode_outcome(self(x, dose))
        return {'loss': torch.nn.functional.mse_loss(pred, y_true)}

    def _mu_at(self, x: torch.Tensor, dose: float) -> torch.Tensor:
        t = torch.full((x.shape[0],), float(dose), device=x.device)
        return self._inv_normalization(self._decode_outcome(self(x, t)))

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        with torch.no_grad():
            x = self._normalization(batch)
            return self._mu_at(x, self.reference_dose), self._mu_at(x, self.target_dose)

    def predict_dose_response(self, X: ArrayLike, t_grid: ArrayLike) -> np.ndarray:
        """Predicted dose-response ``mu(x, t)`` over a dose grid; shape ``[n, len(t_grid)]``."""
        self.eval()
        x = self._normalization(torch.from_numpy(_to_float32(X)))
        grid = np.asarray(t_grid, dtype=float).reshape(-1)
        with torch.no_grad():
            columns = [self._mu_at(x, float(dose)).reshape(-1).cpu().numpy() for dose in grid]
        return np.stack(columns, axis=1)
