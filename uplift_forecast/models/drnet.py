__all__ = ['DRNet']


from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel, _to_float32
from ..common._modules import _FcnBlock, _FcnHead
from ..losses import MSELoss


class DRNet(BaseNeuralUpliftModel):
    """Dose-Response Network (Schwab et al., AAAI 2020, arXiv:1902.00981).

    A shared representation feeds one outcome head per dosage stratum: the dose
    range ``[0, 1]`` is split into ``n_strata`` equal bins and the head for the bin
    containing ``t`` predicts ``mu(x, t)`` from ``[representation, t]``. Hierarchical
    per-stratum heads let the response bend differently across the dose range.
    Doses are expected in ``[0, 1]`` (scale your treatment accordingly).

    ``predict`` reports ``mu(x, target_dose) - mu(x, reference_dose)`` and
    ``predict_dose_response`` returns the full curve over a dose grid. DRNet trains
    on the factual MSE (point outcome).

    Args:
        input_size: Number of input features.
        hidden_size: Width of the representation and head layers.
        activation: Non-linearity, see ``get_activation_fn``.
        loss: Point outcome loss (sizes the heads); defaults to ``MSELoss()``.
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
        head_n_layers: Number of hidden layers in each per-stratum head.
        head_hidden_size: Hidden width for heads (defaults to ``hidden_size``).
        n_strata: Number of dosage strata (per-stratum heads).
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
        head_n_layers: int = 2,
        head_hidden_size: int | None = None,
        n_strata: int = 5,
        reference_dose: float = 0.0,
        target_dose: float = 1.0,
        **trainer_kwargs,
    ):
        if n_strata < 1:
            raise ValueError(f'n_strata must be >= 1; got {n_strata}.')
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
        self.head_n_layers = head_n_layers
        self.head_hidden_size = head_hidden_size or hidden_size
        self.n_strata = n_strata
        self.reference_dose = reference_dose
        self.target_dose = target_dose
        self._activation = activation

        self.representation = _FcnBlock(input_size, hidden_size, rep_n_layers, activation)
        self._build_output_heads()

    def _build_output_heads(self) -> None:
        self.heads = nn.ModuleList(
            _FcnHead(self.hidden_size + 1, self.head_hidden_size, self._outcome_size, self.head_n_layers,
                     self._activation)
            for _ in range(self.n_strata)
        )

    def _stratum(self, dose: torch.Tensor) -> torch.Tensor:
        return (dose.reshape(-1).clamp(0.0, 1.0 - 1e-6) * self.n_strata).long()

    def forward(self, features: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        rep = self.representation(features)
        dose = dose.reshape(-1, 1).clamp(0.0, 1.0)
        head_input = torch.cat([rep, dose], dim=1)
        stratum = self._stratum(dose)
        pred = rep.new_empty(rep.shape[0], self._outcome_size)
        for s, head in enumerate(self.heads):
            mask = stratum == s
            if mask.any():
                pred[mask] = head(head_input[mask])
        return pred

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        del loss_fn
        x, dose, y_true = batch
        x, dose, y_true = self._normalization(x, dose, y_true)
        pred = self._decode_outcome(self(x, dose))
        return {'loss': F.mse_loss(pred, y_true)}

    def _mu_at(self, x: torch.Tensor, dose: float) -> torch.Tensor:
        t = torch.full((x.shape[0],), float(dose), device=x.device)
        return self._inv_normalization(self._decode_outcome(self(x, t)))

    def predict_step(self, batch: Any, batch_idx: int) -> tuple:
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
