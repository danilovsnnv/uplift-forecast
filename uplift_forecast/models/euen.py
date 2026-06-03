__all__ = ['EUEN']


from functools import partial
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel
from ..common._modules import _FcnBlock, _FcnHead
from ..losses import MSELoss


class EUEN(BaseNeuralUpliftModel):
    """Explicit Uplift Effect Network (branch-switch model, KDD-2024 benchmark taxonomy).

    Two towers: a *control* tower estimates the control outcome ``mu0(x)`` and an
    *uplift* tower estimates the treatment effect ``tau(x)`` directly, so the
    treated outcome is ``y1 = mu0 + tau`` rather than a difference of two
    independently-fit outcome heads. Parameterising the uplift term explicitly
    keeps it from being swamped by the (usually larger) outcome level.

    The outcome-head width follows ``self.loss.outputsize``; with a ZILN loss the
    additive form acts on the ZILN parameters, so prefer a point loss (the default
    ``MSELoss``) for the textbook EUEN.

    Args:
        input_size: Number of input features.
        hidden_size: Width of each tower's representation. Heads use ``hidden_size // 2``.
        activation: Non-linearity, see ``get_activation_fn``.
        loss: Training loss; defaults to ``MSELoss()``.
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
        rep_n_layers: Number of hidden layers in each tower's representation.
        head_n_layers: Number of hidden layers in each outcome/uplift head.
        head_hidden_size: Hidden width for the heads (defaults to ``hidden_size // 2``).
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
        self.head_n_layers = head_n_layers
        self.head_hidden_size = head_hidden_size or (hidden_size // 2)
        self._activation = activation

        self.control_rep = _FcnBlock(input_size, hidden_size, rep_n_layers, activation)
        self.uplift_rep = _FcnBlock(input_size, hidden_size, rep_n_layers, activation)

        self._build_output_heads()

    def _build_output_heads(self) -> None:
        self.control_head = self._make_head()
        self.uplift_head = self._make_head()

    def _make_head(self) -> _FcnHead:
        return _FcnHead(
            self.hidden_size, self.head_hidden_size, self._outcome_size, self.head_n_layers, self._activation,
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu0 = self.control_head(self.control_rep(features))
        tau = self.uplift_head(self.uplift_rep(features))
        return mu0, mu0 + tau

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        y0_pred, y1_pred = self(x)
        return loss_fn(y_true=y_true, t_true=treatment_true, y0_pred=y0_pred, y1_pred=y1_pred)

    def predict_step(self, batch: Any, batch_idx: int) -> tuple:
        del batch_idx
        with torch.no_grad():
            x = self._normalization(batch)
            y0_pred, y1_pred = self(x)
            y0_pred = self._inv_normalization(self._decode_outcome(y0_pred))
            y1_pred = self._inv_normalization(self._decode_outcome(y1_pred))
        return y0_pred, y1_pred
