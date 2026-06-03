__all__ = ['DESCN']


from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel
from ..common._modules import _FcnBlock, _FcnHead
from ..losses import MSELoss


class DESCN(BaseNeuralUpliftModel):
    """Deep Entire Space Cross Network (Zhong et al., KDD 2022, arXiv:2207.09920).

    A shared representation feeds a *pseudo-treatment* (propensity) head and two
    response heads. The propensity signal is crossed into the response heads (their
    input is the representation concatenated with the predicted propensity), letting
    the model account for treatment bias and group-size imbalance jointly with
    response estimation — the "entire space" multi-task idea.

    The treatment-classification term is weighted inside ``_step`` (it is the
    model's own auxiliary objective, not the outcome loss), so any compatible
    outcome ``loss`` can be used; the head width follows ``self.loss.outputsize``.

    Args:
        input_size: Number of input features.
        hidden_size: Width of the shared representation. Heads use ``hidden_size // 2``.
        activation: Non-linearity, see ``get_activation_fn``.
        loss: Outcome training loss; defaults to ``MSELoss()``.
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
        head_n_layers: Number of hidden layers in each head.
        head_hidden_size: Hidden width for heads (defaults to ``hidden_size // 2``).
        propensity_weight: Weight on the pseudo-treatment classification BCE term.
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
        propensity_weight: float = 1.0,
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
        self.propensity_weight = propensity_weight
        self._activation = activation

        self.shared = _FcnBlock(input_size, hidden_size, rep_n_layers, activation)
        self.propensity_head = nn.Linear(hidden_size, 1)

        self._build_output_heads()

    def _build_output_heads(self) -> None:
        # Cross network: response heads read the representation plus the propensity.
        in_dim = self.hidden_size + 1
        self.head0 = _FcnHead(in_dim, self.head_hidden_size, self._outcome_size, self.head_n_layers, self._activation)
        self.head1 = _FcnHead(in_dim, self.head_hidden_size, self._outcome_size, self.head_n_layers, self._activation)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        t_pred = torch.sigmoid(self.propensity_head(shared))
        crossed = torch.cat([shared, t_pred], dim=1)
        return self.head0(crossed), self.head1(crossed), t_pred

    def _step(self, batch, loss_fn: nn.Module) -> dict:
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        y0_pred, y1_pred, t_pred = self(x)
        out = loss_fn(y_true=y_true, t_true=treatment_true, y0_pred=y0_pred, y1_pred=y1_pred, t_pred=t_pred)
        loss_t = self.propensity_weight * F.binary_cross_entropy(t_pred, treatment_true)
        out['loss'] = out['loss'] + loss_t
        out['loss_propensity'] = loss_t
        return out

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        with torch.no_grad():
            x = self._normalization(batch)
            y0_pred, y1_pred, t_pred = self(x)
            y0_pred = self._inv_normalization(self._decode_outcome(y0_pred))
            y1_pred = self._inv_normalization(self._decode_outcome(y1_pred))
        return y0_pred, y1_pred, t_pred
