__all__ = ['FlexTENet']


from functools import partial
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel
from ..common._modules import _FcnBlock, _FcnHead
from ..losses import MSELoss


class FlexTENet(BaseNeuralUpliftModel):
    """FlexTENet (Curth & van der Schaar, NeurIPS 2021, arXiv:2106.03765).

    Each outcome head reads a *shared* subspace (common to both arms) and an
    arm-*private* subspace, so the two potential-outcome functions can share
    structure where it helps and diverge where it doesn't. An orthogonality
    penalty between the shared and private first-layer weights regularises how
    much information the subspaces are allowed to overlap.

    The outcome-head width follows ``self.loss.outputsize`` (1 point / 3 ZILN).

    Args:
        input_size: Number of input features.
        hidden_size: Total representation width, split into shared/private by ``shared_ratio``.
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
        rep_n_layers: Number of hidden layers in the shared and private blocks.
        head_n_layers: Number of hidden layers in each outcome head.
        head_hidden_size: Hidden width for outcome heads (defaults to ``hidden_size``).
        shared_ratio: Fraction of ``hidden_size`` allocated to the shared subspace.
        ortho_lambda: Weight on the shared/private orthogonality penalty.
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
        rep_n_layers: int = 2,
        head_n_layers: int = 2,
        head_hidden_size: int | None = None,
        shared_ratio: float = 0.5,
        ortho_lambda: float = 1e-3,
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
        if not 0.0 < shared_ratio < 1.0:
            raise ValueError(f'shared_ratio must be in (0, 1); got {shared_ratio}.')
        self.rep_n_layers = rep_n_layers
        self.head_n_layers = head_n_layers
        self.head_hidden_size = head_hidden_size or hidden_size
        self.shared_ratio = shared_ratio
        self.ortho_lambda = ortho_lambda
        self._activation = activation

        self.shared_dim = max(1, int(hidden_size * shared_ratio))
        self.private_dim = max(1, hidden_size - self.shared_dim)

        self.shared_block = _FcnBlock(input_size, self.shared_dim, rep_n_layers, activation)
        self.private0_block = _FcnBlock(input_size, self.private_dim, rep_n_layers, activation)
        self.private1_block = _FcnBlock(input_size, self.private_dim, rep_n_layers, activation)

        self._build_output_heads()

    def _build_output_heads(self) -> None:
        in_dim = self.shared_dim + self.private_dim
        self.head0 = _FcnHead(in_dim, self.head_hidden_size, self._outcome_size, self.head_n_layers, self._activation)
        self.head1 = _FcnHead(in_dim, self.head_hidden_size, self._outcome_size, self.head_n_layers, self._activation)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_block(features)
        y0 = self.head0(torch.cat([shared, self.private0_block(features)], dim=1))
        y1 = self.head1(torch.cat([shared, self.private1_block(features)], dim=1))
        return y0, y1

    def _orthogonal_penalty(self) -> torch.Tensor:
        # Penalise overlap between the shared and private input subspaces via the
        # first-layer weight cross-products (Frobenius norm of W_shared @ W_private^T).
        w_shared = self.shared_block.layers[0].weight
        penalty = w_shared.new_zeros(())
        for private in (self.private0_block, self.private1_block):
            cross = w_shared @ private.layers[0].weight.T
            penalty = penalty + cross.pow(2).sum()
        return penalty

    def _step(self, batch, loss_fn: nn.Module) -> dict:
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        y0_pred, y1_pred = self(x)
        out = loss_fn(y_true=y_true, t_true=treatment_true, y0_pred=y0_pred, y1_pred=y1_pred)
        ortho = self.ortho_lambda * self._orthogonal_penalty()
        out['loss'] = out['loss'] + ortho
        out['loss_ortho'] = ortho
        return out

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        with torch.no_grad():
            x = self._normalization(batch)
            y0_pred, y1_pred = self(x)
            y0_pred = self._inv_normalization(self._decode_outcome(y0_pred))
            y1_pred = self._inv_normalization(self._decode_outcome(y1_pred))
        return y0_pred, y1_pred
