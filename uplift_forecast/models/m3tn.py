__all__ = ['M3TN']


from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from pytorch_lightning import Trainer
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel, _PredictDataset, _to_float32
from ..common._data_module import UpliftDataModule
from ..common._modules import _FcnBlock, _FcnHead
from ..losses import MSELoss


class M3TN(BaseNeuralUpliftModel):
    """Multiple-treatment Meta-learning network (M3TN, ICASSP 2024, arXiv:2401.14426).

    A shared Multi-gate Mixture-of-Experts (MMoE) representation feeds an additive
    uplift reparameterization: a control head predicts ``mu_0(x)`` and one uplift
    head per treated arm predicts ``tau_k(x)``, so ``mu_k(x) = mu_0(x) + tau_k(x)``.
    This shares the (large) outcome level across arms and parameterises only the
    (small) per-arm effect, which scales better than ``K`` independent outcome heads.

    Treatment is an integer arm in ``{0..n_treatments-1}`` (0 = control). ``predict``
    returns one uplift column per treated arm as a ``[n, n_treatments-1]`` array,
    collapsing to a flat ``[n]`` array in the binary case. M3TN trains and validates
    on the factual outcome MSE (point outcomes); the ``loss`` argument sizes the
    point head but the objective is computed internally.

    Args:
        input_size: Number of input features.
        n_treatments: Total number of arms including control (``K``, so ``K >= 2``).
        hidden_size: Expert / representation width. Heads use ``hidden_size // 2``.
        n_experts: Number of shared MMoE experts.
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
        rep_n_layers: Number of hidden layers in each expert.
        head_n_layers: Number of hidden layers in each head.
        head_hidden_size: Hidden width for heads (defaults to ``hidden_size // 2``).
        **trainer_kwargs: Forwarded to ``pytorch_lightning.Trainer``.
    """

    def __init__(
        self,
        input_size: int,
        n_treatments: int = 2,
        hidden_size: int = 200,
        n_experts: int = 4,
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
        **trainer_kwargs,
    ):
        if n_treatments < 2:
            raise ValueError(f'n_treatments must be >= 2 (control + >=1 treated); got {n_treatments}.')
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
        self.n_treatments = n_treatments
        self.n_experts = n_experts
        self.rep_n_layers = rep_n_layers
        self.head_n_layers = head_n_layers
        self.head_hidden_size = head_hidden_size or (hidden_size // 2)
        self._activation = activation

        self.experts = nn.ModuleList(
            _FcnBlock(input_size, hidden_size, rep_n_layers, activation) for _ in range(n_experts)
        )
        self.gates = nn.ModuleList(nn.Linear(input_size, n_experts) for _ in range(n_treatments))

        self._build_output_heads()

    def _build_output_heads(self) -> None:
        self.control_head = _FcnHead(self.hidden_size, self.head_hidden_size, 1, self.head_n_layers, self._activation)
        self.uplift_heads = nn.ModuleList(
            _FcnHead(self.hidden_size, self.head_hidden_size, 1, self.head_n_layers, self._activation)
            for _ in range(self.n_treatments - 1)
        )

    def _arm_representation(self, expert_out: torch.Tensor, arm: int, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gates[arm](x), dim=1).unsqueeze(-1)
        return (weights * expert_out).sum(dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expert_out = torch.stack([expert(features) for expert in self.experts], dim=1)
        mu0 = self.control_head(self._arm_representation(expert_out, 0, features))
        columns = [mu0]
        for arm, head in enumerate(self.uplift_heads, start=1):
            tau = head(self._arm_representation(expert_out, arm, features))
            columns.append(mu0 + tau)
        return torch.cat(columns, dim=1)

    def _step(self, batch, loss_fn: nn.Module) -> dict:
        del loss_fn
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        mu = self(x)
        idx = treatment_true.long().clamp(0, self.n_treatments - 1)
        factual = mu.gather(1, idx)
        return {'loss': F.mse_loss(factual, y_true)}

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        with torch.no_grad():
            x = self._normalization(batch)
            mu = self._inv_normalization(self(x))
        return (mu,)

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict per-arm uplift vs control as ``[n, K-1]`` (flat ``[n]`` when K=2)."""
        dataset = _PredictDataset(torch.from_numpy(_to_float32(X)))
        datamodule = UpliftDataModule(
            predict_dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **self.dataloader_kwargs,
        )
        trainer = Trainer(**self.trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        mu = torch.vstack([part[0] for part in fcsts]).cpu().numpy()
        y0 = mu[:, 0]
        y1 = mu[:, 1:]
        uplift = y1 - y0[:, None]
        if uplift.shape[1] == 1:
            uplift, y1 = uplift[:, 0], y1[:, 0]
        if return_components:
            return uplift, y0, y1
        return uplift
