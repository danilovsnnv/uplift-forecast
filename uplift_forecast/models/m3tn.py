__all__ = ['M3TN']


from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel
from ..common._modules import _FcnBlock, _FcnHead
from ..common._multiarm_neural import _MultiArmNeuralMixin
from ..losses import MSELoss


class M3TN(_MultiArmNeuralMixin, BaseNeuralUpliftModel):
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
        self._build_additive_heads(self._make_arm_head)

    def _make_arm_head(self) -> _FcnHead:
        return _FcnHead(self.hidden_size, self.head_hidden_size, 1, self.head_n_layers, self._activation)

    def _arm_representation(self, expert_out: torch.Tensor, arm: int, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gates[arm](x), dim=1).unsqueeze(-1)
        return (weights * expert_out).sum(dim=1)

    def _multiarm_representation(self, x: torch.Tensor) -> list[torch.Tensor]:
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        return [self._arm_representation(expert_out, arm, x) for arm in range(self.n_treatments)]

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        del loss_fn  # M3TN always trains on the factual point MSE
        return self._multiarm_factual_step(batch)

    def predict_step(self, batch: Any, batch_idx: int) -> tuple:
        del batch_idx
        return self._multiarm_predict_step(batch)

    def predict(self, X: Any, *, return_components: bool = False) -> Any:
        """Predict per-arm uplift vs control as ``[n, K-1]`` (flat ``[n]`` when K=2)."""
        return self._multiarm_predict(X, return_components=return_components)
