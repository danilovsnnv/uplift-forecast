__all__ = ['TARNet']


from functools import partial
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel
from ..common._modules import _FcnBlock, _FcnHead
from ..common._multiarm_neural import _MultiArmNeuralMixin
from ..losses import MSELoss


class TARNet(_MultiArmNeuralMixin, BaseNeuralUpliftModel):
    """Treatment-Agnostic Representation Network (Shalit et al., 2017), binary or multi-arm.

    A shared representation Φ feeds two separate outcome heads (control /
    treatment). The model is trained on the factual outcome loss only — it is the
    `α = 0` (no IPM penalty) case of `CFRNet`. Use `CFRNet` when you want the
    counterfactual-regression imbalance penalty.

    The outcome-head width follows `self.loss.outputsize`, so the model works with
    any compatible loss (e.g. `uplift_forecast.frameworks.RERUM` swaps in a ZILN
    ranking objective without TARNet needing to know about it).

    With `n_treatments > 2` the shared representation feeds M3TN-style additive heads
    (a control head plus one direct-uplift head per treated arm, `mu_k = mu_0 + tau_k`)
    trained on the factual point MSE, and `predict` returns `[n, n_treatments-1]`
    (one uplift column per treated arm vs control).

    Args:
        input_size: Number of input features.
        n_treatments: Total number of arms including control (`K`, so `K >= 2`). With
            `K = 2` it is the standard binary TARNet (two heads, any compatible loss);
            with `K > 2` it switches to the multi-arm additive-head point-MSE path.
        hidden_size: Width of the representation layers.
        activation: Non-linearity for representation and head layers (paper uses ELU).
        loss: Training loss; defaults to `MSELoss()` (factual regression).
        valid_loss: Validation loss; defaults to `loss`.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        valid_batch_size: Validation batch size (defaults to `batch_size`).
        scaler_type: Feature/target scaler — `identity`, `standard`, `robust`, `minmax`.
        normalize_y: Whether to scale the target.
        random_seed: Seed used in `on_fit_start`.
        alias: Display name for this instance.
        optimizer: `partial(SomeOptimizer, ...)` (optional).
        optimizer_kwargs: Extra kwargs for the optimizer.
        scheduler: `partial(SomeScheduler, ...)` (optional).
        scheduler_kwargs: Extra kwargs for the scheduler.
        dataloader_kwargs: Extra kwargs forwarded to every `DataLoader`.
        rep_n_layers: Number of hidden layers in the representation network.
        head_n_layers: Number of hidden layers in each outcome head.
        head_hidden_size: Hidden width for outcome heads (defaults to `hidden_size`).
        **trainer_kwargs: Forwarded to `pytorch_lightning.Trainer`.
    """

    def __init__(
        self,
        input_size: int,
        n_treatments: int = 2,
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
        head_n_layers: int = 3,
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
        self.rep_n_layers = rep_n_layers
        self.head_n_layers = head_n_layers
        self.head_hidden_size = head_hidden_size or hidden_size
        self._activation = activation

        self.representation = _FcnBlock(input_size, hidden_size, rep_n_layers, activation)
        self._rep_dim = self.representation.output_dim

        self._build_output_heads()

    def _build_output_heads(self) -> None:
        if self._is_multi_arm:
            self._build_additive_heads(self._make_arm_head)
        else:
            self.head0 = self._make_head()
            self.head1 = self._make_head()

    def _make_head(self) -> _FcnHead:
        return _FcnHead(
            self._rep_dim, self.head_hidden_size, self._outcome_size, self.head_n_layers, self._activation,
        )

    def _make_arm_head(self) -> _FcnHead:
        return _FcnHead(self._rep_dim, self.head_hidden_size, 1, self.head_n_layers, self._activation)

    def _multiarm_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.representation(x)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rep = self.representation(features)
        return self.head0(rep), self.head1(rep)

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        if self._is_multi_arm:
            return self._multiarm_factual_step(batch)
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        y0_pred, y1_pred = self(x)
        return loss_fn(y_true=y_true, t_true=treatment_true, y0_pred=y0_pred, y1_pred=y1_pred)

    def predict_step(self, batch: Any, batch_idx: int) -> tuple:
        del batch_idx
        if self._is_multi_arm:
            return self._multiarm_predict_step(batch)
        with torch.no_grad():
            x = self._normalization(batch)
            y0_pred, y1_pred = self(x)
            y0_pred = self._inv_normalization(self._decode_outcome(y0_pred))
            y1_pred = self._inv_normalization(self._decode_outcome(y1_pred))
        return y0_pred, y1_pred

    def predict(self, X: Any, *, return_components: bool = False) -> Any:
        if self._is_multi_arm:
            return self._multiarm_predict(X, return_components=return_components)
        return super().predict(X, return_components=return_components)
