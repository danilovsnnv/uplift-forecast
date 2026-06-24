__all__ = ['DragonNet']


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
from ..losses import DragonNetLoss, MSELoss


class DragonNet(_MultiArmNeuralMixin, BaseNeuralUpliftModel):
    """DragonNet uplift model (Shi et al., 2019), binary or multi-arm.

    A shared representation feeds two outcome heads (control / treatment) plus
    a treatment-classifier head. The outcome-head width follows
    `self.loss.outputsize` — the default `DragonNetLoss` uses zero-inflated-
    lognormal outputs `(logit, loc, scale)`.

    With `n_treatments > 2` the shared representation feeds M3TN-style additive heads
    (`mu_k = mu_0 + tau_k`) trained on the factual point MSE plus a `K`-way
    treatment-classification (propensity) cross-entropy from the same representation;
    the targeted-regularization `epsilon` term is binary-only. `predict` returns
    `[n, n_treatments-1]` (one uplift column per treated arm vs control), and the
    default loss switches to `MSELoss()`.

    Args:
        input_size: Number of input features.
        n_treatments: Total number of arms including control (`K`, so `K >= 2`). With
            `K = 2` it is the standard binary DragonNet; with `K > 2` it switches to
            the multi-arm additive-head point-MSE path with a K-way propensity head.
        hidden_size: Width of the shared representation. Outcome heads use
            `hidden_size // 2`.
        activation: Non-linearity, see `get_activation_fn`.
        loss: Training loss; defaults to `DragonNetLoss()`.
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
        **trainer_kwargs: Forwarded to `pytorch_lightning.Trainer`.
    """

    def __init__(
        self,
        input_size: int,
        n_treatments: int = 2,
        hidden_size: int = 200,
        activation: str = 'ReLU',
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
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
        **trainer_kwargs,
    ):
        if n_treatments < 2:
            raise ValueError(f'n_treatments must be >= 2 (control + >=1 treated); got {n_treatments}.')
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            loss=loss if loss is not None else (MSELoss() if n_treatments > 2 else DragonNetLoss()),
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
        # 3-layer shared representation (paper architecture)
        self.fcnn = _FcnBlock(input_size, hidden_size, n_layers=3, activation=activation)
        self.lin_treatment = nn.Linear(hidden_size, n_treatments if self._is_multi_arm else 1)

        self._build_output_heads()

        if not self._is_multi_arm:
            self.epsilon = nn.Linear(1, 1)
            torch.nn.init.xavier_normal_(self.epsilon.weight)

    def _build_output_heads(self) -> None:
        if self._is_multi_arm:
            self._build_additive_heads(self._make_arm_head)
            return
        y_hidden_size = self.hidden_size // 2
        self.fcnn_ct = self._make_head(y_hidden_size)
        self.fcnn_tr = self._make_head(y_hidden_size)

    def _make_head(self, hidden_size: int) -> _FcnHead:
        return _FcnHead(self.hidden_size, hidden_size, self._outcome_size, n_layers=2, activation=self.activation)

    def _make_arm_head(self) -> _FcnHead:
        return _FcnHead(self.hidden_size, self.hidden_size // 2, 1, n_layers=2, activation=self.activation)

    def _multiarm_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcnn(x)

    def forward(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.fcnn(batch)
        y_ct = self.fcnn_ct(shared)
        y_tr = self.fcnn_tr(shared)
        treatment = torch.sigmoid(self.lin_treatment(shared))
        eps = self.epsilon(torch.ones_like(treatment)[:, [0]])
        return y_ct, y_tr, treatment, eps

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        if self._is_multi_arm:
            return self._multiarm_step(batch)
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        y_pred_ct, y_pred_tr, treatment_pred, eps_pred = self(x)
        return loss_fn(
            y_true=y_true,
            t_true=treatment_true,
            t_pred=treatment_pred,
            y0_pred=y_pred_ct,
            y1_pred=y_pred_tr,
            eps=eps_pred,
        )

    def _multiarm_step(self, batch: Any) -> dict:
        x, treatment_true, y_true = batch
        x, treatment_true, y_true = self._normalization(x, treatment_true, y_true)
        shared = self.fcnn(x)
        factual = self._factual_mse(self._additive_outcomes(shared), treatment_true, y_true)
        treatment_ce = F.cross_entropy(self.lin_treatment(shared), treatment_true.view(-1).long())
        alpha = getattr(self.loss, 'alpha', 1.0)
        return {'loss': factual + alpha * treatment_ce}

    def predict_step(self, batch: Any, batch_idx: int) -> tuple:
        del batch_idx
        if self._is_multi_arm:
            return self._multiarm_predict_step(batch)
        with torch.no_grad():
            x = self._normalization(batch)
            y_ct_logits, y_tr_logits, treatment_pred, eps = self(x)
            y_ct = self._inv_normalization(self._decode_outcome(y_ct_logits))
            y_tr = self._inv_normalization(self._decode_outcome(y_tr_logits))
        return y_ct, y_tr, treatment_pred, eps

    def predict(self, X: Any, *, return_components: bool = False) -> Any:
        if self._is_multi_arm:
            return self._multiarm_predict(X, return_components=return_components)
        return super().predict(X, return_components=return_components)
