__all__ = ['CFRNet']


from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..common._base_neural import BaseNeuralUpliftModel, get_activation_fn
from ..common._multiarm_neural import _MultiArmNeuralMixin
from ..losses import CFRLoss, compute_imbalance, safe_sqrt


class _CFRRepresentation(nn.Module):
    """Representation network used by CFRNet."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        nonlin: str,
        varsel: bool,
        batch_norm: bool,
        normalization_mode: str,
        drop_in: float,
        weight_init: float,
    ) -> None:
        super().__init__()
        activation_name = 'ELU' if nonlin.lower() == 'elu' else 'ReLU'
        self.activation = get_activation_fn(activation_name)
        self.drop_in = drop_in
        self.batch_norm = batch_norm
        self.normalization_mode = normalization_mode
        self.input_size = input_size
        self.hidden_size = hidden_size

        if varsel:
            init_scale = torch.ones(input_size) / float(input_size)
            self.varsel_scale = nn.Parameter(init_scale)
        else:
            self.register_parameter('varsel_scale', None)

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        use_layers = max(n_layers - (1 if varsel else 0), 0)
        in_dim = input_size

        for _ in range(use_layers):
            layer = nn.Linear(in_dim, hidden_size)
            self._init_linear(layer, weight_init, in_dim)
            self.layers.append(layer)

            if batch_norm:
                affine = normalization_mode != 'bn_fixed'
                self.bn_layers.append(nn.BatchNorm1d(hidden_size, affine=affine))

            in_dim = hidden_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        if self.varsel_scale is not None:
            hidden = hidden * self.varsel_scale

        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if self.batch_norm and idx < len(self.bn_layers):
                hidden = self.bn_layers[idx](hidden)
            hidden = self.activation(hidden)
            hidden = F.dropout(hidden, p=self.drop_in, training=self.training)
        return hidden

    @staticmethod
    def _init_linear(linear: nn.Linear, weight_init: float, in_dim: int) -> None:
        std = weight_init / torch.sqrt(torch.tensor(float(in_dim)))
        with torch.no_grad():
            linear.weight.normal_(mean=0.0, std=float(std))
            linear.bias.zero_()

    @property
    def linear_layers(self) -> list[nn.Linear]:
        return list(self.layers)

    @property
    def output_dim(self) -> int:
        return self.hidden_size if len(self.layers) > 0 else self.input_size


class _CFROutcomeHead(nn.Module):
    """Outcome head network used by CFRNet."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        nonlin: str,
        drop_out: float,
        weight_init: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        activation_name = 'ELU' if nonlin.lower() == 'elu' else 'ReLU'
        self.activation = get_activation_fn(activation_name)
        self.drop_out = drop_out
        self.out_dim = out_dim

        layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(n_layers):
            layer = nn.Linear(current_dim, hidden_dim)
            self._init_linear(layer, weight_init, current_dim)
            layers.append(layer)
            current_dim = hidden_dim

        pred = nn.Linear(current_dim, out_dim)
        self._init_linear(pred, weight_init, current_dim)

        self._components = nn.ModuleDict({'fcs': layers, 'pred': pred})

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self._components['fcs']:
            hidden = layer(hidden)
            hidden = self.activation(hidden)
            hidden = F.dropout(hidden, p=self.drop_out, training=self.training)
        return self._components['pred'](hidden)

    @staticmethod
    def _init_linear(linear: nn.Linear, weight_init: float, in_dim: int) -> None:
        std = weight_init / torch.sqrt(torch.tensor(float(in_dim)))
        with torch.no_grad():
            linear.weight.normal_(mean=0.0, std=float(std))
            linear.bias.zero_()

    @property
    def module_dict(self) -> nn.ModuleDict:
        return self._components


class CFRNet(_MultiArmNeuralMixin, BaseNeuralUpliftModel):
    """Counterfactual Regression network (Shalit et al., 2017), binary or multi-arm.

    A shared representation feeds either two separate outcome heads
    (`split_output=True`) or a single joint head conditioned on treatment.
    Imbalance between treated and control representations is penalised through
    `CFRLoss`'s IPM term.

    With `n_treatments > 2` the shared (normalised) representation feeds M3TN-style
    additive heads (`mu_k = mu_0 + tau_k`) trained on the factual point MSE, and the
    IPM imbalance is summed over each treated-arm-vs-control pair (using the
    `CFRLoss`'s `p_alpha` / `imb_fun` configuration). `predict` returns
    `[n, n_treatments-1]` (one uplift column per treated arm vs control).

    Args:
        input_size: Number of input features.
        n_treatments: Total number of arms including control (`K`, so `K >= 2`). With
            `K = 2` it is the standard binary CFRNet; with `K > 2` it switches to the
            multi-arm additive-head point-MSE path with per-arm-vs-control IPM.
        hidden_size: Width of the representation layers.
        activation: Non-linearity for representation and head layers.
        loss: Training loss; defaults to `CFRLoss()`.
        valid_loss: Validation loss; defaults to `loss`.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        valid_batch_size: Validation batch size.
        scaler_type: Feature/target scaler.
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
        varsel: Multiply features by a learnable per-feature scale before the
            first representation layer (variable selection trick).
        split_output: Use two separate outcome heads. If False, use one joint
            head that takes the treatment indicator as an extra input.
        batch_norm: Apply batch normalisation after every representation layer.
        normalization: Representation post-normalisation — `'divide'`, `'bn'`,
            `'bn_fixed'`, or `'none'`.
        keep_prob_in: 1 minus the dropout rate inside the representation.
        keep_prob_out: 1 minus the dropout rate inside outcome heads.
        weight_init: Initialisation scaling factor for `nn.Linear` weights.
        **trainer_kwargs: Forwarded to `pytorch_lightning.Trainer`.
    """

    def __init__(
        self,
        input_size: int,
        n_treatments: int = 2,
        hidden_size: int = 256,
        activation: str = 'ReLU',
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
        varsel: bool = False,
        split_output: bool = True,
        batch_norm: bool = False,
        normalization: str = 'none',
        keep_prob_in: float = 1.0,
        keep_prob_out: float = 1.0,
        weight_init: float = 0.1,
        **trainer_kwargs,
    ):
        if n_treatments < 2:
            raise ValueError(f'n_treatments must be >= 2 (control + >=1 treated); got {n_treatments}.')
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            loss=loss if loss is not None else CFRLoss(),
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
        self.rep_hidden_size = hidden_size
        self.head_hidden_size = head_hidden_size or hidden_size
        self.n_in = rep_n_layers
        self.n_out = head_n_layers
        self.varsel = varsel
        self.split_output = split_output
        self.batch_norm = batch_norm
        self.normalization_mode = normalization.lower()
        self.drop_in = 1.0 - keep_prob_in
        self.drop_out = 1.0 - keep_prob_out
        self.weight_init = weight_init

        self.imb_mat: torch.Tensor | None = None

        self.representation = _CFRRepresentation(
            input_size=self.input_size,
            hidden_size=self.rep_hidden_size,
            n_layers=self.n_in,
            nonlin=activation,
            varsel=self.varsel,
            batch_norm=self.batch_norm,
            normalization_mode=self.normalization_mode,
            drop_in=self.drop_in,
            weight_init=self.weight_init,
        )
        self._rep_dim = self.representation.output_dim
        self._activation = activation

        # Head width follows the loss output size (1 scalar by default, 3 for ZILN).
        self._build_output_heads()

    def _build_output_heads(self) -> None:
        if self._is_multi_arm:
            self._build_additive_heads(self._make_arm_head)
            return
        head_kwargs = {
            'hidden_dim': self.head_hidden_size,
            'n_layers': self.n_out,
            'nonlin': self._activation,
            'drop_out': self.drop_out,
            'weight_init': self.weight_init,
            'out_dim': self._outcome_size,
        }
        if self.split_output:
            self.head0 = _CFROutcomeHead(in_dim=self._rep_dim, **head_kwargs)
            self.head1 = _CFROutcomeHead(in_dim=self._rep_dim, **head_kwargs)
            self.joint_head = None
        else:
            self.head0 = None
            self.head1 = None
            self.joint_head = _CFROutcomeHead(in_dim=self._rep_dim + 1, **head_kwargs)
        self._head_out_dim = self._outcome_size

    def _make_arm_head(self) -> _CFROutcomeHead:
        return _CFROutcomeHead(
            in_dim=self._rep_dim,
            hidden_dim=self.head_hidden_size,
            n_layers=self.n_out,
            nonlin=self._activation,
            drop_out=self.drop_out,
            weight_init=self.weight_init,
            out_dim=1,
        )

    def _multiarm_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize_representation(self.representation(x))

    def _multiarm_imbalance(self, reps_norm: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        """IPM penalty summed over each treated-arm-vs-control pair (CFRLoss config)."""
        p_alpha = getattr(self.loss, 'p_alpha', 0.0)
        if p_alpha <= 0:
            return reps_norm.new_zeros(())
        t = treatment.view(-1).long()
        use_p_correction = getattr(self.loss, 'use_p_correction', False)
        total = reps_norm.new_zeros(())
        for arm in range(1, self.n_treatments):
            mask = (t == 0) | (t == arm)
            sub_t = (t[mask] == arm).float()
            if sub_t.sum() == 0 or (sub_t == 0).sum() == 0:
                continue
            propensity = torch.full((int(mask.sum()), 1), float(sub_t.mean()) if use_p_correction else 0.5)
            imb_loss, _, _ = compute_imbalance(
                reps_norm[mask],
                sub_t.unsqueeze(1),
                propensity.to(reps_norm.device),
                imb_fun=getattr(self.loss, 'imb_fun', 'mmd2_lin'),
                r_alpha=getattr(self.loss, 'r_alpha', 1.0),
                rbf_sigma=getattr(self.loss, 'rbf_sigma', 0.1),
                wass_lambda=getattr(self.loss, 'wass_lambda', 10.0),
                wass_iterations=getattr(self.loss, 'wass_iterations', 50),
                wass_bpt=getattr(self.loss, 'wass_bpt', False),
            )
            total = total + imb_loss
        return p_alpha * total

    def forward(
        self,
        features: torch.Tensor,
        treatment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        treatment = treatment.float()
        if treatment.dim() == 1:
            treatment = treatment.unsqueeze(1)

        representations = self.representation(features)
        representations_norm = self._normalize_representation(representations)
        predictions = self._forward_heads(representations_norm, treatment)
        return predictions, representations, representations_norm

    def _normalize_representation(self, representations: torch.Tensor) -> torch.Tensor:
        if self.normalization_mode == 'divide':
            norm = safe_sqrt(torch.sum(representations**2, dim=1, keepdim=True))
            return representations / norm
        if self.normalization_mode in {'bn', 'bn_fixed'}:
            return (representations - representations.mean(dim=0)) / (
                representations.std(dim=0) + 1e-6
            )
        return representations

    def _forward_heads(
        self,
        representations: torch.Tensor,
        treatment: torch.Tensor,
    ) -> torch.Tensor:
        if self.split_output:
            treated_mask = (treatment.view(-1) > 0.5).to(representations.device)
            predictions = torch.empty(
                representations.size(0), self._head_out_dim, device=representations.device,
            )
            if treated_mask.any():
                predictions[treated_mask] = self.head1(representations[treated_mask])
            if (~treated_mask).any():
                predictions[~treated_mask] = self.head0(representations[~treated_mask])
            return predictions

        head_input = torch.cat([representations, treatment], dim=1)
        return self.joint_head(head_input)

    def _head_modules(self) -> list[nn.ModuleDict]:
        if self.split_output:
            return [head.module_dict for head in (self.head0, self.head1) if head is not None]
        return [self.joint_head.module_dict] if self.joint_head is not None else []

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        if self._is_multi_arm:
            return self._multiarm_step(batch)
        features, treatment, factual = batch
        x, treatment_true, y_true = self._normalization(features, treatment, factual)
        propensity = torch.full_like(treatment_true, treatment_true.mean())

        preds, _, reps_norm = self(x, treatment_true)
        y_pred_ct, _, _ = self(x, torch.zeros_like(treatment_true))
        y_pred_tr, _, _ = self(x, torch.ones_like(treatment_true))

        loss = loss_fn(
            y_true=y_true,
            t_true=treatment_true,
            t_pred=propensity,
            y0_pred=y_pred_ct,
            y1_pred=y_pred_tr,
            y_pred=preds,
            representations_norm=reps_norm,
            rep_layers=self.representation.linear_layers,
            head_modules=self._head_modules(),
        )
        imb_mat = loss.pop('imb_mat', None)
        if imb_mat is not None:
            self.imb_mat = imb_mat
        return loss

    def _multiarm_step(self, batch: Any) -> dict:
        features, treatment, factual = batch
        x, treatment_true, y_true = self._normalization(features, treatment, factual)
        reps_norm = self._normalize_representation(self.representation(x))
        mu = self._additive_outcomes(reps_norm)
        return {'loss': self._factual_mse(mu, treatment_true, y_true) + self._multiarm_imbalance(reps_norm, treatment_true)}

    def predict_step(self, batch: Any, batch_idx: int) -> tuple:
        del batch_idx
        if self._is_multi_arm:
            return self._multiarm_predict_step(batch)
        with torch.no_grad():
            features = self._normalization(batch)
            shape = (features.shape[0], 1)
            predictions_t, _, _ = self(features, torch.ones(shape, device=features.device))
            predictions_c, _, _ = self(features, torch.zeros(shape, device=features.device))
            predictions_t = self._inv_normalization(self._decode_outcome(predictions_t))
            predictions_c = self._inv_normalization(self._decode_outcome(predictions_c))
        return predictions_c, predictions_t

    def predict(self, X: Any, *, return_components: bool = False) -> Any:
        if self._is_multi_arm:
            return self._multiarm_predict(X, return_components=return_components)
        return super().predict(X, return_components=return_components)
