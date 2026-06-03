__all__ = ['TwoStageUplift']


from functools import partial
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..losses import TwoStageLoss
from .tarnet import TARNet


class TwoStageUplift(TARNet):
    """Two-stage (conversion-gate x value) revenue-uplift model.

    A shared representation feeds two outcome heads (control / treatment), as in
    ``TARNet``, but the default objective is ``TwoStageLoss``: each head emits a
    conversion logit and a value, the predicted outcome is ``P(convert) * value``,
    and training fits conversion over all units and value over converters only.
    This is the e-commerce ``P(convert) x E[value | convert]`` decomposition.

    ``normalize_y`` defaults to ``False`` because the conversion gate is keyed on
    the true ``y > 0``; rescaling the target (median/mean subtraction) would move
    the zero point and corrupt that indicator.

    Args:
        input_size: Number of input features.
        hidden_size: Width of the representation layers.
        activation: Non-linearity for representation and head layers.
        loss: Training loss; defaults to ``TwoStageLoss()`` (``outputsize=2``).
        valid_loss: Validation loss; defaults to ``loss``.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        valid_batch_size: Validation batch size (defaults to ``batch_size``).
        scaler_type: Feature/target scaler — ``identity``, ``standard``, ``robust``, ``minmax``.
        normalize_y: Whether to scale the target (default ``False``; see above).
        random_seed: Seed used in ``on_fit_start``.
        alias: Display name for this instance.
        optimizer: ``partial(SomeOptimizer, ...)`` (optional).
        optimizer_kwargs: Extra kwargs for the optimizer.
        scheduler: ``partial(SomeScheduler, ...)`` (optional).
        scheduler_kwargs: Extra kwargs for the scheduler.
        dataloader_kwargs: Extra kwargs forwarded to every ``DataLoader``.
        rep_n_layers: Number of hidden layers in the representation network.
        head_n_layers: Number of hidden layers in each outcome head.
        head_hidden_size: Hidden width for outcome heads (defaults to ``hidden_size``).
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
        normalize_y: bool = False,
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
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            loss=loss if loss is not None else TwoStageLoss(),
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
            rep_n_layers=rep_n_layers,
            head_n_layers=head_n_layers,
            head_hidden_size=head_hidden_size,
            **trainer_kwargs,
        )
