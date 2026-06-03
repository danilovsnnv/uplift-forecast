__all__ = ['BaseNeuralUpliftModel', 'get_activation_fn']


import random
import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from ._data_module import UpliftDataModule
from ._scalers import FeatureScaler
from ._uplift_model import UpliftModel, _to_numpy_1d


_ACTIVATION_MAP: dict[str, Callable] = {
    'ReLU': F.relu,
    'Softplus': F.softplus,
    'Tanh': F.tanh,
    'SELU': F.selu,
    'LeakyReLU': F.leaky_relu,
    'Sigmoid': F.sigmoid,
    'ELU': F.elu,
    'GLU': F.glu,
}


def get_activation_fn(activation: str) -> Callable:
    """Return the torch activation function for `activation` name. Defaults to ELU."""
    return _ACTIVATION_MAP.get(activation, F.elu)


def _to_float32(arr: ArrayLike) -> np.ndarray:
    if hasattr(arr, 'to_numpy'):
        arr = arr.to_numpy()
    return np.asarray(arr, dtype='float32')


def _train_dataset(X: ArrayLike, treatment: ArrayLike, y: ArrayLike) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(_to_float32(X)),
        torch.from_numpy(_to_numpy_1d(treatment).astype('float32').reshape(-1, 1)),
        torch.from_numpy(_to_numpy_1d(y).astype('float32').reshape(-1, 1)),
    )


class _PredictDataset(Dataset):
    """Single-tensor dataset for inference (returns tensors, not tuples)."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


class BaseNeuralUpliftModel(LightningModule, UpliftModel):
    """Base for PyTorch Lightning uplift models.

    Handles optimizer / scheduler wiring, feature scaling, and the Lightning
    Trainer lifecycle. Subclasses implement the architecture by overriding:

    - ``forward`` and ``predict_step`` — model-specific inference.
    - ``_step(batch, loss_fn) -> dict`` — one forward + loss pass; dict must
      contain a ``'loss'`` key. ``training_step`` and ``validation_step`` call
      this and handle logging, so subclasses do not repeat the boilerplate.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Width of hidden layers.
        activation (str): Activation name, see get_activation_fn.
        loss (nn.Module): Training loss.
        valid_loss (nn.Module): Validation loss; defaults to loss.
        learning_rate (float): Optimizer learning rate.
        batch_size (int): Training batch size.
        valid_batch_size (int): Inference / validation batch size.
        scaler_type (str): Feature scaler — 'identity', 'standard', 'robust', 'minmax'.
        normalize_y (bool): Whether to scale the target.
        y_transform (callable): Optional transform applied to y after scaling.
        y_inv_transform (callable): Inverse of y_transform, applied at predict time.
        random_seed (int): Seed for on_fit_start reproducibility.
        alias (str): Display name used by UpliftForecast.
        optimizer (partial): partial(SomeOptimizer, ...).
        optimizer_kwargs (dict): Extra optimizer kwargs; lr is overridden by learning_rate.
        scheduler (partial): partial(SomeScheduler, ...).
        scheduler_kwargs (dict): Extra scheduler kwargs.
        dataloader_kwargs (dict): Forwarded to every DataLoader.
        **trainer_kwargs: Forwarded to pytorch_lightning.Trainer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 200,
        activation: str = 'ReLU',
        loss: nn.Module | None = None,
        valid_loss: nn.Module | None = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        valid_batch_size: int | None = None,
        scaler_type: str = 'robust',
        normalize_y: bool = True,
        y_transform: Callable | None = None,
        y_inv_transform: Callable | None = None,
        random_seed: int | None = None,
        alias: str | None = None,
        optimizer: partial | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler: partial | None = None,
        scheduler_kwargs: dict | None = None,
        dataloader_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['loss', 'valid_loss', 'y_transform', 'y_inv_transform'])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.loss = loss if loss is not None else nn.L1Loss()
        self.valid_loss = valid_loss or self.loss

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size or batch_size

        self.scaler_type = scaler_type
        self.normalize_y = normalize_y
        self.y_transform = y_transform
        self.y_inv_transform = y_inv_transform

        self.scaler = FeatureScaler(scaler_type=scaler_type)
        self.y_scaler = FeatureScaler(scaler_type=scaler_type)

        self.random_seed = random_seed
        self.alias = alias

        if optimizer is not None and not issubclass(optimizer.func, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be a partial wrapping a torch.optim.Optimizer subclass.'
            )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        if scheduler is not None and not issubclass(scheduler.func, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(
                'scheduler must be a partial wrapping a torch.optim.lr_scheduler.LRScheduler subclass.'
            )
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.dataloader_kwargs = dataloader_kwargs or {}
        self.trainer_kwargs = trainer_kwargs

    @property
    def _outcome_size(self) -> int:
        """Per-arm outcome-head width required by the current loss (1 if unspecified)."""
        return getattr(self.loss, 'outputsize', 1)

    def _decode_outcome(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw outcome-head output to a predicted outcome using the loss's decoder."""
        decode = getattr(self.loss, 'decode', None)
        return decode(logits) if decode is not None else logits

    def set_loss(self, loss: nn.Module, valid_loss: nn.Module | None = None) -> None:
        """Replace the training loss and rebuild outcome heads to match `loss.outputsize`.

        Lets a framework (e.g. RERUM) swap in a different objective on an already
        constructed model without the model knowing about that framework.
        """
        self.loss = loss
        self.valid_loss = valid_loss if valid_loss is not None else loss
        self._build_output_heads()

    def _build_output_heads(self) -> None:
        """(Re)build the outcome heads sized to `self._outcome_size`.

        Subclasses with loss-dependent output heads override this. The default is
        a no-op for models whose heads do not depend on the loss output size.
        """

    def _step(self, batch: Any, loss_fn: nn.Module) -> dict:
        """Single forward + loss pass. Must return a dict with a 'loss' key."""
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self._step(batch, self.loss)
        if torch.isnan(loss['loss']):
            raise RuntimeError(f'{type(self).__name__} training loss is NaN.')
        self.log('train_loss', loss['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss['loss']

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        with torch.no_grad():
            loss = self._step(batch, self.valid_loss)
        self.log('val_loss', loss['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss['loss']

    def configure_optimizers(self) -> dict:
        opt_cls = self.optimizer if self.optimizer is not None else torch.optim.Adam
        opt_kwargs = deepcopy(self.optimizer_kwargs)
        if 'lr' in opt_kwargs:
            warnings.warn("'lr' in optimizer_kwargs is ignored; use learning_rate instead.", stacklevel=2)
        opt_kwargs['lr'] = self.learning_rate
        optimizer = opt_cls(params=self.parameters(), **opt_kwargs)

        config = {'optimizer': optimizer}
        if self.scheduler is not None:
            config['lr_scheduler'] = {
                'scheduler': self.scheduler(optimizer=optimizer, **self.scheduler_kwargs),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        return config

    def on_fit_start(self) -> None:
        if self.random_seed is None:
            return
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        seed_everything(self.random_seed, workers=True)

    def __repr__(self) -> str:
        return self.alias or type(self).__name__

    def fit(
        self,
        X: ArrayLike,
        treatment: ArrayLike,
        y: ArrayLike,
        eval_set: tuple | None = None,
        **_,
    ) -> 'BaseNeuralUpliftModel':
        """Train on (X, treatment, y). eval_set=(X_val, t_val, y_val) is optional."""
        self._fit_scalers(X, y)
        datamodule = UpliftDataModule(
            train_dataset=_train_dataset(X, treatment, y),
            valid_dataset=_train_dataset(*eval_set) if eval_set is not None else None,
            batch_size=self.batch_size,
            valid_batch_size=self.valid_batch_size,
            shuffle_train=True,
            **self.dataloader_kwargs,
        )
        trainer = Trainer(**self.trainer_kwargs)
        trainer.fit(self, datamodule=datamodule)
        self.metrics = trainer.callback_metrics
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict uplift. return_components=True also returns y0, y1."""
        dataset = _PredictDataset(torch.from_numpy(_to_float32(X)))
        datamodule = UpliftDataModule(
            predict_dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **self.dataloader_kwargs,
        )
        trainer = Trainer(**self.trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        y_ct, y_tr, *_ = (torch.vstack(parts) for parts in zip(*fcsts, strict=False))
        y0 = y_ct.cpu().numpy().reshape(-1)
        y1 = y_tr.cpu().numpy().reshape(-1)
        uplift = y1 - y0
        if return_components:
            return uplift, y0, y1
        return uplift

    def _fit_scalers(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit feature/target scalers on the full training data, once.

        Keeps normalization statistics fixed across batches and at predict time,
        so a row's prediction does not depend on the batch it is scored in.
        """
        self.scaler.fit(torch.from_numpy(_to_float32(X)))
        if self.normalize_y:
            y_t = torch.from_numpy(_to_numpy_1d(y).astype('float32').reshape(-1, 1))
            self.y_scaler.fit(y_t)

    def _normalization(
        self,
        x: torch.Tensor,
        treatment: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> Any:
        x = self.scaler.transform(x)
        if y is None:
            return x
        if self.normalize_y:
            y = self.y_scaler.transform(y)
        if self.y_transform is not None:
            y = self.y_transform(y)
        return x, treatment, y

    def _inv_normalization(self, y_hat: torch.Tensor) -> torch.Tensor:
        if self.y_inv_transform is not None:
            y_hat = self.y_inv_transform(y_hat)
        if not self.normalize_y:
            return y_hat
        return self.y_scaler.inverse_transform(y_hat)
