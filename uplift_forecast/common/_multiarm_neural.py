__all__ = ['_MultiArmNeuralMixin']


from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from pytorch_lightning import Trainer
from torch import nn

from ._base_neural import _PredictDataset, _to_float32
from ._data_module import UpliftDataModule


class _MultiArmNeuralMixin:
    """Shared multi-arm (K-arm) machinery for neural uplift models.

    Implements M3TN's additive reparameterization (a shared control head ``mu_0`` plus
    one direct-uplift head ``tau_k`` per treated arm, so ``mu_k = mu_0 + tau_k``)
    together with the factual point-outcome training step and the ``[n, K-1]``
    prediction path. A model builds its own representation network and supplies the
    per-arm representation(s) via ``_multiarm_representation``; everything below the
    representation is shared here. Multi-arm outcome heads are point estimates
    (width 1) and the objective is the factual MSE, regardless of the loss that sizes
    a binary model's heads.

    Treatment is an integer arm in ``{0..K-1}`` (0 = control). ``_multiarm_predict``
    returns one uplift column per treated arm as ``[n, K-1]``, collapsing to a flat
    ``[n]`` array in the binary (``K = 2``) case.
    """

    n_treatments: int

    @property
    def _is_multi_arm(self) -> bool:
        """True when more than one treated arm requires the K-arm additive path."""
        return self.n_treatments > 2

    def _build_additive_heads(self, head_factory: Callable[[], nn.Module]) -> None:
        """A shared control head plus one direct-uplift head per treated arm."""
        self.control_head = head_factory()
        self.uplift_heads = nn.ModuleList(head_factory() for _ in range(self.n_treatments - 1))

    def _additive_outcomes(self, reps: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Per-arm outcomes ``[n, K]`` from a shared representation or one per arm.

        Passing a single tensor reuses it for every arm (shared representation);
        passing a list of ``K`` tensors uses a per-arm representation (e.g. M3TN's
        MMoE gating). Each treated arm is the control prediction plus its uplift head.
        """
        if torch.is_tensor(reps):
            reps = [reps] * self.n_treatments
        mu0 = self.control_head(reps[0])
        columns = [mu0]
        for head, rep in zip(self.uplift_heads, reps[1:], strict=True):
            columns.append(mu0 + head(rep))
        return torch.cat(columns, dim=1)

    def _factual_mse(self, mu: torch.Tensor, treatment: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """MSE of each row's predicted factual arm against the observed outcome."""
        idx = treatment.long().clamp(0, self.n_treatments - 1)
        return F.mse_loss(mu.gather(1, idx), y)

    def _multiarm_representation(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Representation feeding the additive heads — a shared tensor or one per arm."""
        raise NotImplementedError

    def _multiarm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-arm outcomes ``[n, K]`` (factual + counterfactual) for inference."""
        return self._additive_outcomes(self._multiarm_representation(x))

    def _multiarm_factual_step(self, batch: Any) -> dict:
        """Factual-MSE training step (models with extra penalties override ``_step``)."""
        x, treatment, y = batch
        x, treatment, y = self._normalization(x, treatment, y)
        return {'loss': self._factual_mse(self._multiarm_forward(x), treatment, y)}

    def _multiarm_predict_step(self, batch: Any) -> tuple:
        with torch.no_grad():
            x = self._normalization(batch)
            mu = self._inv_normalization(self._multiarm_forward(x))
        return (mu,)

    def _multiarm_predict(
        self,
        X: ArrayLike,
        *,
        return_components: bool = False,
    ) -> Any:
        """Per-arm uplift vs control as ``[n, K-1]`` (flat ``[n]`` when ``K = 2``)."""
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
