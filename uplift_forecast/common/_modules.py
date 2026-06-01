__all__ = ['_FcnBlock', '_FcnHead']


import torch
from torch import nn

from ._base_neural import get_activation_fn


class _FcnBlock(nn.Module):
    """Shared fully-connected representation block.

    Stacks `n_layers` linear layers of width `hidden_size`, each followed by
    `activation`. Used as the representation network Φ in TARNet and DragonNet.

    Args:
        in_dim: Input dimension.
        hidden_size: Width of every hidden layer.
        n_layers: Number of linear layers.
        activation: Non-linearity name, passed to `get_activation_fn`.
    """

    def __init__(self, in_dim: int, hidden_size: int, n_layers: int, activation: str):
        super().__init__()
        self.activation_fn = get_activation_fn(activation)
        dims = [in_dim] + [hidden_size] * n_layers
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.output_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x


class _FcnHead(nn.Module):
    """Per-arm outcome head: `n_layers` hidden layers followed by an output projection.

    Args:
        in_dim: Input dimension (representation width).
        hidden_size: Width of the hidden layers.
        output_size: Width of the final projection (1 for scalar; 3 for ZILN).
        n_layers: Number of hidden layers before the output projection.
        activation: Non-linearity name, passed to `get_activation_fn`.
    """

    def __init__(self, in_dim: int, hidden_size: int, output_size: int, n_layers: int, activation: str):
        super().__init__()
        self.activation_fn = get_activation_fn(activation)
        dims = [in_dim] + [hidden_size] * n_layers
        self.hidden = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.out = nn.Linear(dims[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        return self.out(x)
