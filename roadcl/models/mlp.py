from __future__ import annotations

import math

import torch
from torch import nn

from roadcl.utils.runtime import seed_everything

N_CLASSES = 3


class FeatureMLP(nn.Module):
    """One-hidden-layer perceptron over window features with a logit for every class."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bag_pool(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Log-mean-exp over the windows of a bag, between mean and max pooling."""
    return torch.logsumexp(logits, dim=dim) - math.log(logits.shape[dim])


def build_network(in_dim: int, hidden: int, seed: int) -> FeatureMLP:
    seed_everything(seed)
    return FeatureMLP(in_dim, hidden)


def parameter_counts(net: nn.Module) -> tuple[int, int]:
    count = sum(p.numel() for p in net.parameters())
    return count, count * 4
