from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from roadcl.models.mlp import bag_pool


def predict_recording(net: nn.Module, features: torch.Tensor, ids: np.ndarray,
                      active: list[int]) -> tuple[int, np.ndarray]:
    """Pool all windows of one recording and return the winning class and its probabilities."""
    net.eval()
    with torch.inference_mode():
        pooled = bag_pool(net(features[ids])[:, active], dim=0)
        probs = F.softmax(pooled, dim=-1).cpu().numpy()
    return active[int(np.argmax(probs))], probs


def cpu_latency_ms(net: nn.Module, batch: torch.Tensor, repeats: int, warmup: int = 3) -> float:
    """Mean inference time per window over `repeats` timed passes."""
    net.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            net(batch)
        start = time.perf_counter()
        for _ in range(repeats):
            net(batch)
    return (time.perf_counter() - start) * 1000 / (repeats * max(1, len(batch)))
