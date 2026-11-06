from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distillation import softened_kl, weighted_softened_kl


class GRFDObjective(nn.Module):
    """CE on labels + KD from the RF teacher + optional LwF term against the previous student."""

    def __init__(self, kd_weight: float, lwf_weight: float = 0.0, temperature: float = 3.0, adaptive_kd: bool = False):
        super().__init__()
        self.ce_weight = 1.0 - kd_weight - lwf_weight
        if self.ce_weight <= 0:
            raise ValueError(f"kd_weight + lwf_weight must be < 1, got {kd_weight} + {lwf_weight}")
        self.kd_weight = kd_weight
        self.lwf_weight = lwf_weight
        self.temperature = temperature
        self.adaptive_kd = adaptive_kd

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: torch.Tensor,
        old_probs: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        if self.adaptive_kd:
            kd = weighted_softened_kl(logits, teacher_probs, reliability, self.temperature)
        else:
            kd = softened_kl(logits, teacher_probs, self.temperature)

        loss = self.ce_weight * ce + self.kd_weight * kd
        if self.lwf_weight > 0 and old_probs is not None:
            loss = loss + self.lwf_weight * softened_kl(logits, old_probs, self.temperature)
        return loss
