from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_targets(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(probs.clamp_min(1e-8).log() / temperature, dim=1)


def softened_kl(student_logits: torch.Tensor, target_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    target = _soft_targets(target_probs, temperature)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_student, target, reduction="batchmean") * temperature**2


def weighted_softened_kl(
    student_logits: torch.Tensor,
    target_probs: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    target = _soft_targets(target_probs, temperature)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    per_row = F.kl_div(log_student, target, reduction="none").sum(dim=1) * temperature**2
    return (per_row * weights).mean()
