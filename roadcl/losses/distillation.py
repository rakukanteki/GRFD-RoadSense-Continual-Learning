from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from roadcl.models.mlp import bag_pool


def task_loss(logits: torch.Tensor, targets: torch.Tensor, objective: str) -> torch.Tensor:
    """Bag cross-entropy on pooled logits, or window cross-entropy with the recording label."""
    if objective == "bag":
        return F.cross_entropy(bag_pool(logits), targets)
    if objective == "window":
        return F.cross_entropy(logits.flatten(0, 1), targets.repeat_interleave(logits.shape[1]))
    raise ValueError(f"unknown objective {objective!r}")


def teacher_confidence(probs: torch.Tensor) -> torch.Tensor:
    """One minus the normalized entropy of the teacher distribution."""
    entropy_term = (probs * probs.clamp_min(1e-9).log()).sum(-1) / math.log(probs.shape[-1])
    return (1 + entropy_term).clamp(0, 1)


def bag_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     temperature: float, gated: bool, gate_power: float = 1.0):
    """Per-bag KL to the frozen stage-1 model, optionally weighted by its confidence."""
    student = bag_pool(student_logits)
    with torch.no_grad():
        teacher_probs = F.softmax(bag_pool(teacher_logits) / temperature, dim=-1)
        confidence = teacher_confidence(teacher_probs)
        gate = (confidence ** gate_power).detach() if gated else torch.ones_like(confidence)
    per_bag = F.kl_div(F.log_softmax(student / temperature, dim=-1), teacher_probs,
                       reduction="none").sum(-1) * temperature ** 2
    return (per_bag * gate).mean(), per_bag, confidence, gate
