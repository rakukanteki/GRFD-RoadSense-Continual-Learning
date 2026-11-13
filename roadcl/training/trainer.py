from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from roadcl.config import TrainingConfig
from roadcl.losses import bag_distillation, task_loss
from roadcl.utils.runtime import seed_everything


@dataclass
class FitResult:
    train_seconds: float
    loss_trace: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    gate_trace: list[tuple[int, float, float, float]] = field(default_factory=list)

    @property
    def last_loss(self) -> float:
        return self.loss_trace[-1][2]


@dataclass
class Distillation:
    teacher: nn.Module
    classes: list[int]
    memory: dict[str, np.ndarray]
    gated: bool
    weight: float


def _sample_bag(rng: np.random.Generator, ids: np.ndarray, size: int) -> np.ndarray:
    return rng.choice(ids, size=size, replace=len(ids) < size)


def fit_model(net: nn.Module, features: torch.Tensor, bags: dict[str, np.ndarray], label: dict[str, int],
              active: list[int], objective: str, steps: int, seed: int, cfg: TrainingConfig,
              distill: Distillation | None = None) -> FitResult:
    """Class-balanced bag sampling with an optional replay-only distillation batch per step."""
    if not bags:
        raise ValueError("no training bags")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    by_class = {k: sorted(r for r in bags if label[r] == k) for k in active}
    available = [k for k in active if by_class[k]]
    if not available:
        raise ValueError("no active class has training recordings")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    net.train()
    teacher_records = sorted(distill.memory) if distill else []
    if distill is not None:
        distill.teacher.eval()
        for p in distill.teacher.parameters():
            p.requires_grad_(False)

    result = FitResult(train_seconds=0.0)
    start = time.perf_counter()
    for step in range(steps):
        targets = rng.choice(available, size=cfg.bags_per_step, replace=True)
        batch = np.stack([
            _sample_bag(rng, bags[rng.choice(by_class[k])], cfg.windows_per_bag) for k in targets
        ])
        logits = net(features[batch])[:, :, active]
        y = torch.as_tensor([active.index(int(k)) for k in targets], dtype=torch.long)
        base = task_loss(logits, y, objective)
        loss = base

        kd_value = gate_mean = float("nan")
        if distill is not None and teacher_records:
            names = rng.choice(teacher_records, size=cfg.kd_bags_per_step, replace=True)
            kd_batch = features[np.stack([
                _sample_bag(rng, distill.memory[r], cfg.windows_per_bag) for r in names
            ])]
            kd, per_bag, confidence, gate = bag_distillation(
                net(kd_batch)[:, :, distill.classes],
                distill.teacher(kd_batch)[:, :, distill.classes],
                cfg.temperature, distill.gated, cfg.gate_power,
            )
            loss = loss + distill.weight * kd
            kd_value = float(per_bag.mean().detach())
            gate_mean = float(gate.mean())
            result.gate_trace.extend(
                (step, float(c), float(g), float(b))
                for c, g, b in zip(confidence.numpy(), gate.numpy(), per_bag.detach().numpy())
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        result.loss_trace.append((step, float(base.detach()), float(loss.detach()), kd_value, gate_mean))
    result.train_seconds = time.perf_counter() - start
    return result
