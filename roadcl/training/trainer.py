from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from roadcl.config import TrainingConfig
from roadcl.losses import bag_distillation, task_loss
from roadcl.models.mlp import bag_pool
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


def fisher_diagonal(net: nn.Module, features: torch.Tensor, old_bags: dict[str, np.ndarray],
                    label: dict[str, int], active: list[int], bag_size: int):
    """Empirical diagonal Fisher with one equally weighted contribution per old recording."""
    net.eval()
    fisher = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    for record, ids in old_bags.items():
        chosen = ids[np.linspace(0, len(ids) - 1, min(len(ids), bag_size), dtype=int)]
        score = bag_pool(net(features[chosen].unsqueeze(0))[:, :, active])
        net.zero_grad(set_to_none=True)
        F.cross_entropy(score, torch.tensor([active.index(label[record])])).backward()
        for name, p in net.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().square() / len(old_bags)
    net.zero_grad(set_to_none=True)
    return fisher, {n: p.detach().clone() for n, p in net.named_parameters()}


def _freeze(teacher: nn.Module) -> None:
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)


def fit_model(net: nn.Module, features: torch.Tensor, bags: dict[str, np.ndarray], label: dict[str, int],
              active: list[int], objective: str, steps: int, seed: int, cfg: TrainingConfig,
              distill: Distillation | None = None, *, bag_size: int | None = None,
              lwf: Distillation | None = None, ewc: tuple[dict, dict] | None = None,
              aug: torch.Tensor | None = None, augment_records=(), ace_new: int | None = None) -> FitResult:
    """Class-balanced bag sampling with optional replay distillation, LwF, EWC, augmentation or ER-ACE."""
    if not bags:
        raise ValueError("no training bags")
    bag_size = bag_size or cfg.windows_per_bag
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    by_class = {k: sorted(r for r in bags if label[r] == k) for k in active}
    available = [k for k in active if by_class[k]]
    if not available:
        raise ValueError("no active class has training recordings")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    net.train()
    teacher_records = sorted(distill.memory) if distill else []
    for teacher in (distill, lwf):
        if teacher is not None:
            _freeze(teacher.teacher)

    result = FitResult(train_seconds=0.0)
    start = time.perf_counter()
    for step in range(steps):
        targets = rng.choice(available, size=cfg.bags_per_step, replace=True)
        names, sampled = [], []
        for k in targets:
            names.append(rng.choice(by_class[k]))
            sampled.append(_sample_bag(rng, bags[names[-1]], bag_size))
        batch = np.stack(sampled)
        x = features[batch]
        if aug is not None:
            mask = np.array([r in augment_records for r in names]) & (rng.random(len(names)) < 0.5)
            x[mask] = aug[batch[mask]]
        logits = net(x)[:, :, active]
        y = torch.as_tensor([active.index(int(k)) for k in targets], dtype=torch.long)
        if ace_new is not None:
            # ACE masks old classes on incoming bags only; with one arriving class this term is zero.
            keep = y != active.index(ace_new)
            pooled = bag_pool(logits)
            base = (F.cross_entropy(pooled[keep], y[keep], reduction="sum") / len(y)) if keep.any() else pooled.sum() * 0
        else:
            base = task_loss(logits, y, objective)
        loss = base

        kd_value = gate_mean = float("nan")
        if distill is not None and teacher_records:
            kd_names = rng.choice(teacher_records, size=cfg.kd_bags_per_step, replace=True)
            kd_batch = features[np.stack([
                _sample_bag(rng, distill.memory[r], bag_size) for r in kd_names
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
        if lwf is not None:
            # Distillation on incoming inputs only; no old examples are stored.
            student = bag_pool(net(x)[:, :, lwf.classes])
            with torch.no_grad():
                soft = F.softmax(bag_pool(lwf.teacher(x)[:, :, lwf.classes]) / cfg.temperature, dim=-1)
            loss = loss + lwf.weight * cfg.temperature ** 2 * F.kl_div(
                F.log_softmax(student / cfg.temperature, dim=-1), soft, reduction="batchmean")
        if ewc is not None:
            fisher, reference = ewc
            loss = loss + 0.5 * cfg.ewc_lambda * sum((fisher[n] * (p - reference[n]).square()).sum()
                                                     for n, p in net.named_parameters())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        result.loss_trace.append((step, float(base.detach()), float(loss.detach()), kd_value, gate_mean))
    result.train_seconds = time.perf_counter() - start
    return result
