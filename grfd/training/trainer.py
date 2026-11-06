from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from grfd.config import Config
from grfd.losses import GRFDObjective
from grfd.models import aligned_proba, build_teacher, oob_reliability, predict_proba

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    model: nn.Module
    history: pd.DataFrame
    teacher_seconds: float
    seconds: float


def _validation_ce(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[start:start + batch_size]).to(device))
            targets = torch.from_numpy(y[start:start + batch_size]).to(device)
            total += F.cross_entropy(logits, targets, reduction="sum").item()
    return total / len(X)


def train_student(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    *,
    name: str,
    n_classes: int,
    device: torch.device,
    old_model: nn.Module | None = None,
    lwf_weight: float = 0.0,
    adaptive_kd: bool = False,
    teacher_path: Path | None = None,
) -> StepResult:
    """Fit a fresh RF teacher on train_idx, then distil it into the student with early stopping on val CE."""
    tcfg = cfg.training
    seed = cfg.experiment.seed
    started = time.perf_counter()
    train_idx = np.asarray(train_idx, dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    teacher = build_teacher(cfg.teacher, seed, oob=adaptive_kd)
    fit_start = time.perf_counter()
    teacher.fit(X_train, y_train)
    teacher_seconds = time.perf_counter() - fit_start
    log.info("%s: RF train=%.4f val=%.4f", name, teacher.score(X_train, y_train), teacher.score(X_val, y_val))
    if teacher_path is not None:
        joblib.dump(teacher, teacher_path)

    teacher_probs = aligned_proba(teacher, X_train, n_classes)
    if old_model is not None:
        old_probs = predict_proba(old_model, X_train, tcfg.eval_batch_size).astype(np.float32)
    else:
        old_probs = np.zeros_like(teacher_probs)

    reliability = np.ones(len(train_idx), dtype=np.float32)
    if adaptive_kd:
        reliability = oob_reliability(teacher, y_train, teacher_probs, n_classes, cfg.reliability, seed)
        log.info(
            "  OOB reliability weights: mean=%.3f min=%.3f max=%.3f",
            reliability.mean(), reliability.min(), reliability.max(),
        )
    del teacher

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(teacher_probs),
            torch.from_numpy(old_probs),
            torch.from_numpy(reliability),
        ),
        batch_size=tcfg.batch_size,
        shuffle=True,
    )
    objective = GRFDObjective(
        kd_weight=tcfg.kd_weight,
        lwf_weight=lwf_weight if old_model is not None else 0.0,
        temperature=tcfg.temperature,
        adaptive_kd=adaptive_kd,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    best_ce = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for xb, yb, tb, ob, rb in loader:
            xb, yb, tb, ob, rb = (t.to(device) for t in (xb, yb, tb, ob, rb))
            logits = model(xb)
            loss = objective(logits, yb, tb, ob, rb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()

            running_loss += loss.item() * len(yb)
            correct += (logits.argmax(-1) == yb).sum().item()
            seen += len(yb)

        val_ce = _validation_ce(model, X_val, y_val, tcfg.eval_batch_size, device)
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "train_acc": correct / seen, "val_ce": val_ce})
        if epoch == 1 or epoch % tcfg.log_every == 0 or epoch == tcfg.epochs:
            log.info("  epoch %3d train_acc=%.4f train_loss=%.4f val_ce=%.4f", epoch, correct / seen, running_loss / seen, val_ce)

        if val_ce < best_ce - tcfg.min_delta:
            best_ce = val_ce
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= tcfg.patience:
                log.info("  early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    log.info("%s: best val_ce=%.4f in %.1fs", name, best_ce, seconds)

    frame = pd.DataFrame(history)
    frame["rf_fit_seconds"] = teacher_seconds
    frame["teacher_reliability_mode"] = "oob_correctness" if adaptive_kd else "fixed"
    del loader, teacher_probs, old_probs
    gc.collect()
    return StepResult(model=model, history=frame, teacher_seconds=teacher_seconds, seconds=seconds)
