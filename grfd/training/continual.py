from __future__ import annotations

import copy
import gc
import json
import logging
import math

import numpy as np
import torch
import torch.nn as nn

from grfd.config import Config, VariantConfig
from grfd.data import ClassBalancedReservoir
from grfd.metrics import classification_metrics, continual_statistics, probe_metrics, student_footprint
from grfd.models import build_student, predict_proba, save_checkpoint

from .context import FoldContext, Recorder
from .trainer import train_student

log = logging.getLogger(__name__)


def _frozen_copy(model: nn.Module) -> nn.Module:
    old = copy.deepcopy(model).eval()
    for p in old.parameters():
        p.requires_grad_(False)
    return old


def _probe_accuracy(model: nn.Module, ctx: FoldContext, probe: np.ndarray, batch_size: int) -> float:
    probs = predict_proba(model, ctx.X[probe], batch_size)
    return probe_metrics(ctx.y[probe], probs, ctx.n_classes)["accuracy"]


def run_variant(ctx: FoldContext, name: str, spec: VariantConfig, cfg: Config, rec: Recorder, device: torch.device) -> None:
    """Train one student sequentially over the fold's experiences and log probe accuracy after each one."""
    fold = ctx.fold
    seed = cfg.experiment.seed
    batch_size = cfg.training.eval_batch_size
    out_dir = ctx.fold_dir / name
    out_dir.mkdir(exist_ok=True)
    n_tasks = len(ctx.tasks)
    row_bytes = ctx.X.shape[1] * np.dtype(np.float32).itemsize + np.dtype(np.int64).itemsize

    model = build_student(ctx.X.shape[1], ctx.n_classes, cfg.student, seed, device)
    initial = np.array([_probe_accuracy(model, ctx, probe, batch_size) for probe in ctx.probes])
    memory = ClassBalancedReservoir(math.ceil(spec.replay / ctx.n_classes), seed + fold)
    matrix = np.zeros((n_tasks, n_tasks))
    empty = np.asarray([], dtype=int)

    for t, current in enumerate(ctx.tasks):
        task = t + 1
        replay = memory.indices() if t and spec.replay else empty
        old = _frozen_copy(model) if t and spec.lwf else None
        log.info("FOLD %d | %s | experience %d/%d | new=%s replay=%s", fold, name, task, n_tasks, f"{len(current):,}", f"{len(replay):,}")

        step = train_student(
            model, ctx.X, ctx.y, np.concatenate([current, replay]), ctx.split.val, cfg,
            name=f"fold{fold}_{name}_task{task}",
            n_classes=ctx.n_classes,
            device=device,
            old_model=old,
            lwf_weight=cfg.training.lwf_weight if spec.lwf else 0.0,
            adaptive_kd=spec.adaptive_kd,
            teacher_path=out_dir / f"teacher_task_{task}.joblib" if cfg.continual.save_task_teachers else None,
        )
        model = step.model
        rec.add("C3_update_costs", {
            "fold": fold,
            "variant": name,
            "after_task": task,
            "new_rows": len(current),
            "replay_rows": len(replay),
            "rf_fit_seconds": step.teacher_seconds,
            "student_update_seconds": step.seconds - step.teacher_seconds,
            "total_update_seconds": step.seconds,
        })
        rec.extend("training_history", step.history.assign(fold=fold, model=name, task=task).to_dict("records"))

        ckpt_meta = dict(feature_cols=ctx.feature_cols, class_names=ctx.class_names, fold=fold, after_task=task, variant=name)
        save_checkpoint(model, out_dir / f"student_task_{task}.pt", **ckpt_meta)
        if name == "replay_lwf":
            save_checkpoint(model, ctx.fold_dir / f"student_task_{task}.pt", **ckpt_meta)

        probs = predict_proba(model, ctx.X_test, batch_size)
        metrics = classification_metrics(ctx.y_test, probs, ctx.class_names)
        rec.add("continual_timeline", {"fold": fold, "model": name, "after_task": task, **metrics})

        for j, probe in enumerate(ctx.probes):
            pm = probe_metrics(ctx.y[probe], predict_proba(model, ctx.X[probe], batch_size), ctx.n_classes)
            matrix[t, j] = pm["accuracy"]
            rec.add("probe_accuracy_matrix_long", {"fold": fold, "model": name, "after_task": task, "test_probe": j + 1, **pm})

        if spec.replay:
            memory.update(current, ctx.y)
        kept = len(memory)
        rec.add("memory_budget", {
            "fold": fold,
            "model": name,
            "after_task": task,
            "replay_examples": kept,
            "memory_bytes_raw": kept * row_bytes,
            "memory_mb_raw": kept * row_bytes / 1024**2,
            "memory_capacity_examples": spec.replay,
            "memory_counts": json.dumps({ctx.class_names[c]: n for c, n in memory.counts().items()}),
        })
        log.info(
            "POOLED TEST acc=%.4f macro_f1=%.4f pothole_recall=%.4f",
            metrics["accuracy"], metrics["macro_f1"], metrics.get("recall_pothole", float("nan")),
        )
        del old
        gc.collect()

    np.save(out_dir / "probe_accuracy_matrix.npy", matrix)
    np.save(out_dir / "initial_probe_accuracy.npy", initial)
    stats = continual_statistics(matrix, initial)
    rec.add("continual_summary_per_fold", {
        "fold": fold,
        "model": name,
        **stats,
        "final_replay_examples": len(memory),
        "final_memory_mb_raw": len(memory) * row_bytes / 1024**2,
    })
    rec.add("fold_metrics", {"fold": fold, "model": name, **metrics})
    ctx.save_predictions(name, probs)
    rec.add("efficiency", {"fold": fold, "model": name, **student_footprint(model, ctx.X_test[:2048])})
    log.info("FINAL %s %s", name, {k: round(v, 4) for k, v in stats.items()})
    del model
    gc.collect()
