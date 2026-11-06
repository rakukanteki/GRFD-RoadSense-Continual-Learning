from __future__ import annotations

import gc
import logging

import joblib
import torch
from sklearn.metrics import accuracy_score, f1_score

from grfd.config import Config
from grfd.metrics import classification_metrics, student_footprint, teacher_latency_ms
from grfd.models import build_student, predict_proba, save_checkpoint

from .context import FoldContext, Recorder
from .trainer import train_student

log = logging.getLogger(__name__)


def run_joint(ctx: FoldContext, cfg: Config, rec: Recorder, device: torch.device) -> None:
    """Train the GRFD student on the full fold train split and compare it with its RF teacher (C1)."""
    fold = ctx.fold
    teacher_path = ctx.fold_dir / "joint_teacher.joblib"
    student_path = ctx.fold_dir / "joint_student.pt"

    model = build_student(ctx.X.shape[1], ctx.n_classes, cfg.student, cfg.experiment.seed, device)
    step = train_student(
        model, ctx.X, ctx.y, ctx.split.train, ctx.split.val, cfg,
        name=f"fold{fold}_joint", n_classes=ctx.n_classes, device=device, teacher_path=teacher_path,
    )
    rec.extend("training_history", step.history.assign(fold=fold, model="joint", task=0).to_dict("records"))
    save_checkpoint(step.model, student_path, feature_cols=ctx.feature_cols, class_names=ctx.class_names, fold=fold, after_task=0)

    X_test, y_test = ctx.X_test, ctx.y_test
    probs = predict_proba(step.model, X_test, cfg.training.eval_batch_size)
    metrics = classification_metrics(y_test, probs, ctx.class_names)
    rec.add("fold_metrics", {"fold": fold, "model": "joint", **metrics})
    ctx.save_predictions("joint", probs)

    sample = X_test[:2048]
    footprint = student_footprint(step.model, sample)
    rec.add("efficiency", {"fold": fold, "model": "joint", **footprint})

    teacher = joblib.load(teacher_path)
    rf_pred = teacher.predict(X_test)
    rec.extend("C1_compression_per_fold", [
        {
            "fold": fold,
            "model": "Random Forest teacher",
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "macro_f1": float(f1_score(y_test, rf_pred, average="macro", zero_division=0)),
            "bytes": teacher_path.stat().st_size,
            "mb": teacher_path.stat().st_size / 1024**2,
            "cpu_ms_per_sample": teacher_latency_ms(teacher, sample),
            "train_seconds": step.teacher_seconds,
            "parameters": float("nan"),
        },
        {
            "fold": fold,
            "model": "Gated student",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "bytes": student_path.stat().st_size,
            "mb": student_path.stat().st_size / 1024**2,
            "cpu_ms_per_sample": footprint["ms_per_sample_cpu"],
            "train_seconds": step.seconds - step.teacher_seconds,
            "parameters": footprint["params"],
        },
    ])
    log.info("JOINT acc=%.4f macro_f1=%.4f auc=%.4f", metrics["accuracy"], metrics["macro_f1"], metrics["auc_ovr"])
    del teacher, model, step
    gc.collect()
