from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

from .tables import ORIGINAL_VARIANTS, last_fold_predictions

log = logging.getLogger(__name__)


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def plot_continual_curves(timeline: pd.DataFrame, path: Path, title: str = "Continual learning", scale: float = 1.0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for model, group in timeline.groupby("model"):
        mean = group.groupby("after_task")[["accuracy", "macro_f1"]].mean()
        axes[0].plot(mean.index, mean.accuracy * scale, marker="o", label=model)
        axes[1].plot(mean.index, mean.macro_f1, marker="o", label=model)
    for ax, metric in zip(axes, ["Accuracy", "Macro-F1"]):
        ax.set(xlabel="After experience", ylabel=metric, title=f"{title}: {metric}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_training_curves(history: pd.DataFrame, path: Path) -> None:
    cols = ["train_loss", "train_acc", "val_ce"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for model, part in history.groupby("model"):
        curve = part.groupby("epoch")[cols].mean()
        for ax, col in zip(axes, cols):
            ax.plot(curve.index, curve[col], label=model)
    for ax, title in zip(axes, ["Training objective", "Training accuracy", "Validation CE"]):
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def _probe_matrix(probes: pd.DataFrame, model: str) -> np.ndarray:
    sub = probes[probes.model == model]
    return sub.pivot_table(index="after_task", columns="test_probe", values="accuracy", aggfunc="mean").to_numpy()


def _format_probe_axes(ax, title: str, n: int) -> None:
    ticks = np.arange(n)
    ax.set(title=title, xlabel="Random test probe", ylabel="After experience", xticks=ticks, yticks=ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)


def plot_probe_heatmaps(probes: pd.DataFrame, accuracy_path: Path, forgetting_path: Path) -> None:
    models = list(probes.model.unique())
    matrices = {m: _probe_matrix(probes, m) for m in models}
    n = next(iter(matrices.values())).shape[0]

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.2), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[0], models):
        mat = matrices[model]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        _format_probe_axes(ax, model, n)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    _save(fig, accuracy_path)

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.2), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[0], models):
        mat = matrices[model]
        drop = np.full_like(mat, np.nan)
        for i in range(n):
            for j in range(i):
                drop[i, j] = max(0.0, np.max(mat[j:i + 1, j]) - mat[i, j])
        vmax = max(0.01, np.nanmax(drop)) if np.isfinite(drop).any() else 0.01
        im = ax.imshow(np.ma.masked_invalid(drop), vmin=0, vmax=vmax, cmap="magma", aspect="auto")
        _format_probe_axes(ax, model, n)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    _save(fig, forgetting_path)


def plot_efficiency(folds: pd.DataFrame, efficiency: pd.DataFrame, path: Path) -> None:
    joined = folds[["fold", "model", "accuracy"]].merge(efficiency, on=["fold", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, group in joined.groupby("model"):
        axes[0].scatter(group.ms_per_sample_cpu, group.accuracy, label=model)
        axes[1].scatter(group.student_mb, group.accuracy, label=model)
    axes[0].set(xlabel="CPU inference ms / example", ylabel="Accuracy", title="Accuracy vs latency")
    axes[1].set(xlabel="Student weights (MB)", ylabel="Accuracy", title="Accuracy vs size")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_and_roc(predictions: dict, confusion_path: Path, roc_path: Path) -> None:
    models = [m for m in ["joint"] + ORIGINAL_VARIANTS if m in predictions]
    if not models:
        return
    class_names = [str(c) for c in predictions[models[0]]["class_names"]]
    labels = np.arange(len(class_names))

    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        data = predictions[model]
        ConfusionMatrixDisplay.from_predictions(
            data["y_true"], data["probs"].argmax(1), labels=labels, display_labels=class_names,
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30,
        )
        ax.set_title(model)
    fig.tight_layout()
    _save(fig, confusion_path)

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        data = predictions[model]
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((data["y_true"] == i).astype(int), data["probs"][:, i])
            ax.plot(fpr, tpr, label=f"{name}: {auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "--", alpha=0.5)
        ax.set(xlabel="FPR", ylabel="TPR", title=model)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, roc_path)


def plot_c1_size(c1: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, group in c1.groupby("model"):
        ax.scatter(group.mb, group.accuracy * 100, label=model, alpha=0.8)
    ax.set(xlabel="Serialized checkpoint size (MiB)", ylabel="Held-out accuracy (%)", title="C1: accuracy vs checkpoint size")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def make_figures(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    folds = _read(run_dir, "fold_metrics")
    timeline = _read(run_dir, "continual_timeline")
    history = _read(run_dir, "training_history")
    probes = _read(run_dir, "probe_accuracy_matrix_long")
    efficiency = _read(run_dir, "efficiency")
    c1 = _read(run_dir, "C1_compression_per_fold")

    if not timeline.empty:
        plot_continual_curves(timeline, out / "continual_performance_accuracy_f1.png")
        extra = timeline[~timeline.model.isin(ORIGINAL_VARIANTS)]
        if not extra.empty:
            plot_continual_curves(extra, out / "C2_C3_sequential_curves.png", title="C2+C3", scale=100)
    if not history.empty:
        plot_training_curves(history, out / "train_loss_validation_curves.png")
    if not probes.empty:
        plot_probe_heatmaps(
            probes,
            out / "task_wise_accuracy_heatmap_random_probes.png",
            out / "forgetting_heatmap_random_probes.png",
        )
    if not folds.empty and not efficiency.empty:
        plot_efficiency(folds, efficiency, out / "accuracy_efficiency_tradeoff.png")
    if not c1.empty:
        plot_c1_size(c1, out / "C1_accuracy_size.png")

    plot_confusion_and_roc(last_fold_predictions(run_dir), out / "confusion_matrix_models.png", out / "roc_curves_models.png")
