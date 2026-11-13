from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ABLATION_KEYS = {
    "memory_sweep": "memory_per_class",
    "stratified_memory": "memory_per_class",
    "representation": "representation",
    "supervision": "objective",
    "gate_ablation": "kd_lambda",
    "stratified_gate_weight": "kd_lambda",
    "model_capacity": "width",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def method_bars(summary: pd.DataFrame, out: Path) -> None:
    main = summary[(summary.suite == "main") & (summary.rotation == "original")]
    for order, group in main.groupby("first_anomaly"):
        group = group.sort_values("macro_f1_s2_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(group.method, group.macro_f1_s2_mean, yerr=group.macro_f1_s2_sd, capsize=2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recording-level stage-2 macro-F1")
        ax.set_title(f"{order} first")
        ax.tick_params(axis="x", labelrotation=35)
        _save(fig, out / f"methods_{order.lower()}_first.png")


def ablation_lines(summary: pd.DataFrame, out: Path) -> None:
    for suite, key in ABLATION_KEYS.items():
        group = summary[(summary.suite == suite) & (summary.rotation == "original")]
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for order, sub in group.groupby("first_anomaly"):
            sub = sub.sort_values(key)
            ax.plot(sub[key].astype(str), sub.macro_f1_s2_mean, marker="o", label=f"{order} first")
        ax.set_xlabel(key)
        ax.set_ylabel("Stage-2 macro-F1")
        ax.set_ylim(0, 1)
        ax.legend()
        _save(fig, out / f"ablation_{suite}.png")


def memory_performance(scores: pd.DataFrame, out: Path) -> None:
    subset = scores[(scores.suite == "main") & (scores.rotation == "original")
                    & scores.method.isin(["random", "stratified", "diverse", "gated_kd"])]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, g in subset.groupby("method"):
        ax.scatter(g.buffer_feature_bytes, g.macro_f1_s2, label=method, alpha=0.6)
    ax.set_xlabel("Replay feature bytes")
    ax.set_ylabel("Stage-2 macro-F1")
    ax.legend()
    _save(fig, out / "memory_performance.png")


def stability_plasticity(scores: pd.DataFrame, out: Path) -> None:
    main = scores[(scores.suite == "main") & (scores.rotation == "original")]
    for order, frame in main.groupby("first_anomaly"):
        means = frame.groupby("method")[["old_forgetting_pp", "new_recall"]].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, row in means.iterrows():
            ax.scatter(row.old_forgetting_pp, 100 * row.new_recall)
            ax.annotate(method, (row.old_forgetting_pp, 100 * row.new_recall), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
        ax.set_xlabel("Old-class forgetting (pp)")
        ax.set_ylabel("New-class recall (%)")
        ax.set_ylim(-3, 103)
        _save(fig, out / f"stability_plasticity_{order.lower()}.png")

        recall_cols = [c for c in frame.columns if c.endswith("_recall_s2")]
        ax = frame.groupby("method")[recall_cols].mean().plot.bar(figsize=(10, 5), rot=35)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Stage-2 recall")
        ax.legend(loc="lower right", fontsize=8)
        _save(ax.figure, out / f"per_class_recall_{order.lower()}.png")


def robustness(summary: pd.DataFrame, out: Path) -> None:
    group = summary[summary.suite.isin(["main", "representation"])]
    if group.empty:
        return
    table = group.groupby(["method", "representation", "rotation"]).macro_f1_s2_mean.mean().unstack("rotation")
    ax = table.plot.bar(figsize=(10, 4.5), rot=35)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Stage-2 macro-F1 (mean of orders)")
    ax.legend(fontsize=7, ncol=3)
    _save(ax.figure, out / "robustness.png")


def training_curves(losses: pd.DataFrame, out: Path) -> None:
    if losses.empty:
        return
    base = losses[(losses.suite == "main") & (losses.seed_id == 0) & (losses.fold == 0) & (losses.split_repeat == 0)]
    for order, frame in base.groupby("first_anomaly"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, g in frame[frame.stage == 2].groupby("method"):
            ax.plot(g.step, g.total_loss, label=method, alpha=0.8)
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Stage-2 total loss")
        _save(fig, out / f"training_convergence_{order.lower()}.png")


def gate_distribution(gates: pd.DataFrame, out: Path) -> None:
    gates = gates[gates.method.isin(["stratified_gated_kd", "stratified_fixed_kd", "gated_kd"])] if not gates.empty else gates
    if gates.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, g in gates.groupby("method"):
        ax.hist(g.gate_weight, bins=30, alpha=0.5, label=method, density=True)
    ax.set_xlabel("Distillation gate weight")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, out / "gate_weight_distribution.png")


def make_figures(scores: pd.DataFrame, summary: pd.DataFrame, losses: pd.DataFrame,
                 gates: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    method_bars(summary, out)
    ablation_lines(summary, out)
    memory_performance(scores, out)
    stability_plasticity(scores, out)
    robustness(summary, out)
    training_curves(losses, out)
    gate_distribution(gates, out)
