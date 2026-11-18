from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAMES = {"weight_align": "Weight Aligning", "er_ace": "ER-ACE", "herding": "Herding", "lwf": "LwF",
         "joint": "Joint", "reservoir": "Balanced reservoir", "fine_tune": "Fine-tuning",
         "random": "Class-quota random", "global_random": "Uniform random", "stratified": "Recording + impact",
         "stratified_gated_kd": "Recording + impact + gated KD", "impact_only": "Impact only",
         "recording_balanced": "Recording only", "diverse": "Diverse", "fixed_kd": "Diverse + fixed KD",
         "gated_kd": "Diverse + gated KD", "stratified_fixed_kd": "Recording + impact + fixed KD"}
CONDITIONS = ["original", "x90", "y90", "z90", "sensor_noise", "gyro_missing"]
CONDITION_LABELS = ["Clean", "90° about x", "90° about y", "90° about z", "Sensor noise", "Gyroscope missing"]
SWEEPS = [("memory_sweep", "memory_per_class"), ("stratified_memory", "memory_per_class"),
          ("model_capacity", "width"), ("bag_size", "bag_size"), ("representation", "representation"),
          ("gate_ablation", "kd_lambda"), ("stratified_gate_weight", "kd_lambda")]


def method_bars(final: pd.DataFrame, out: Path) -> None:
    """Clean-recording macro-F1 with session-bootstrap intervals for the main suite and the replay ablation."""
    for suite, filename in (("main", "baseline"), ("component_ablation", "replay_ablation")):
        sub = final[(final.suite == suite) & (final.rotation == "original")]
        sub.to_csv(out / f"{filename}.csv", index=False)
        for order, df in sub.groupby("first_anomaly"):
            df = df.sort_values("estimate")
            fig, ax = plt.subplots(figsize=(3.5, max(2, len(df) * 0.22)))
            ax.barh(df.method.map(NAMES), df.estimate, color="#447799")
            ax.hlines(df.method.map(NAMES), df.lower, df.upper, color="black", linewidth=1)
            ax.set(xlim=(0, 1), xlabel="Recording-level macro-F1", title=f"{order} first")
            fig.tight_layout()
            fig.savefig(out / f"{filename}_{order}.svg", bbox_inches="tight")
            plt.close(fig)


def optimization_curves(losses: pd.DataFrame, out: Path) -> None:
    """Mean out-of-fold stage-2 supervised loss of the comparable main methods."""
    methods = ["fine_tune", "random", "stratified", "stratified_gated_kd"]
    subset = losses[(losses.suite == "main") & (losses.stage == 2) & losses.method.isin(methods)]
    for order, g in subset.groupby("first_anomaly"):
        fig, ax = plt.subplots(figsize=(3.5, 2.45))
        for (name, series), style in zip(g.groupby("method"), ["-", "--", "-.", ":"]):
            curve = series.groupby("step").task_loss.mean()
            ax.plot(curve.index + 1, curve, style, label=name.replace("_", " "), lw=1.1)
        ax.set(xlabel="Incremental optimizer step", ylabel="Mean supervised loss", title=f"{order} first")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"optimization_{order}.svg")
        plt.close(fig)


def robustness(final: pd.DataFrame, out: Path) -> None:
    """Stress-test intervals for clean and augmented training of the gated-KD replay method."""
    robust = final[(final.method == "stratified_gated_kd") & final.suite.isin(["main", "augmented"])]
    robust.to_csv(out / "robustness.csv", index=False)
    for order, frame in robust.groupby("first_anomaly"):
        fig, ax = plt.subplots(figsize=(3.5, 2.65))
        for suite, offset, color, marker in (("main", -0.12, "#245E91", "o"), ("augmented", 0.12, "#A54C22", "s")):
            g = frame[frame.suite == suite].set_index("rotation").reindex(CONDITIONS)
            y = np.arange(len(CONDITIONS)) + offset
            ax.hlines(y, g.lower, g.upper, color=color, lw=1)
            ax.scatter(g.estimate, y, s=13, color=color, marker=marker,
                       label="Clean training" if suite == "main" else "Augmented training")
        ax.set_yticks(range(len(CONDITIONS)), CONDITION_LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recording-level macro-F1")
        ax.set_title(f"{order} first")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"robustness_{order}.svg", bbox_inches="tight")
        plt.close(fig)


def sweeps(final: pd.DataFrame, out: Path) -> None:
    """One figure per one-dimensional sweep of the full profile."""
    for suite, field in SWEEPS:
        sub = final[(final.suite == suite) & (final.rotation == "original")]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        for order, g in sub.groupby("first_anomaly"):
            g = g.sort_values(field)
            ax.plot(g[field].astype(str), g.estimate, marker="o", label=f"{order} first")
        ax.set(xlabel=field.replace("_", " "), ylabel="Macro-F1", ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{suite}.svg")
        plt.close(fig)


def make_figures(summary: pd.DataFrame, losses: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "svg.fonttype": "none"})
    final = summary[summary.metric == "macro_f1_s2"]
    method_bars(final, out)
    robustness(final, out)
    sweeps(final, out)
    if not losses.empty:
        optimization_curves(losses, out)
