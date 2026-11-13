from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from roadcl.config import CLASSES
from roadcl.metrics import aggregate_scores, paired_differences, summarize

log = logging.getLogger(__name__)


def gate_summary(gates: pd.DataFrame) -> pd.DataFrame:
    if gates.empty:
        return pd.DataFrame()
    return gates.groupby(["suite", "method", "first_anomaly"], as_index=False).agg(
        gate_mean=("gate_weight", "mean"), gate_sd=("gate_weight", "std"),
        teacher_confidence_mean=("teacher_confidence", "mean"), kd_per_bag_mean=("kd_per_bag", "mean"))


def replay_composition(replay: pd.DataFrame) -> pd.DataFrame:
    if replay.empty:
        return pd.DataFrame()
    keys = ["suite", "method", "first_anomaly", "split_repeat", "fold", "seed_id", "class_name"]
    return replay.groupby(keys).agg(
        selected_windows=("window_index", "count"), unique_recordings=("record", "nunique"),
        peak_min=("peak_accel", "min"), peak_median=("peak_accel", "median"),
        peak_max=("peak_accel", "max")).reset_index()


def confusion_tables(predictions: pd.DataFrame, out: Path) -> None:
    """Stage-2 confusion counts per configuration, pooled over folds and seeds."""
    out.mkdir(parents=True, exist_ok=True)
    final = predictions[(predictions.stage == 2) & (predictions.rotation == "original")]
    for key, group in final.groupby(["suite", "method", "first_anomaly"]):
        matrix = confusion_matrix(group.truth, group.prediction, labels=CLASSES)
        name = "_".join(key).lower().replace("/", "_")
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{name}.csv")


def headline(summary: pd.DataFrame) -> pd.DataFrame:
    main = summary[(summary.suite == "main") & (summary.rotation == "original")]
    cols = ["first_anomaly", "method", "macro_f1_s2_mean", "old_forgetting_pp_mean", "new_recall_mean",
            "ece_s2_mean", "buffer_feature_bytes_mean"]
    return main[cols].sort_values(["first_anomaly", "macro_f1_s2_mean"], ascending=[True, False])


def build_tables(run_dir: Path, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score saved predictions and write the summary, paired, gate and replay tables."""
    predictions = pd.read_csv(run_dir / "predictions.csv")
    scores = aggregate_scores(predictions)
    scores.to_csv(run_dir / "scores_by_fold_seed.csv", index=False)
    summary = summarize(scores, iterations, seed)
    summary.to_csv(run_dir / "scores_summary.csv", index=False)
    paired_differences(scores, iterations, seed).to_csv(run_dir / "paired_method_comparisons.csv", index=False)
    confusion_tables(predictions, run_dir / "confusion")

    gates_path, replay_path = run_dir / "gate_diagnostics.csv", run_dir / "replay_diagnostics.csv"
    if gates_path.exists() and gates_path.stat().st_size > 1:
        gate_summary(pd.read_csv(gates_path)).to_csv(run_dir / "gate_summary.csv", index=False)
    if replay_path.exists() and replay_path.stat().st_size > 1:
        replay_composition(pd.read_csv(replay_path)).to_csv(run_dir / "replay_composition.csv", index=False)
    if not summary.empty:
        log.info("stage-2 results, original orientation:\n%s", headline(summary).round(3).to_string(index=False))
    return scores, summary
