from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadcl.config import CLASSES, Config
from roadcl.metrics.scores import GROUP_KEYS
from roadcl.utils.runtime import file_sha256

RESOURCE_FIELDS = ["buffer_feature_bytes", "buffer_metadata_bytes", "model_weight_bytes", "scaler_bytes",
                   "recording_latency_ms", "incremental_update_s", "process_lifetime_peak_rss_bytes",
                   "sampled_trial_peak_rss_bytes", "sampled_trial_rss_increase_bytes"]
RESOURCE_KEYS = ["suite", "method", "first_anomaly", "memory_per_class", "representation", "objective",
                 "width", "bag_size", "kd_lambda"]


def _replace_section(path: Path, marker: str, body: str) -> None:
    path.write_text(path.read_text().split(marker)[0] + marker + body)


def dataset_tables(out: Path, data_dir: Path) -> None:
    """Acquisition sidecars and window support appended to the dataset audit."""
    audit = out / "dataset_audit.md"
    sidecars = []
    for row in pd.read_csv(out / "recording_audit.csv").itertuples():
        path = data_dir / row.record.split("/", 1)[1] / "Metadata.csv"
        if path.is_file():
            meta = pd.read_csv(path).iloc[0]
            sidecars.append(dict(record=row.record, label=row.label, recording_time=meta["recording time"],
                                 device_model=meta["device name"], sha256=file_sha256(path)))
    if sidecars:
        sides = pd.DataFrame(sidecars)
        sides.to_csv(out / "recording_metadata.csv", index=False)
        days = sides.assign(date=sides.recording_time.str.slice(0, 10)).groupby(["label", "date"]).size()
        _replace_section(audit, "\n## Acquisition dates\n",
                         "\nAvailable acquisition sidecars by label and local date\n\n" + days.to_string() +
                         "\n\nCollection date and anomaly label can be confounded. A recording-session split cannot "
                         "establish generalization across days, trips, devices or drivers.\n")
    counts = pd.read_csv(out / "window_index.csv").groupby(["label", "record"]).size()
    distribution = counts.groupby(level="label").agg(["count", "sum", "min", "median", "max"])
    distribution.columns = ["usable_recordings", "windows", "min_windows", "median_windows", "max_windows"]
    distribution.to_csv(out / "window_counts.csv")
    _replace_section(audit, "\n## Window support",
                     "\n\n```text\n" + distribution.to_string() + "\n```\n\nClass labels apply to recordings. A "
                     "recording can contain windows without a labeled impact. Overlapping windows contribute "
                     "features to a recording prediction and share its evaluation unit.\n")


def resource_and_descriptor_tables(out: Path, summary: pd.DataFrame) -> None:
    final = summary[summary.metric == "macro_f1_s2"]
    final[final.suite.eq("descriptor") | (final.suite.eq("main") & final.method.eq("stratified"))].to_csv(
        out / "impact_descriptors.csv", index=False)
    resources = pd.read_csv(out / "resource_metrics.csv")
    resources.groupby(RESOURCE_KEYS)[RESOURCE_FIELDS].median().reset_index().to_csv(
        out / "resource_summary.csv", index=False)


def write_diagnostics(out: Path) -> None:
    """Replay composition, gate summary and per-recording sensitivity tables."""
    keys = GROUP_KEYS[3:-1]
    replay_path = out / "replay_selection.csv"
    if replay_path.is_file():
        pd.read_csv(replay_path).groupby(keys + ["split_repeat", "fold", "seed_id", "class_name"], as_index=False).agg(
            windows=("window_index", "size"), recordings=("record", "nunique"), minimum_peak=("peak_accel", "min"),
            median_peak=("peak_accel", "median"), maximum_peak=("peak_accel", "max")
        ).to_csv(out / "replay_composition.csv", index=False)
    gate_path = out / "gate_diagnostics.csv"
    if gate_path.is_file():
        pd.read_csv(gate_path).groupby(keys + ["split_repeat", "fold", "seed_id"], as_index=False).agg(
            mean_gate=("gate_weight", "mean"), mean_teacher_confidence=("teacher_confidence", "mean"),
            mean_kl=("kd_per_bag", "mean")).to_csv(out / "gate_summary.csv", index=False)
    fields = ["first_anomaly", "suite", "method", "representation", "objective", "memory_per_class", "kd_lambda",
              "width", "bag_size", "stage", "rotation", "source_group", "record", "truth", "prediction", "confidence"]
    pred = pd.read_csv(out / "predictions.csv", usecols=fields)
    pred["correct"] = pred.truth.eq(pred.prediction)
    pred.groupby(fields[:-2], as_index=False).agg(fraction_correct=("correct", "mean"),
                                                  mean_confidence=("confidence", "mean")).to_csv(
        out / "per_recording_sensitivity.csv", index=False)


def research_report(out: Path, cfg: Config, summary: pd.DataFrame) -> None:
    """Markdown results with measurement scope and interpretation limits."""
    pair = pd.read_csv(out / "paired_method_comparisons.csv")
    resources = pd.read_csv(out / "resource_metrics.csv")
    final = summary[(summary.metric == "macro_f1_s2") & (summary.rotation == "original")]
    lines = ["# Recording-level results", "",
             f"{len(resources)} outer fits, {cfg.protocol.folds} folds, {cfg.protocol.seeds} training seeds and both arrival orders.",
             "Each estimate averages seed-specific pooled out-of-fold recording metrics. The 95% percentile intervals "
             "resample recording sessions within observed label composition using shared draws across methods. They "
             "condition on the fitted models and class mix. Multiple comparisons are exploratory.", "",
             "| Training configuration | Bump first, macro-F1 [95% CI] | Pothole first, macro-F1 [95% CI] |",
             "|---|---:|---:|"]
    for (suite, method), frame in final[final.suite.isin(["main", "component_ablation", "descriptor", "augmented"])].groupby(["suite", "method"]):
        cells = [f"{suite} / {method}"]
        for order in CLASSES[1:]:
            row = frame[frame.first_anomaly == order].iloc[0]
            cells.append(f"{row.estimate:.3f} [{row.lower:.3f}, {row.upper:.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Paired effects", "",
              "Differences below compare recording-aware impact replay with gated distillation against the "
              "class-quota and uniform random replay controls on clean recordings. Positive values favor gated distillation."]
    for row in pair[(pair.rotation == "original") & (pair.suite == "main") & (pair.method == "stratified_gated_kd")
                    & pair.reference.isin(["main/random", "main/global_random"])].itertuples():
        lines.append(f"- {row.first_anomaly} first versus {row.reference.split('/')[-1]}, "
                     f"{row.macro_f1_difference:+.3f} [{row.lower:+.3f}, {row.upper:+.3f}].")
    lines += ["", "## Resource measurements", "",
              "The axes MLP and scaler occupy %d and %d bytes respectively." % (resources.model_weight_bytes.min(), resources.scaler_bytes.min()),
              "Buffer feature bytes count retained float32 vectors. Metadata bytes count int64 window identifiers. "
              "These payloads exclude runtime containers. The reference benchmark caches data for all folds.",
              "Host latency includes window features, scaling, classification, recording pooling and decision on "
              "preframed windows. The optional checkpoint benchmark also frames the raw sensor arrays. CSV I/O is outside both timings.",
              "The trial sampler observes RSS every 10 ms. The separate OS high-water mark spans the whole process. "
              "Timing and RAM depend on host load, cache reuse and the recording length.",
              "", "## Interpretation limits", "",
              "The input tables identify recordings by source filename. A filename group is a proxy for an acquisition "
              "session unless a verified session manifest is provided. The acquisition sidecars associate each anomaly "
              "class with one collection date. Inference about independent trips, dates, devices or drivers requires "
              "additional provenance and replication.",
              "The ER-ACE incoming loss has a single active class in stage 2 and contributes zero supervised gradient. "
              "Offline joint training has full historical access. Herding and Weight Aligning use the shared "
              "recording-level MLP head. The notebook details their optimization and memory budgets."]
    (out / "results_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
