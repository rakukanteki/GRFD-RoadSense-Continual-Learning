from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ORIGINAL_VARIANTS = ["naive", "replay", "lwf", "replay_lwf"]
CL_COLUMNS = ["AIA_probe", "AF_probe", "BWT_probe", "FWT_probe", "FAA_probe", "final_replay_examples", "final_memory_mb_raw"]


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _mean_std(df: pd.DataFrame, by, cols: list[str], digits: int = 4) -> str:
    cols = [c for c in cols if c in df.columns]
    return df.groupby(by)[cols].agg(["mean", "std"]).round(digits).to_string()


def _section(title: str, body: str) -> None:
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}\n{body}")


def summarize_run(run_dir: str | Path) -> None:
    """Print the paper tables for a finished run and write the C2/C3 subsets as CSV."""
    run_dir = Path(run_dir)
    folds = _read(run_dir, "fold_metrics")
    if folds.empty:
        raise FileNotFoundError(f"no fold_metrics.csv in {run_dir}")

    class_names = [c.removeprefix("recall_") for c in folds.columns if c.startswith("recall_")]
    show = ["accuracy", "macro_f1", "weighted_f1", "auc_ovr"] + [f"recall_{c}" for c in class_names]

    _section("CLASSIFICATION PERFORMANCE ACROSS FOLDS (mean, std)", _mean_std(folds, "model", show))
    _section("PER-FOLD RESULTS", folds[["fold", "model"] + show].round(4).to_string(index=False))

    baselines = _read(run_dir, "baseline_model_comparison")
    if not baselines.empty:
        both = pd.concat([baselines, folds[folds.model == "joint"]], ignore_index=True)
        _section("BASELINE COMPARISON", _mean_std(both, "model", show))

    original = folds[folds.model.isin(ORIGINAL_VARIANTS)]
    if not original.empty:
        _section("CONTINUAL ABLATION, FINAL TEST", _mean_std(original, "model", show))

    summary = _read(run_dir, "continual_summary_per_fold")
    if not summary.empty:
        _section("CONTINUAL METRICS ON RANDOM STRATIFIED PROBES", _mean_std(summary, "model", CL_COLUMNS, 5))
        print("These are random-probe diagnostics, not domain-incremental forgetting or transfer.")

    timeline = _read(run_dir, "continual_timeline")
    if not timeline.empty:
        _section("POOLED TEST METRICS BY EXPERIENCE", _mean_std(timeline, ["model", "after_task"], ["accuracy", "macro_f1", "auc_ovr"]))

    memory = _read(run_dir, "memory_budget")
    if not memory.empty:
        body = memory.groupby(["model", "after_task"])[["replay_examples", "memory_mb_raw"]].mean().round(4).to_string()
        _section("MEMORY BUDGET (RAW REPLAY ARRAYS)", body)

    efficiency = _read(run_dir, "efficiency")
    if not efficiency.empty:
        _section("STUDENT SIZE AND CPU LATENCY", _mean_std(efficiency, "model", ["params", "student_mb", "ms_per_sample_cpu"], 6))

    _summarize_c123(run_dir, folds, timeline, memory, show)


def _summarize_c123(run_dir: Path, folds: pd.DataFrame, timeline: pd.DataFrame, memory: pd.DataFrame, show: list[str]) -> None:
    c1 = _read(run_dir, "C1_compression_per_fold")
    if not c1.empty:
        cols = ["accuracy", "macro_f1", "bytes", "mb", "cpu_ms_per_sample", "train_seconds", "parameters"]
        _section("C1 TEACHER VS STUDENT", _mean_std(c1, "model", cols, 5))
        sizes = c1.pivot_table(index="fold", columns="model", values="bytes")
        if {"Random Forest teacher", "Gated student"} <= set(sizes.columns):
            reduction = 100 * (1 - sizes["Gated student"] / sizes["Random Forest teacher"])
            spread = reduction.std(ddof=1) if len(reduction) > 1 else 0.0
            print(f"Serialized size reduction: {reduction.mean():.2f}% (sd {spread:.2f} pp)")

    extra = [m for m in folds.model.unique() if m not in ORIGINAL_VARIANTS + ["joint"]]
    if not extra:
        return

    final = folds[folds.model.isin(extra)]
    final.to_csv(run_dir / "C2_C3_final_per_fold.csv", index=False)
    _section("C2/C3 VARIANTS, FINAL TEST", _mean_std(final, "model", show))

    if not timeline.empty:
        curve = timeline[timeline.model.isin(extra)]
        curve.to_csv(run_dir / "C2_C3_timeline_per_fold.csv", index=False)

    costs = _read(run_dir, "C3_update_costs")
    if not costs.empty:
        cols = ["rf_fit_seconds", "student_update_seconds", "total_update_seconds", "replay_rows"]
        _section("C3 UPDATE COST PER EXPERIENCE (seconds)", _mean_std(costs, "variant", cols, 2))

    if not memory.empty:
        mem = memory[memory.model.isin(extra)]
        mem.to_csv(run_dir / "C3_memory_budget.csv", index=False)
        last = mem[mem.after_task == mem.after_task.max()]
        _section("C3 FINAL MEMORY BUDGET", _mean_std(last, "model", ["replay_examples", "memory_mb_raw"]))


def last_fold_predictions(run_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    pred_dir = Path(run_dir) / "predictions"
    files = sorted(pred_dir.glob("fold_*_*.npz"))
    if not files:
        return {}
    last_fold = max(f.name.split("_")[1] for f in files)
    out = {}
    for f in files:
        _, fold, model = f.stem.split("_", 2)
        if fold == last_fold:
            with np.load(f) as data:
                out[model] = {k: data[k] for k in data.files}
    return out
