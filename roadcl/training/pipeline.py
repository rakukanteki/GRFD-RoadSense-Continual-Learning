from __future__ import annotations

import copy
import json
import logging
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import f1_score

from roadcl.config import CLASSES, Config, save_config
from roadcl.data import fold_assignments, load_recordings, make_feature_views, make_windows
from roadcl.data.features import ROTATIONS
from roadcl.data.splits import group_splits
from roadcl.metrics.scores import GROUP_KEYS
from roadcl.training.experiments import Experiment, build_suite
from roadcl.training.trial import Trial, TrialSpec
from roadcl.utils.runtime import file_sha256, sample_trial_ram, seed_everything, setup_logging

log = logging.getLogger(__name__)

OUTPUT_TABLES = {
    "losses": "training_curves.csv",
    "gates": "gate_diagnostics.csv",
    "replay_selection": "replay_selection.csv",
    "replay_diagnostics": "replay_diagnostics.csv",
}


def write_environment(out: Path, cfg: Config) -> None:
    info = dict(system=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__, threads=cfg.protocol.threads,
                augmentation_seed=cfg.data.perturbation_seed, bootstrap_seed=cfg.protocol.seed + 951)
    (out / "environment.json").write_text(json.dumps(info, indent=2))


def _write_dataset_audit(run_dir: Path, cfg: Config, metadata: pd.DataFrame, n_windows: int) -> None:
    audit = pd.read_csv(run_dir / "input_audit.csv")
    counts = metadata.groupby("label").agg(recordings=("record", "size"), groups=("source_group", "nunique"))
    manifest = cfg.data.session_manifest
    provenance = (f"Manifest {manifest} SHA256 {file_sha256(manifest)}" if manifest else
                  "Canonical source_file is the session proxy. Independent trip, driver and vehicle identities are unverified.")
    (run_dir / "dataset_audit.md").write_text(
        "# Dataset audit\n\n" + provenance + "\n\n```text\n" + audit.to_string(index=False) +
        "\n```\n\nUsable recordings\n\n```text\n" + counts.to_string() + "\n```" +
        f"\n\n{n_windows} finite windows; {cfg.data.window} samples, stride {cfg.data.stride}, {cfg.data.fs} Hz. "
        "Windows stay inside contiguous runs of one segment. Exact duplicate traces are removed within class.\n")


def _with_lr(cfg: Config, lr: float) -> Config:
    trial_cfg = copy.copy(cfg)
    trial_cfg.training = replace(cfg.training, lr=lr)
    return trial_cfg


def select_lr(exp: Experiment, repeat: int, fold: int, first: int, development: pd.DataFrame, inner: list,
              windows, views: dict, cfg: Config) -> tuple[float, list]:
    """Learning rate with the best mean inner-fold stage-2 macro-F1; ties keep the earlier grid value."""
    candidates = []
    for lr in cfg.protocol.lr_grid:
        scores = []
        for train_ix, val_ix in inner:
            spec = TrialSpec(repeat, fold, 0, first, sorted(development.iloc[train_ix].record),
                             sorted(development.iloc[val_ix].record))
            rows = Trial(exp, spec, windows, views, _with_lr(cfg, lr), validation=True).run()
            scores.append(f1_score([r["truth"] for r in rows], [r["prediction"] for r in rows],
                                   labels=CLASSES, average="macro", zero_division=0))
        candidates.append((lr, scores))
    return max(candidates, key=lambda pair: np.mean(pair[1]))[0], candidates


def run_benchmark(cfg: Config) -> Path:
    """Run every experiment over outer folds, both arrival orders and seeds, with nested LR selection."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "RUN_COMPLETE.txt").unlink(missing_ok=True)
    setup_logging(run_dir / "train.log")
    save_config(cfg, run_dir / "config.yaml")
    torch.set_num_threads(cfg.protocol.threads)
    write_environment(run_dir, cfg)
    seed_everything(cfg.protocol.seed)

    data, metadata = load_recordings(cfg.data, run_dir)
    windows = make_windows(data, metadata, cfg.data, run_dir)
    del data
    log.info("%d windows from %d recordings", len(windows.raw), len(windows.metadata))
    split = fold_assignments(windows.metadata, cfg.protocol, run_dir)
    suite = build_suite(cfg.protocol.profile)
    (run_dir / "experiments.json").write_text(json.dumps([asdict(e) for e in suite], indent=2))
    views = make_feature_views(windows.raw, sorted({e.representation for e in suite}), cfg.data)
    _write_dataset_audit(run_dir, cfg, windows.metadata, len(windows.raw))

    collected: dict[str, list] = {key: [] for key in OUTPUT_TABLES}
    predictions, resources, tuning, inner_rows = [], [], [], []
    total = cfg.protocol.split_repeats * cfg.protocol.folds * cfg.protocol.seeds * 2 * len(suite)
    for (repeat, fold), held in split.groupby(["split_repeat", "fold"]):
        test = sorted(held.record)
        development = windows.metadata[~windows.metadata.record.isin(test)].reset_index(drop=True)
        train = sorted(development.record)
        inner, inner_seed = group_splits(development, cfg.protocol.inner_folds, cfg.protocol.seed + fold + 1000 * repeat)
        for j, (_, val_ix) in enumerate(inner):
            inner_rows.extend(dict(split_repeat=repeat, fold=fold, inner_fold=j, split_seed=inner_seed, **row)
                              for row in development.iloc[val_ix].to_dict("records"))
        for first in (1, 2):
            cache = {}
            for exp in suite:
                # Suites that differ only in name share one selected learning rate; augmentation does not.
                key = replace(exp, suite="augmented" if exp.suite == "augmented" else "clean")
                if key not in cache:
                    cache[key] = select_lr(exp, repeat, fold, first, development, inner, windows, views, cfg)
                lr, candidates = cache[key]
                fit_seed = cfg.protocol.seed + 100003 * repeat + 10007 * fold + 13 * first
                tuning.extend(dict(**asdict(exp), first_anomaly=CLASSES[first], split_repeat=repeat, fold=fold,
                                   inner_fold=j, fit_seed=fit_seed, lr=value_lr, macro_f1=value,
                                   selected=value_lr == lr)
                              for value_lr, values in candidates for j, value in enumerate(values))
                trial_cfg = _with_lr(cfg, lr)
                for seed_id in range(cfg.protocol.seeds):
                    spec = TrialSpec(repeat, fold, seed_id, first, train, test)
                    with sample_trial_ram() as usage:
                        out = Trial(exp, spec, windows, views, trial_cfg, raw_windows=windows.raw).run()
                    out.resources.update(usage)
                    predictions.extend(out.predictions)
                    resources.append(out.resources)
                    for name in OUTPUT_TABLES:
                        collected[name].extend(getattr(out, name))
                    if out.checkpoint:
                        torch.save(out.checkpoint, run_dir / "checkpoints" / f"{exp.method}_{CLASSES[first]}_fold0.pt")
            log.info("fits %d/%d (repeat %d, fold %d, %s first)", len(resources), total, repeat, fold, CLASSES[first])

    pd.DataFrame(inner_rows).to_csv(run_dir / "inner_assignments.csv", index=False)
    pd.DataFrame(tuning).to_csv(run_dir / "inner_selection.csv", index=False)
    pd.DataFrame(predictions).merge(windows.metadata[["record", "source_group"]], on="record",
                                    validate="many_to_one").to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(resources).to_csv(run_dir / "resource_metrics.csv", index=False)
    for name, filename in OUTPUT_TABLES.items():
        path = run_dir / filename
        path.unlink(missing_ok=True)
        if collected[name]:
            pd.DataFrame(collected[name]).to_csv(path, index=False)
    verify_outputs(run_dir, cfg)
    (run_dir / "RUN_COMPLETE.txt").write_text("Completed the configured experiment suite.\n")
    return run_dir


def verify_outputs(out: Path, cfg: Config) -> dict:
    """Check complete fits, grouped nested validation, shared held-out sets and replay membership."""
    p = cfg.protocol
    outer = pd.read_csv(out / "fold_assignments.csv")
    inner = pd.read_csv(out / "inner_assignments.csv")
    resources = pd.read_csv(out / "resource_metrics.csv")
    expected_runs = len(build_suite(p.profile)) * p.split_repeats * p.folds * p.seeds * 2
    if len(resources) != expected_runs:
        raise AssertionError("incomplete outer fit set")
    for (repeat, fold), g in inner.groupby(["split_repeat", "fold"]):
        assignment = outer[outer.split_repeat == repeat]
        held = assignment[assignment.fold == fold]
        if set(g.record) != set(assignment.record) - set(held.record) or g.record.duplicated().any():
            raise AssertionError("inner validation coverage mismatch")
        if set(g.source_group) & set(held.source_group) or g.groupby("source_group").inner_fold.nunique().max() != 1:
            raise AssertionError("session overlap in nested validation")
    keys = GROUP_KEYS + ["stage"]
    pred = pd.read_csv(out / "predictions.csv", usecols=keys + ["record", "truth"])
    held_records = outer[["split_repeat", "fold", "record"]]
    matched = pred.merge(held_records, on=["split_repeat", "fold", "record"], how="left", indicator=True,
                         validate="many_to_one")
    if not matched._merge.eq("both").all() or pred.duplicated(keys + ["record"]).any():
        raise AssertionError("unexpected or duplicated held-out prediction")
    group_sizes = pred.groupby(keys).size().reset_index(name="n")
    if len(group_sizes) != expected_runs * 2 * (len(ROTATIONS) + 2):
        raise AssertionError("missing stage or stress condition")
    for (repeat, fold, first, stage), g in group_sizes.groupby(["split_repeat", "fold", "first_anomaly", "stage"]):
        held = outer[(outer.split_repeat == repeat) & (outer.fold == fold)]
        count = len(held) if stage == 2 else held.label.isin(["Normal", first]).sum()
        if not g.n.eq(count).all():
            raise AssertionError("methods evaluated on different recording sets")
    if (out / "replay_selection.csv").is_file():
        replay = pd.read_csv(out / "replay_selection.csv")
        exposed = replay.merge(held_records, on=["split_repeat", "fold", "record"], how="inner")
        if len(exposed) or not (replay.class_name.eq("Normal") | replay.class_name.eq(replay.first_anomaly)).all():
            raise AssertionError("held-out or incoming class in old replay")
        counts = replay.groupby(GROUP_KEYS[:-1]).window_index.agg(["size", "nunique"]).reset_index()
        if (counts["size"] != counts["nunique"]).any() or (counts["size"] > 2 * counts.memory_per_class).any():
            raise AssertionError("replay uniqueness or capacity violated")
    result = dict(outer_fits=len(resources), recording_predictions=len(pred), outer_folds=p.folds,
                  inner_folds=p.inner_folds, source_groups=int(outer.source_group.nunique()),
                  shared_held_out_sets=True, grouped_inner_validation=True, replay_membership_checked=True)
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    return result
