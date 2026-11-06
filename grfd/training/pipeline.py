from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from grfd.config import Config, save_config
from grfd.data import FoldPreprocessor, load_sensor_table, stratified_buckets, stratified_folds
from grfd.utils import resolve_device, seed_everything, setup_logging

from .baselines import run_baselines
from .context import FoldContext, Recorder
from .continual import run_variant
from .joint import run_joint

log = logging.getLogger(__name__)


def run_cross_validation(cfg: Config) -> Path:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    save_config(cfg, run_dir / "config.yaml")

    exp = cfg.experiment
    seed_everything(exp.seed)
    device = resolve_device(exp.device)
    log.info("run_dir=%s device=%s folds=%d experiences=%d", run_dir, device, exp.n_folds, cfg.continual.n_tasks)

    table = load_sensor_table(cfg.data, exp.seed)
    y = table.labels
    rec = Recorder(run_dir)

    for split in stratified_folds(y, exp.n_folds, exp.val_size, exp.seed):
        fold = split.fold
        fold_dir = run_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True)

        prep = FoldPreprocessor.fit(table.features, split.train, table.class_names)
        prep.save(
            fold_dir / "preprocessor.joblib",
            dropped_columns=cfg.data.drop_cols,
            label_column=cfg.data.label_col,
            fold=fold,
        )
        X = prep.transform(table.features)
        np.savez_compressed(fold_dir / "splits.npz", train=split.train, val=split.val, test=split.test)

        tasks = stratified_buckets(split.train, y, cfg.continual.n_tasks, exp.seed + fold)
        probes = stratified_buckets(split.test, y, cfg.continual.n_tasks, exp.seed + 1000 + fold)
        for k, part in enumerate(tasks, start=1):
            np.save(fold_dir / f"task_{k}_train_indices.npy", part)

        ctx = FoldContext(
            split=split,
            X=X,
            y=y,
            class_names=table.class_names,
            feature_cols=table.feature_cols,
            tasks=tasks,
            probes=probes,
            fold_dir=fold_dir,
            is_last=fold == exp.n_folds,
        )
        log.info(
            "FOLD %d/%d train=%s val=%s test=%s",
            fold, exp.n_folds, f"{len(split.train):,}", f"{len(split.val):,}", f"{len(split.test):,}",
        )

        if exp.run_joint:
            run_joint(ctx, cfg, rec, device)
        if exp.run_baselines:
            run_baselines(ctx, cfg, rec)
        if exp.run_continual:
            for name in cfg.continual.run:
                run_variant(ctx, name, cfg.continual.variants[name], cfg, rec, device)

        rec.flush()
        log.info("fold %d finished, tables written to %s", fold, run_dir)

    return run_dir
