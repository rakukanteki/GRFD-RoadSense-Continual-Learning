from __future__ import annotations

import json
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from roadcl.config import CLASSES, Config, save_config
from roadcl.data import fold_assignments, load_recordings, make_feature_views, make_windows
from roadcl.training.experiments import build_suite
from roadcl.training.trial import Trial, TrialSpec
from roadcl.utils.runtime import seed_everything, setup_logging

log = logging.getLogger(__name__)

OUTPUT_TABLES = {
    "predictions": "predictions.csv",
    "replay_selection": "replay_selection.csv",
    "losses": "training_curves.csv",
    "gates": "gate_diagnostics.csv",
    "replay_diagnostics": "replay_diagnostics.csv",
}


def _write_environment(run_dir: Path) -> None:
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "threads": torch.get_num_threads()}
    (run_dir / "environment.json").write_text(json.dumps(info, indent=2))


def run_benchmark(cfg: Config) -> Path:
    """Run every experiment of the suite over folds, seeds and both anomaly arrival orders."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    setup_logging(run_dir / "train.log")
    save_config(cfg, run_dir / "config.yaml")
    torch.set_num_threads(cfg.protocol.threads)
    _write_environment(run_dir)
    seed_everything(cfg.protocol.seed)

    data, metadata = load_recordings(cfg.data, run_dir)
    windows = make_windows(data, metadata, cfg.data, run_dir)
    del data
    log.info("%d windows from %d recordings", len(windows.raw), len(windows.metadata))
    split = fold_assignments(windows.metadata, cfg.protocol, run_dir)

    suite = build_suite(cfg.protocol.profile)
    views = make_feature_views(windows.raw, sorted({e.representation for e in suite}), cfg.data)
    windows.raw = None

    collected: dict[str, list] = {key: [] for key in OUTPUT_TABLES}
    resources = []
    total = cfg.protocol.split_repeats * cfg.protocol.folds * cfg.protocol.seeds * 2 * len(suite)
    done = 0
    for repeat in range(cfg.protocol.split_repeats):
        for fold in range(cfg.protocol.folds):
            test = split[(split.split_repeat == repeat) & (split.fold == fold)].record.tolist()
            train = sorted(set(windows.metadata.record) - set(test))
            if not test or not train:
                raise ValueError(f"fold {fold} has an empty partition")
            for seed_id in range(cfg.protocol.seeds):
                for first in (1, 2):
                    spec = TrialSpec(repeat, fold, seed_id, first, train, test)
                    for exp in suite:
                        out = Trial(exp, spec, windows, views, cfg).run()
                        for key in OUTPUT_TABLES:
                            collected[key].extend(getattr(out, key))
                        resources.append(out.resources)
                        if out.checkpoint:
                            name = f"{exp.method}_{CLASSES[first].lower()}_first_fold0.pt"
                            torch.save(out.checkpoint, run_dir / "checkpoints" / name)
                        done += 1
                    log.info("trials %d/%d (repeat %d, fold %d, seed %d, %s first)",
                             done, total, repeat, fold, seed_id, CLASSES[first])

    for key, filename in OUTPUT_TABLES.items():
        pd.DataFrame(collected[key]).to_csv(run_dir / filename, index=False)
    pd.DataFrame(resources).to_csv(run_dir / "resource_metrics.csv", index=False)
    (run_dir / "RUN_COMPLETE.txt").write_text("Completed the configured experiment suite.\n")
    return run_dir
