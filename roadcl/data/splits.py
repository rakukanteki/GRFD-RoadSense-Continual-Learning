from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from roadcl.config import CLASSES, ProtocolConfig


def group_splits(meta: pd.DataFrame, count: int, seed: int) -> tuple[list, int]:
    """First seeded StratifiedGroupKFold whose every test fold holds all three classes, and its seed."""
    if meta.groupby("label").source_group.nunique().min() < count:
        raise ValueError("not enough independent source groups per class for class-complete folds")
    for attempt in range(1000):
        cv = StratifiedGroupKFold(count, shuffle=True, random_state=seed + attempt)
        folds = list(cv.split(meta, meta.label, meta.source_group))
        if all(meta.iloc[test].label.nunique() == len(CLASSES) for _, test in folds):
            return folds, seed + attempt
    raise ValueError("cannot form class-complete grouped folds")


def fold_assignments(metadata: pd.DataFrame, cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    """Recording-disjoint folds, stratified by class and grouped by source recording."""
    rows = []
    for repeat in range(cfg.split_repeats):
        folds, used_seed = group_splits(metadata, cfg.folds, cfg.seed + 777 * repeat)
        for fold, (_, test) in enumerate(folds):
            rows.extend((repeat, fold, metadata.record.iat[i], used_seed) for i in test)
    split = pd.DataFrame(rows, columns=["split_repeat", "fold", "record", "split_seed"])

    for _, grp in split.groupby("split_repeat"):
        if grp.record.nunique() != len(metadata) or grp.groupby("record").size().max() != 1:
            raise AssertionError("every recording must be tested exactly once per repeat")
        merged = grp.merge(metadata[["record", "source_group"]], on="record", validate="many_to_one")
        if merged.groupby("source_group").fold.nunique().max() > 1:
            raise AssertionError("a source group leaked across folds")
    split = split.merge(metadata[["record", "label", "source_group"]], on="record", validate="many_to_one")
    split.groupby(["split_repeat", "fold", "label"]).agg(
        n_recordings=("record", "nunique"), n_sources=("source_group", "nunique")
    ).reset_index().to_csv(out_dir / "fold_class_counts.csv", index=False)
    split.to_csv(out_dir / "fold_assignments.csv", index=False)
    return split
