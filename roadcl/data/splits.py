from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from roadcl.config import CLASSES, ProtocolConfig


def fold_assignments(metadata: pd.DataFrame, cfg: ProtocolConfig, out_dir: Path) -> pd.DataFrame:
    """Recording-disjoint folds, stratified by class and grouped by source recording."""
    if metadata.groupby("label").source_group.nunique().min() < cfg.folds:
        raise ValueError("not enough independent source groups per class for the requested fold count")
    y = metadata.label.map(CLASSES.index).to_numpy()
    groups = metadata.source_group.to_numpy()

    rows = []
    for repeat in range(cfg.split_repeats):
        cv = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed + 777 * repeat)
        for fold, (_, test) in enumerate(cv.split(metadata, y, groups)):
            rows.extend((repeat, fold, metadata.record.iat[i]) for i in test)
    split = pd.DataFrame(rows, columns=["split_repeat", "fold", "record"])

    for _, grp in split.groupby("split_repeat"):
        if grp.record.nunique() != len(metadata) or grp.groupby("record").size().max() != 1:
            raise AssertionError("every recording must be tested exactly once per repeat")
        merged = grp.merge(metadata[["record", "source_group"]], on="record", validate="many_to_one")
        if merged.groupby("source_group").fold.nunique().max() > 1:
            raise AssertionError("a source group leaked across folds")
    split.to_csv(out_dir / "fold_assignments.csv", index=False)
    return split
