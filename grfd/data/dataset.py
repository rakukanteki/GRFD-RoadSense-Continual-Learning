from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from grfd.config import DataConfig

log = logging.getLogger(__name__)


@dataclass
class SensorTable:
    features: pd.DataFrame
    labels: np.ndarray
    class_names: list[str]
    group_col: str | None

    @property
    def feature_cols(self) -> list[str]:
        return list(self.features.columns)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load_sensor_table(cfg: DataConfig, seed: int = 42) -> SensorTable:
    """Read the labeled CSV, pick numeric sensor columns and drop exact duplicates."""
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    label = cfg.label_col
    if label not in df.columns:
        raise ValueError(f"missing label column {label!r}")
    if cfg.max_rows and len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=seed).sort_index()

    df[label] = df[label].astype(str).str.strip().str.lower()
    df = df.loc[df[label] != "nan"].copy()

    group_col = next((c for c in cfg.group_candidates if c in df.columns), None)
    excluded = set(cfg.drop_cols) | set(cfg.group_candidates) | {label}
    feature_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("no numeric feature columns left after dropping")

    conflicts = int((df.groupby(feature_cols, dropna=False, sort=False)[label].nunique() > 1).sum())
    log.info("rows=%d features=%d label conflicts=%d", len(df), len(feature_cols), conflicts)
    if conflicts:
        raise ValueError("identical feature rows carry different labels; resolve them before training")

    before = len(df)
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    log.info("dropped %d duplicate rows, %d remain", before - len(df), len(df))

    encoder = LabelEncoder().fit(df[label])
    features = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    log.info(
        "classes=%s counts=%s missing values=%d",
        list(encoder.classes_),
        dict(Counter(df[label])),
        int(features.isna().sum().sum()),
    )
    log.warning("random-row CV is not group-disjoint; rows from one trip can land in train and test")

    return SensorTable(
        features=features,
        labels=encoder.transform(df[label]).astype(np.int64),
        class_names=list(encoder.classes_),
        group_col=group_col,
    )
