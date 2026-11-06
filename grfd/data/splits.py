from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class FoldSplit:
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_folds(y: np.ndarray, n_folds: int, val_size: float, seed: int) -> Iterator[FoldSplit]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (outer, test) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        train, val = train_test_split(outer, test_size=val_size, random_state=seed + fold, stratify=y[outer])
        yield FoldSplit(fold=fold, train=train, val=val, test=test)


def stratified_buckets(indices: np.ndarray, y: np.ndarray, n_buckets: int, seed: int) -> list[np.ndarray]:
    """Split indices into class-stratified random buckets of near-equal size."""
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    for cls in np.unique(y[indices]):
        members = indices[y[indices] == cls].copy()
        rng.shuffle(members)
        for k, part in enumerate(np.array_split(members, n_buckets)):
            buckets[k].extend(part.tolist())

    out = [np.asarray(b, dtype=int) for b in buckets]
    for bucket in out:
        rng.shuffle(bucket)
    assert len(np.unique(np.concatenate(out))) == len(indices)
    return out
