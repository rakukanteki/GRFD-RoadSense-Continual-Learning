from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from grfd.config import TeacherConfig


def build_teacher(cfg: TeacherConfig, seed: int, oob: bool = False) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_features=cfg.max_features,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=seed,
        n_jobs=cfg.n_jobs,
        oob_score=oob,
        bootstrap=True,
    )


def aligned_proba(estimator, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba padded to all classes, for estimators that missed a class during fit."""
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    out[:, estimator.classes_.astype(int)] = estimator.predict_proba(X)
    return out
