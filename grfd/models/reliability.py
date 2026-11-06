from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from grfd.config import ReliabilityConfig


def confidence_features(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-8, 1.0)
    ranked = np.sort(clipped, axis=1)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return np.column_stack((ranked[:, -1], ranked[:, -1] - ranked[:, -2], entropy)).astype(np.float64)


def oob_reliability(
    teacher: RandomForestClassifier,
    y_train: np.ndarray,
    teacher_probs: np.ndarray,
    n_classes: int,
    cfg: ReliabilityConfig,
    seed: int,
) -> np.ndarray:
    """Per-row KD weights from a logistic model of OOB teacher correctness, fitted on training rows only."""
    weights = np.ones(len(y_train), dtype=np.float32)

    oob = np.nan_to_num(np.asarray(teacher.oob_decision_function_, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    valid = oob.sum(axis=1) > 1e-12
    if valid.sum() >= cfg.min_oob_rows:
        oob = oob / np.maximum(oob.sum(axis=1, keepdims=True), 1e-12)
        aligned = np.zeros((len(y_train), n_classes), dtype=np.float64)
        aligned[:, teacher.classes_.astype(int)] = oob

        correct = (aligned[valid].argmax(axis=1) == y_train[valid]).astype(int)
        if len(np.unique(correct)) == 2:
            calibrator = LogisticRegression(max_iter=cfg.max_iter, random_state=seed)
            calibrator.fit(confidence_features(aligned[valid]), correct)
            weights = calibrator.predict_proba(confidence_features(teacher_probs))[:, 1].astype(np.float32)
        else:
            weights = np.full(len(y_train), float(correct.mean()), dtype=np.float32)

    return np.clip(weights, cfg.min_weight, cfg.max_weight)
