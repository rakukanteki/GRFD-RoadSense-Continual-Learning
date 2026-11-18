from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from roadcl.config import CLASSES

GROUP_KEYS = ["split_repeat", "fold", "seed_id", "first_anomaly", "suite", "method", "representation",
              "objective", "memory_per_class", "kd_lambda", "width", "bag_size", "rotation"]


def recording_scores(truth: list[int], pred: list[int], active: list[int]) -> dict | None:
    """Macro-F1, accuracy and per-class recall over test recordings; None if a class is absent."""
    if not truth:
        return None
    cm = confusion_matrix(truth, pred, labels=active)
    support = cm.sum(axis=1)
    if np.any(support == 0):
        return None
    recall = np.diag(cm) / support
    precision, rec, f1, counts = precision_recall_fscore_support(truth, pred, labels=active, zero_division=0)
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float((np.asarray(truth) == np.asarray(pred)).mean()),
        "per_class": {CLASSES[k]: {"precision": float(precision[j]), "recall": float(rec[j]),
                                   "f1": float(f1[j]), "support": int(counts[j])}
                      for j, k in enumerate(active)},
    }


def calibration(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, float]:
    """Expected calibration error over equal-width bins and the Brier score of the confidence."""
    if not len(confidence):
        return np.nan, np.nan
    ece = 0.0
    for b in range(bins):
        upper = confidence < (b + 1) / bins if b < bins - 1 else confidence <= 1.0
        mask = (confidence >= b / bins) & upper
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece), float(np.mean((confidence - correct.astype(float)) ** 2))


def _score_stage(subset: pd.DataFrame, active: list[int], stage: int, row: dict) -> bool:
    if subset.record.duplicated().any():
        raise AssertionError("a recording was evaluated twice in one stage")
    truth = [CLASSES.index(x) for x in subset.truth]
    pred = [CLASSES.index(x) for x in subset.prediction]
    result = recording_scores(truth, pred, active)
    if result is None:
        return False
    row[f"macro_f1_s{stage}"] = result["macro_f1"]
    row[f"accuracy_s{stage}"] = result["accuracy"]
    row[f"balanced_accuracy_s{stage}"] = result["balanced_accuracy"]
    for name, metric in result["per_class"].items():
        row[f"{name.lower()}_precision_s{stage}"] = metric["precision"]
        row[f"{name.lower()}_recall_s{stage}"] = metric["recall"]
        row[f"{name.lower()}_f1_s{stage}"] = metric["f1"]
    ece, brier = calibration(subset.confidence.to_numpy(float),
                             (subset.truth == subset.prediction).to_numpy())
    row[f"ece_s{stage}"] = ece
    row[f"confidence_brier_s{stage}"] = brier
    proba = subset[[f"prob_{CLASSES[k].lower()}" for k in active]].to_numpy(float)
    indicator = np.eye(len(active))[[active.index(k) for k in truth]]
    row[f"multiclass_brier_s{stage}"] = float(np.mean(np.sum((proba - indicator) ** 2, axis=1)))
    return True


def aggregate_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per trial and test view with stage scores, forgetting and new-class recall."""
    rows = []
    for key, frame in predictions.groupby(GROUP_KEYS, dropna=False):
        row = dict(zip(GROUP_KEYS, key))
        first = CLASSES.index(row["first_anomaly"])
        old_name, new_name = CLASSES[first].lower(), CLASSES[3 - first].lower()
        if not all(_score_stage(frame[frame.stage == stage], active, stage, row)
                   for stage, active in ((1, [0, first]), (2, [0, 1, 2]))):
            continue
        row["old_forgetting_pp"] = 50 * (row["normal_recall_s1"] + row[f"{old_name}_recall_s1"]
                                         - row["normal_recall_s2"] - row[f"{old_name}_recall_s2"])
        row["new_recall"] = row[f"{new_name}_recall_s2"]
        row["normal_forgetting_pp"] = 100 * (row["normal_recall_s1"] - row["normal_recall_s2"])
        row["old_anomaly_forgetting_pp"] = 100 * (row[f"{old_name}_recall_s1"] - row[f"{old_name}_recall_s2"])
        row["backward_transfer"] = -row["old_forgetting_pp"] / 100
        row["buffer_feature_bytes"] = int(frame.buffer_feature_bytes.max())
        row["stage2_train_s"] = float(frame[frame.stage == 2].train_seconds.max())
        rows.append(row)
    return pd.DataFrame(rows)
