from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


def macro_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    try:
        return float(roc_auc_score(y_true, probs, labels=np.arange(n_classes), multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def label_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels = np.arange(len(class_names))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, name in enumerate(class_names):
        out[f"precision_{name}"] = float(p[i])
        out[f"recall_{name}"] = float(r[i])
        out[f"f1_{name}"] = float(f[i])
        out[f"support_{name}"] = int(s[i])
    return out


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    out = label_metrics(y_true, probs.argmax(axis=1), class_names)
    out["auc_ovr"] = macro_auc(y_true, probs, len(class_names))
    return out


def probe_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=np.arange(n_classes), average="macro", zero_division=0)),
    }
