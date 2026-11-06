from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from grfd.data import FoldPreprocessor
from grfd.models import load_checkpoint, predict_proba

log = logging.getLogger(__name__)


def checkpoint_path(run_dir: str | Path, fold: int, variant: str, task: int) -> Path:
    fold_dir = Path(run_dir) / f"fold_{fold:02d}"
    if variant == "joint":
        return fold_dir / "joint_student.pt"
    return fold_dir / variant / f"student_task_{task}.pt"


def predict_frame(
    frame: pd.DataFrame,
    run_dir: str | Path,
    fold: int = 1,
    variant: str = "replay_lwf",
    task: int = 5,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Score a raw sensor table with a saved fold preprocessor and student checkpoint."""
    prep = FoldPreprocessor.load(Path(run_dir) / f"fold_{fold:02d}" / "preprocessor.joblib")
    frame = frame.rename(columns=lambda c: str(c).strip())
    missing = [c for c in prep.features if c not in frame.columns]
    if missing:
        raise ValueError(f"input is missing sensor columns {missing}; refusing to impute whole modalities")

    model, _ = load_checkpoint(checkpoint_path(run_dir, fold, variant, task), device)
    probs = predict_proba(model, prep.transform(frame))
    classes = np.asarray(prep.classes)

    out = pd.DataFrame({"predicted_label": classes[probs.argmax(1)] if len(probs) else []})
    for i, name in enumerate(classes):
        out[f"prob_{name}"] = probs[:, i]
    return out


def report_against_labels(predicted: pd.Series, actual: pd.Series, classes: list[str]) -> None:
    actual = actual.astype(str).str.strip().str.lower().to_numpy()
    known = np.isin(actual, classes)
    if not known.all():
        log.warning("unrecognised labels ignored: %s", sorted(set(actual[~known])))
    if not known.any():
        return
    y_true, y_pred = actual[known], predicted.to_numpy()[known]
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print(f"accuracy  {accuracy_score(y_true, y_pred):.4f}")
    print(f"macro-F1  {f1_score(y_true, y_pred, labels=classes, average='macro', zero_division=0):.4f}")
