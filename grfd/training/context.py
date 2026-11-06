from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grfd.data import FoldSplit


@dataclass
class FoldContext:
    split: FoldSplit
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    feature_cols: list[str]
    tasks: list[np.ndarray]
    probes: list[np.ndarray]
    fold_dir: Path
    is_last: bool

    @property
    def fold(self) -> int:
        return self.split.fold

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def X_test(self) -> np.ndarray:
        return self.X[self.split.test]

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.split.test]

    def save_predictions(self, model_name: str, probs: np.ndarray) -> None:
        """Keep last-fold test probabilities for confusion matrices and ROC curves."""
        if not self.is_last:
            return
        out = self.fold_dir.parent / "predictions"
        out.mkdir(exist_ok=True)
        np.savez_compressed(
            out / f"fold_{self.fold:02d}_{model_name}.npz",
            y_true=self.y_test,
            probs=probs.astype(np.float32),
            class_names=np.asarray(self.class_names),
        )


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.tables: dict[str, list[dict]] = defaultdict(list)

    def add(self, table: str, row: dict) -> None:
        self.tables[table].append(row)

    def extend(self, table: str, rows: list[dict]) -> None:
        self.tables[table].extend(rows)

    def frame(self, table: str) -> pd.DataFrame:
        return pd.DataFrame(self.tables.get(table, []))

    def flush(self) -> None:
        for name, rows in self.tables.items():
            if rows:
                pd.DataFrame(rows).to_csv(self.out_dir / f"{name}.csv", index=False)
