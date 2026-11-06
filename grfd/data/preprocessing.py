from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPreprocessor:
    medians: pd.Series
    scaler: StandardScaler
    features: list[str]
    classes: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_idx: np.ndarray, classes: list[str]) -> "FoldPreprocessor":
        medians = frame.iloc[train_idx].median().fillna(0)
        filled = frame.fillna(medians).to_numpy(dtype=np.float32)
        scaler = StandardScaler().fit(filled[train_idx])
        return cls(medians=medians, scaler=scaler, features=list(frame.columns), classes=list(classes))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        filled = values.fillna(self.medians).fillna(0).to_numpy(dtype=np.float32)
        return self.scaler.transform(filled).astype(np.float32)

    def save(self, path: str | Path, **extra) -> None:
        payload = {"medians": self.medians, "scaler": self.scaler, "features": self.features, "classes": self.classes}
        joblib.dump({**payload, **extra}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FoldPreprocessor":
        payload = joblib.load(path)
        return cls(
            medians=payload["medians"],
            scaler=payload["scaler"],
            features=list(payload["features"]),
            classes=list(payload["classes"]),
        )
