from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from roadcl.config import DataConfig
from roadcl.data.features import window_features
from roadcl.data.windows import slice_windows
from roadcl.models.mlp import FeatureMLP, bag_pool


def load_checkpoint(path: str | Path) -> tuple[FeatureMLP, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"]
    net = FeatureMLP(state["net.0.weight"].shape[1], ckpt["width"])
    net.load_state_dict(state)
    net.eval()
    ckpt["mean"], ckpt["std"] = (np.asarray(ckpt[k], np.float32) for k in ("mean", "std"))
    return net, ckpt


def predict_frame(frame: pd.DataFrame, checkpoint: str | Path, group_col: str = "source_file") -> pd.DataFrame:
    """Predict each recording by featurizing its windows, scaling them with the training statistics and pooling."""
    net, ckpt = load_checkpoint(checkpoint)
    missing = [c for c in ckpt["axes"] if c not in frame.columns]
    if missing:
        raise ValueError(f"input is missing sensor columns {missing}")
    frame = frame.copy()
    if "segment_id" not in frame.columns:
        frame["segment_id"] = 0
    if group_col not in frame.columns:
        frame[group_col] = "recording"
    cfg = DataConfig(fs=ckpt.get("fs", DataConfig.fs), time_tol=ckpt.get("time_tol", DataConfig.time_tol),
                     window=ckpt["window"], stride=ckpt["stride"])
    classes = ckpt["classes"]

    rows = []
    for record, recording in frame.groupby(group_col, sort=True):
        chunks = []
        for _, segment in recording.groupby("segment_id", sort=False):
            windows, _ = slice_windows(segment[ckpt["axes"]].to_numpy(np.float32),
                                       segment.seconds_elapsed.to_numpy(np.float64), cfg)
            if len(windows):
                chunks.append(windows)
        if not chunks:
            rows.append({group_col: record, "n_windows": 0, "prediction": None})
            continue
        z = (window_features(np.concatenate(chunks), ckpt["representation"]) - ckpt["mean"]) / ckpt["std"]
        with torch.inference_mode():
            probs = F.softmax(bag_pool(net(torch.from_numpy(z.astype(np.float32))), dim=0), dim=-1).numpy()
        rows.append({group_col: record, "n_windows": len(z), "prediction": classes[int(np.argmax(probs))],
                     **{f"prob_{c.lower()}": float(p) for c, p in zip(classes, probs)}})
    return pd.DataFrame(rows)
