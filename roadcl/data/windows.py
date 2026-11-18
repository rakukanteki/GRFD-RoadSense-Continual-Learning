from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from roadcl.config import AXES, CLASSES, DataConfig


@dataclass
class WindowSet:
    raw: np.ndarray
    index: pd.DataFrame
    metadata: pd.DataFrame
    bags: dict[str, np.ndarray]
    label: dict[str, int]


def continuous_runs(seconds: np.ndarray, fs: float, tol: float) -> list[np.ndarray]:
    """Split sample positions wherever the time step departs from 1/fs."""
    breaks = np.flatnonzero(~np.isclose(np.diff(seconds), 1 / fs, rtol=0, atol=tol)) + 1
    return np.split(np.arange(len(seconds)), breaks)


def slice_windows(values: np.ndarray, seconds: np.ndarray, cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return finite windows of one segment and their start positions."""
    chunks, starts_out = [], []
    for run in continuous_runs(seconds, cfg.fs, cfg.time_tol):
        if len(run) < cfg.window:
            continue
        starts = np.arange(0, len(run) - cfg.window + 1, cfg.stride)
        positions = run[starts[:, None] + np.arange(cfg.window)[None, :]]
        windows = values[positions]
        finite = np.isfinite(windows).all(axis=(1, 2))
        if finite.any():
            chunks.append(windows[finite])
            starts_out.append(positions[finite, 0])
    if not chunks:
        return np.empty((0, cfg.window, values.shape[1]), np.float32), np.empty(0, np.int64)
    return np.concatenate(chunks), np.concatenate(starts_out)


def make_windows(data: pd.DataFrame, metadata: pd.DataFrame, cfg: DataConfig, out_dir: Path) -> WindowSet:
    """Cut every recording into windows that never bridge a segment boundary or a time gap."""
    chunks, rows = [], []
    for (record, segment), frame in data.groupby(["record", "segment_id"], sort=False):
        seconds = frame.seconds_elapsed.to_numpy(np.float64)
        windows, starts = slice_windows(frame[AXES].to_numpy(np.float32), seconds, cfg)
        if not len(windows):
            continue
        chunks.append(windows)
        peak = np.linalg.norm(windows[:, :, :3], axis=2).max(axis=1)
        rows.extend((record, segment, float(seconds[s]), float(p)) for s, p in zip(starts, peak))
    if not rows:
        raise ValueError("no valid windows; check data.fs and data.time_tol")

    raw = np.concatenate(chunks).astype(np.float32)
    index = pd.DataFrame(rows, columns=["record", "segment_id", "start_s", "peak_accel"])
    index["rms_accel"] = np.sqrt(np.mean(np.sum(raw[:, :, :3] ** 2, axis=2), axis=1))
    index = index.join(metadata.set_index("record"), on="record", validate="many_to_one")

    metadata = metadata.copy()
    metadata["usable"] = metadata.record.isin(index.record)
    metadata.to_csv(out_dir / "recording_audit.csv", index=False)
    # One source id must stay in one fold even when it appears under several classes.
    metadata.groupby("source_group").agg(
        n_records=("record", "nunique"), classes=("label", lambda x: ",".join(sorted(x.unique())))
    ).reset_index().to_csv(out_dir / "source_provenance.csv", index=False)
    metadata = metadata[metadata.usable].reset_index(drop=True)
    index.to_csv(out_dir / "window_index.csv", index=False)

    bags = {r: g.index.to_numpy(dtype=np.int64) for r, g in index.groupby("record", sort=True)}
    label = {r: CLASSES.index(name) for r, name in zip(metadata.record, metadata.label)}
    return WindowSet(raw, index, metadata, bags, label)
