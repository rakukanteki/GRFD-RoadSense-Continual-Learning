from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from roadcl.config import AXES, CLASSES, DataConfig
from roadcl.data.features import window_features
from roadcl.data.windows import slice_windows
from roadcl.inference.predict import load_checkpoint
from roadcl.models.mlp import bag_pool
from roadcl.utils.runtime import file_sha256


def device_name() -> str:
    tree = Path("/proc/device-tree/model")
    if tree.is_file():
        return tree.read_bytes().rstrip(b"\0").decode(errors="replace")
    return platform.processor() or platform.machine()


def benchmark_checkpoint(checkpoint: str | Path, data: DataConfig, out_dir: Path, repeats: int,
                         threads: int, record: str | None = None) -> pd.DataFrame:
    """Time framing, features, scaling, MLP and pooling for one recording held in memory; CSV I/O is untimed."""
    net, ckpt = load_checkpoint(checkpoint)
    record = record or ckpt.get("timing_record")
    if not record:
        raise ValueError("give a record in Class/source_file form")
    cls, source = record.split("/", 1)
    frames = [b[b.source_file.eq(source)] for b in pd.read_csv(
        Path(data.data_dir) / data.files[cls], usecols=["source_file", "segment_id", "seconds_elapsed", *AXES],
        chunksize=20000)]
    rec = pd.concat(frames)
    if rec.empty:
        raise ValueError(f"recording {record} is absent from the configured source")
    segments = [(g[AXES].to_numpy(np.float32), g.seconds_elapsed.to_numpy()) for _, g in rec.groupby("segment_id")]
    cfg = DataConfig(fs=ckpt.get("fs", data.fs), time_tol=ckpt.get("time_tol", data.time_tol),
                     window=ckpt["window"], stride=ckpt["stride"])

    def classify() -> tuple[int, int]:
        chunks = [w for w in (slice_windows(v, t, cfg)[0] for v, t in segments) if len(w)]
        if not chunks:
            raise ValueError("no valid recording windows")
        x = window_features(np.concatenate(chunks), ckpt["representation"])
        return len(x), int(bag_pool(net(torch.from_numpy((x - ckpt["mean"]) / ckpt["std"])), dim=0).argmax())

    sha = file_sha256(checkpoint)
    rows = []
    with torch.inference_mode():
        classify()
        for iteration in range(repeats):
            start = time.perf_counter_ns()
            n, prediction = classify()
            rows.append(dict(iteration=iteration, ms=(time.perf_counter_ns() - start) / 1e6, windows=n, record=record,
                             prediction=CLASSES[prediction], platform=platform.platform(), machine=platform.machine(),
                             device=device_name(), torch=torch.__version__, numpy=np.__version__, threads=threads,
                             checkpoint_sha256=sha))
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "device_latency.csv", index=False)
    return frame
