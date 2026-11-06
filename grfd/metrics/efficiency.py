from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def student_footprint(model: nn.Module, sample: np.ndarray, warmup: int = 2, repeats: int = 3) -> dict:
    params = sum(p.numel() for p in model.parameters())
    size = sum(p.numel() * p.element_size() for p in model.parameters())

    device = next(model.parameters()).device
    model.to("cpu").eval()
    batch = torch.from_numpy(sample)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        for _ in range(repeats):
            start = time.perf_counter()
            model(batch)
            times.append(time.perf_counter() - start)
    model.to(device)

    return {
        "params": int(params),
        "student_mb": size / 1024**2,
        "ms_per_sample_cpu": 1000 * float(np.median(times)) / len(sample),
    }


def teacher_latency_ms(teacher, sample: np.ndarray, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        teacher.predict_proba(sample)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        teacher.predict_proba(sample)
        times.append(1000 * (time.perf_counter() - start) / len(sample))
    return float(np.median(times))
