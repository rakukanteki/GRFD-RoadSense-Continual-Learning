from __future__ import annotations

from collections import Counter

import numpy as np


class ClassBalancedReservoir:
    """Reservoir sampling with a separate fixed-size buffer per class."""

    def __init__(self, capacity_per_class: int, seed: int = 42):
        self.capacity = capacity_per_class
        self.rng = np.random.default_rng(seed)
        self.buffers: dict[int, list[int]] = {}
        self.seen: Counter = Counter()

    def update(self, indices: np.ndarray, labels: np.ndarray) -> None:
        for idx in indices:
            cls = int(labels[idx])
            self.seen[cls] += 1
            buf = self.buffers.setdefault(cls, [])
            if len(buf) < self.capacity:
                buf.append(int(idx))
                continue
            j = int(self.rng.integers(self.seen[cls]))
            if j < self.capacity:
                buf[j] = int(idx)

    def indices(self) -> np.ndarray:
        return np.asarray([i for buf in self.buffers.values() for i in buf], dtype=int)

    def counts(self) -> dict[int, int]:
        return {cls: len(buf) for cls, buf in self.buffers.items()}

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.buffers.values())
