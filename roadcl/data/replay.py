from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from roadcl.config import CLASSES

POLICIES = ("random", "stratified", "diverse", "herding", "kcenter")


@dataclass
class ReplayMemory:
    bags: dict[str, np.ndarray]
    selection: list[tuple[str, str, int, float, str]]

    @property
    def n_windows(self) -> int:
        return sum(len(v) for v in self.bags.values())

    def feature_bytes(self, dim: int) -> int:
        return self.n_windows * dim * np.dtype("float32").itemsize

    def metadata_bytes(self) -> int:
        return self.n_windows * 8


def impact_stratified(records: list[str], bags: dict[str, np.ndarray], peak: np.ndarray,
                      n_keep: int, rng: np.random.Generator) -> list[int]:
    """Round-robin over recordings, cycling through peak-acceleration tertiles inside each one."""
    queues = {}
    for record in records:
        ordered = bags[record][np.argsort(peak[bags[record]])]
        queues[record] = [list(rng.permutation(tier)) for tier in np.array_split(ordered, 3)]
        rng.shuffle(queues[record])
    chosen: list[int] = []
    while len(chosen) < n_keep:
        progress = False
        for record in rng.permutation(records):
            queue = queues[record]
            while queue and not queue[0]:
                queue.pop(0)
            if not queue:
                continue
            tier = queue.pop(0)
            chosen.append(int(tier.pop()))
            progress = True
            if tier:
                queue.append(tier)
            if len(chosen) == n_keep:
                break
        if not progress:
            break
    return chosen


def herding(data: np.ndarray, n_keep: int) -> list[int]:
    """Greedy selection whose running mean tracks the class mean."""
    center = data.mean(axis=0)
    chosen: list[int] = []
    running = np.zeros(data.shape[1], np.float64)
    for t in range(n_keep):
        dist = ((data - ((t + 1) * center - running)) ** 2).sum(1)
        if chosen:
            dist[chosen] = np.inf
        j = int(np.argmin(dist))
        chosen.append(j)
        running += data[j]
    return chosen


def farthest_point(data: np.ndarray, n_keep: int, peak: np.ndarray | None = None,
                   record_ids: np.ndarray | None = None) -> list[int]:
    """k-center selection; with peak and record ids it adds the coverage bonuses of diverse replay."""
    diverse = peak is not None
    if diverse:
        bins = np.quantile(peak, [1 / 3, 2 / 3])
        tier = np.searchsorted(bins, peak)
        rec_count: Counter = Counter()
        tier_count: Counter = Counter()
        chosen = [int(np.argmax(peak))]
        rec_count[record_ids[chosen[0]]] += 1
        tier_count[int(tier[chosen[0]])] += 1
    else:
        center = data.mean(axis=0)
        chosen = [int(np.argmax(((data - center) ** 2).sum(1)))]
    min_dist = ((data - data[chosen[0]]) ** 2).sum(1)
    for _ in range(1, n_keep):
        score = min_dist.copy()
        if diverse:
            scale = max(float(np.median(score)), 1e-8)
            rec_bonus = np.array([1 / (1 + rec_count[r]) for r in record_ids])
            tier_bonus = np.array([1 / (1 + tier_count[int(t)]) for t in tier])
            score = score / scale + 1.5 * rec_bonus + 0.5 * tier_bonus
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        if diverse:
            rec_count[record_ids[j]] += 1
            tier_count[int(tier[j])] += 1
        min_dist = np.minimum(min_dist, ((data - data[j]) ** 2).sum(1))
    return chosen


def select_replay(records: list[str], bags: dict[str, np.ndarray], label: dict[str, int],
                  index: pd.DataFrame, features: np.ndarray, policy: str, per_class: int,
                  seed: int) -> ReplayMemory:
    """Pick `per_class` windows of every old class from its training recordings."""
    if per_class <= 0:
        return ReplayMemory({}, [])
    if policy not in POLICIES:
        raise ValueError(f"unknown replay policy {policy!r}")
    rng = np.random.default_rng(seed)
    peak = index.peak_accel.to_numpy()
    memory: dict[str, list[int]] = {}
    selection = []
    for k in sorted({label[r] for r in records}):
        group = sorted(r for r in records if label[r] == k)
        ids = np.concatenate([bags[r] for r in group])
        n_keep = min(per_class, len(ids))
        if policy == "random":
            chosen = rng.choice(ids, n_keep, replace=False).tolist()
        elif policy == "stratified":
            chosen = impact_stratified(group, bags, peak, n_keep, rng)
        else:
            data = features[ids]
            if policy == "herding":
                local = herding(data, n_keep)
            elif policy == "kcenter":
                local = farthest_point(data, n_keep)
            else:
                record_ids = np.array([index.record.iat[i] for i in ids])
                local = farthest_point(data, n_keep, peak[ids], record_ids)
            chosen = [int(ids[j]) for j in local]
        if len(chosen) != n_keep or len(chosen) != len(set(chosen)):
            raise AssertionError("replay selection must hold unique windows")
        for ix in chosen:
            record = index.record.iat[ix]
            memory.setdefault(record, []).append(ix)
            selection.append((CLASSES[k], record, ix, float(peak[ix]), policy))
    return ReplayMemory({r: np.array(v, dtype=np.int64) for r, v in memory.items()}, selection)
