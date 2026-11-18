from __future__ import annotations

import hashlib
import logging
import random
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import psutil

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


@contextmanager
def sample_trial_ram(interval: float = 0.01):
    """Sample process RSS every `interval` seconds; short allocations can fall between samples."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    usage = {"trial_start_rss_bytes": baseline, "sampled_trial_peak_rss_bytes": baseline}
    stop = threading.Event()

    def sample():
        while not stop.wait(interval):
            usage["sampled_trial_peak_rss_bytes"] = max(usage["sampled_trial_peak_rss_bytes"], process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield usage
    finally:
        stop.set()
        worker.join()
        usage["sampled_trial_peak_rss_bytes"] = max(usage["sampled_trial_peak_rss_bytes"], process.memory_info().rss)
        usage["sampled_trial_rss_increase_bytes"] = usage["sampled_trial_peak_rss_bytes"] - baseline
