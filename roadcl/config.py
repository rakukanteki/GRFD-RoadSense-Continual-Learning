from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

CLASSES = ["Normal", "Bump", "Pothole"]
AXES = [f"{sensor}_{axis}" for sensor in ("accelerometer", "gyroscope", "gravity") for axis in "xyz"]


@dataclass
class DataConfig:
    data_dir: str = "data/Road Anomalies"
    files: dict[str, str] = field(
        default_factory=lambda: {
            "Normal": "normal_road_data_89_82Hz.csv",
            "Bump": "combined_bump.csv",
            "Pothole": "combined_pothole.csv",
        }
    )
    aliases: dict[str, str] = field(
        default_factory=lambda: {
            "31.csv": "30.csv",
            "47.csv": "46.csv",
            "68.csv": "46.csv",
            "79.csv": "46.csv",
            "78.csv": "45.csv",
        }
    )
    fs: float = 89.82
    time_tol: float = 1e-7
    window: int = 90
    stride: int = 45
    noise_std: float = 0.10
    perturbation_seed: int = 1007
    session_manifest: str | None = None


@dataclass
class TrainingConfig:
    steps: int = 100
    bags_per_step: int = 12
    windows_per_bag: int = 8
    kd_bags_per_step: int = 8
    temperature: float = 2.0
    gate_power: float = 1.0
    lr: float = 3e-3
    ewc_lambda: float = 10.0
    lwf_lambda: float = 1.0
    grad_clip: float = 5.0


@dataclass
class ProtocolConfig:
    profile: str = "full"
    folds: int = 5
    inner_folds: int = 2
    lr_grid: list[float] = field(default_factory=lambda: [1e-3, 3e-3])
    split_repeats: int = 1
    seeds: int = 5
    seed: int = 42
    threads: int = 1
    bootstrap_iterations: int = 1000
    latency_repeats: int = 10
    checkpoint: bool = True


@dataclass
class ExperimentConfig:
    name: str = "road_cl"
    output_dir: str = "runs"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.experiment.name


def _merge(target: Any, updates: dict, prefix: str = "") -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            raise KeyError(f"unknown config key: {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{prefix}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            setattr(target, key, value)


def _nest(keys: list[str], value: Any) -> dict:
    for key in reversed(keys):
        value = {key: value}
    return value


def _validate(cfg: Config) -> None:
    if cfg.protocol.profile not in ("smoke", "review", "full"):
        raise ValueError(f"protocol.profile must be 'smoke', 'review' or 'full', got {cfg.protocol.profile!r}")
    if min(cfg.protocol.folds, cfg.protocol.inner_folds, cfg.data.window) < 2 or min(cfg.protocol.lr_grid) <= 0:
        raise ValueError("folds, inner_folds and window must be at least 2 and every lr_grid value positive")
    positive = {
        "training.steps": cfg.training.steps,
        "training.bags_per_step": cfg.training.bags_per_step,
        "training.windows_per_bag": cfg.training.windows_per_bag,
        "training.kd_bags_per_step": cfg.training.kd_bags_per_step,
        "protocol.seeds": cfg.protocol.seeds,
        "protocol.split_repeats": cfg.protocol.split_repeats,
        "protocol.threads": cfg.protocol.threads,
        "protocol.latency_repeats": cfg.protocol.latency_repeats,
        "data.stride": cfg.data.stride,
        "protocol.bootstrap_iterations": cfg.protocol.bootstrap_iterations,
    }
    bad = [name for name, value in positive.items() if value < 1]
    if bad:
        raise ValueError(f"these settings must be positive: {bad}")
    missing = [name for name in CLASSES if name not in cfg.data.files]
    if missing:
        raise KeyError(f"data.files has no entry for {missing}")


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    cfg = Config()
    if path is not None:
        with open(path) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _merge(cfg, _nest(key.strip().split("."), yaml.safe_load(raw)))
    _validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False, allow_unicode=True)
