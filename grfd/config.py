from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass
class DataConfig:
    csv_path: str = "data/all_labeled_data.csv"
    label_col: str = "label"
    group_candidates: list[str] = field(
        default_factory=lambda: ["source_file", "recording_id", "group_id", "session_id", "drive_id"]
    )
    drop_cols: list[str] = field(
        default_factory=lambda: [
            "seconds_elapsed",
            "barometer_relativeAltitude",
            "location_verticalAccuracy",
            "Temperature (Â°C)",
            "Temperature (°C)",
            "Humidity (%)",
            "Cloud Cover (%)",
            "Wind Speed (km/h)",
        ]
    )
    max_rows: int | None = None


@dataclass
class StudentConfig:
    hidden_dim: int = 96
    n_experts: int = 3
    dropout: float = 0.1


@dataclass
class TeacherConfig:
    n_estimators: int = 100
    max_features: str = "sqrt"
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced_subsample"
    n_jobs: int = -1


@dataclass
class TrainingConfig:
    epochs: int = 30
    patience: int = 8
    min_delta: float = 1e-4
    batch_size: int = 256
    eval_batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    temperature: float = 3.0
    kd_weight: float = 0.6
    lwf_weight: float = 0.15
    log_every: int = 5


@dataclass
class ReliabilityConfig:
    min_weight: float = 0.10
    max_weight: float = 1.00
    min_oob_rows: int = 50
    max_iter: int = 150


@dataclass
class VariantConfig:
    replay: int = 0
    lwf: bool = False
    adaptive_kd: bool = False


def _default_variants() -> dict[str, VariantConfig]:
    return {
        "naive": VariantConfig(),
        "replay": VariantConfig(replay=750),
        "lwf": VariantConfig(lwf=True),
        "replay_lwf": VariantConfig(replay=750, lwf=True),
        "fixed_kd_750": VariantConfig(replay=750),
        "adaptive_kd_750": VariantConfig(replay=750, adaptive_kd=True),
        "adaptive_kd_0": VariantConfig(adaptive_kd=True),
        "adaptive_kd_3000": VariantConfig(replay=3000, adaptive_kd=True),
    }


@dataclass
class ContinualConfig:
    n_tasks: int = 5
    run: list[str] = field(default_factory=lambda: list(_default_variants()))
    variants: dict[str, VariantConfig] = field(default_factory=_default_variants)
    save_task_teachers: bool = True


@dataclass
class BaselineConfig:
    rf_trees: int = 200
    lr_max_iter: int = 300
    hgb_max_iter: int = 150


@dataclass
class ExperimentConfig:
    name: str = "grfd_cv_continual"
    output_dir: str = "runs"
    seed: int = 42
    device: str = "auto"
    n_folds: int = 3
    val_size: float = 0.15
    run_joint: bool = True
    run_baselines: bool = True
    run_continual: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    continual: ContinualConfig = field(default_factory=ContinualConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

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
            for name, item in value.items():
                if is_dataclass(current.get(name)) and isinstance(item, dict):
                    _merge(current[name], item, f"{prefix}{key}.{name}.")
                else:
                    current[name] = item
        else:
            setattr(target, key, value)


def _nest(keys: list[str], value: Any) -> dict:
    for key in reversed(keys):
        value = {key: value}
    return value


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

    cfg.continual.variants = {
        name: spec if isinstance(spec, VariantConfig) else VariantConfig(**spec)
        for name, spec in cfg.continual.variants.items()
    }
    unknown = [name for name in cfg.continual.run if name not in cfg.continual.variants]
    if unknown:
        raise KeyError(f"continual.run lists undefined variants: {unknown}")
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False, allow_unicode=True)
