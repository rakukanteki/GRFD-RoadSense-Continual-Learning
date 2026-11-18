from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from roadcl.config import DataConfig

REPRESENTATIONS = ("axes", "gravity_relative", "hybrid")
ROTATIONS = {
    "original": np.eye(3, dtype=np.float32),
    "x90": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    "y90": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
    "z90": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
}
PERTURBATIONS = ("sensor_noise", "gyro_missing")
GYRO_DROP_PROB = 0.25


def _rms(signal: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(signal * signal, axis=1))


def gravity_relative(accel: np.ndarray, gyro: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Vertical, horizontal and total magnitude of acceleration and angular rate."""
    unit = gravity / np.maximum(np.linalg.norm(gravity, axis=-1, keepdims=True), 1e-8)
    channels = []
    for vector in (accel, gyro):
        vertical = (vector * unit).sum(axis=-1)
        horizontal = np.linalg.norm(vector - vertical[..., None] * unit, axis=-1)
        channels.extend((vertical, horizontal, np.linalg.norm(vector, axis=-1)))
    return np.stack(channels, axis=-1)


def window_features(windows: np.ndarray, representation: str) -> np.ndarray:
    """Six statistics per channel: mean, SD, RMS, min, max and RMS of the first difference."""
    accel, gyro, gravity = np.split(windows, 3, axis=-1)
    if representation == "axes":
        signals = np.concatenate([accel, gyro], axis=-1)
    elif representation == "gravity_relative":
        signals = gravity_relative(accel, gyro, gravity)
    elif representation == "hybrid":
        signals = np.concatenate([accel, gyro, gravity_relative(accel, gyro, gravity)], axis=-1)
    else:
        raise ValueError(f"unknown representation {representation!r}")
    stats = [signals.mean(1), signals.std(1), _rms(signals), signals.min(1), signals.max(1),
             _rms(np.diff(signals, axis=1))]
    return np.concatenate(stats, axis=1).astype(np.float32)


def rotate(raw: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (raw.reshape(len(raw), raw.shape[1], 3, 3) @ matrix.T).reshape(raw.shape)


def make_feature_views(raw: np.ndarray, representations: list[str], cfg: DataConfig) -> dict[tuple[str, str], np.ndarray]:
    """Features of the original, rotated, noisy and gyro-free test views and the augmented training view."""
    views = {}
    for rotation, matrix in ROTATIONS.items():
        rotated = rotate(raw, matrix)
        for rep in representations:
            views[rep, rotation] = window_features(rotated, rep)
    if "gravity_relative" in representations:
        reference = views["gravity_relative", "original"]
        for rotation in ROTATIONS:
            if not np.allclose(reference, views["gravity_relative", rotation], rtol=1e-4, atol=1e-4):
                raise AssertionError("gravity-relative features changed under a rigid rotation")
    # Draw order (rotations, gyro dropout, noise) is fixed so the seeded views are reproducible.
    rng = np.random.default_rng(cfg.perturbation_seed)
    quat = rng.normal(size=(len(raw), 4))
    matrices = Rotation.from_quat(quat / np.linalg.norm(quat, axis=1, keepdims=True)).as_matrix().astype(np.float32)
    augmented = (raw.reshape(len(raw), -1, 3) @ matrices.transpose(0, 2, 1)).reshape(raw.shape)
    augmented[rng.random(len(raw)) < GYRO_DROP_PROB, :, 3:6] = 0
    noisy = raw + rng.normal(0, cfg.noise_std, raw.shape).astype(np.float32)
    no_gyro = raw.copy()
    no_gyro[:, :, 3:6] = 0.0
    for rep in representations:
        views[rep, "train_aug"] = window_features(augmented, rep)
        views[rep, "sensor_noise"] = window_features(noisy, rep)
        views[rep, "gyro_missing"] = window_features(no_gyro, rep)
    return views
