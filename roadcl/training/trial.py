from __future__ import annotations

import copy
import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from roadcl.config import AXES, CLASSES, Config
from roadcl.data.features import window_features
from roadcl.data.replay import ReplayMemory, select_replay
from roadcl.data.windows import WindowSet
from roadcl.models.mlp import bag_pool, build_network, parameter_counts
from roadcl.training.evaluate import predict_recording
from roadcl.training.experiments import Experiment, method_spec
from roadcl.training.trainer import Distillation, FitResult, fit_model, fisher_diagonal

TRACE_KEYS = ["split_repeat", "fold", "seed_id", "first_anomaly", "suite", "method", "representation",
              "objective", "memory_per_class", "kd_lambda", "width", "bag_size"]


@dataclass
class TrialSpec:
    split_repeat: int
    fold: int
    seed_id: int
    first: int
    train: list[str]
    test: list[str]

    def run_seed(self, base: int) -> int:
        return base + 100003 * self.split_repeat + 10007 * self.fold + 257 * self.seed_id + 13 * self.first


@dataclass
class TrialOutput:
    predictions: list[dict] = field(default_factory=list)
    replay_selection: list[dict] = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    losses: list[dict] = field(default_factory=list)
    gates: list[dict] = field(default_factory=list)
    replay_diagnostics: list[dict] = field(default_factory=list)
    checkpoint: dict | None = None


def _fit_scaler(raw: np.ndarray, bags: dict[str, np.ndarray], records: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sample = raw[np.concatenate([bags[r] for r in records])]
    return sample.mean(0), sample.std(0).clip(1e-5)


def _process_peak_rss() -> int:
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


class Trial:
    """One fold, seed and arrival order of one experiment: stage 1, replay selection, stage 2."""

    def __init__(self, exp: Experiment, spec: TrialSpec, windows: WindowSet, views: dict, cfg: Config,
                 raw_windows: np.ndarray | None = None, validation: bool = False) -> None:
        self.exp, self.spec, self.ws, self.views, self.cfg = exp, spec, windows, views, cfg
        self.raw_windows, self.validation = raw_windows, validation
        self.old_classes = [0, spec.first]
        self.seed = spec.run_seed(cfg.protocol.seed)
        self.bag_size = exp.bag_size or cfg.training.windows_per_bag
        self.meta = {
            "split_repeat": spec.split_repeat, "fold": spec.fold, "seed_id": spec.seed_id,
            "first_anomaly": CLASSES[spec.first], "suite": exp.suite, "method": exp.method,
            "representation": exp.representation, "objective": exp.objective,
            "memory_per_class": exp.memory_per_class, "kd_lambda": exp.kd_lambda, "width": exp.width,
            "bag_size": self.bag_size,
        }

    def _measure(self, net, active, stage, rotation, payload, elapsed, mean, std) -> list[dict]:
        rows = []
        for record in self.spec.test:
            if self.ws.label[record] not in active:
                continue
            z = torch.from_numpy((self.views[self.exp.representation, rotation][self.ws.bags[record]] - mean) / std)
            pred, probs = predict_recording(net, z, slice(None), active)
            rows.append({
                **self.meta, "stage": stage, "rotation": rotation, "record": record,
                "truth": CLASSES[self.ws.label[record]], "prediction": CLASSES[pred],
                "confidence": float(probs.max()), "buffer_feature_bytes": payload, "train_seconds": elapsed,
                **{f"prob_{CLASSES[c].lower()}": float(probs[active.index(c)]) if c in active else np.nan
                   for c in range(3)},
            })
        return rows

    def run(self) -> TrialOutput | list[dict]:
        exp, ws, cfg, tc = self.exp, self.ws, self.cfg, self.cfg.training
        new_class = 3 - self.spec.first
        old = [r for r in self.spec.train if ws.label[r] in self.old_classes]
        old_bags = {r: ws.bags[r] for r in old}
        new_bags = {r: ws.bags[r] for r in self.spec.train if ws.label[r] == new_class}
        raw = self.views[exp.representation, "original"]
        dim = raw.shape[1]
        augmented = exp.suite == "augmented"

        mu, sd = _fit_scaler(raw, ws.bags, old)
        old_ids = np.concatenate(list(old_bags.values()))

        def old_view(name: str) -> torch.Tensor:
            values = self.views[exp.representation, name]
            z = np.full_like(values, np.nan)
            z[old_ids] = (values[old_ids] - mu) / sd
            return torch.from_numpy(z)

        features = old_view("original")
        stage1 = build_network(dim, exp.width, self.seed)
        fit1 = fit_model(stage1, features, old_bags, ws.label, self.old_classes, exp.objective, tc.steps,
                         self.seed, tc, bag_size=self.bag_size,
                         aug=old_view("train_aug") if augmented else None, augment_records=old_bags)

        update_start = time.perf_counter()
        spec = method_spec(exp)
        memory = ReplayMemory({}, [])
        if spec.policy:
            selection_features = features.numpy()
            if spec.policy == "herding":
                with torch.inference_mode():
                    embedding = F.normalize(stage1.net[:2](features[old_ids]), dim=1).numpy()
                selection_features = np.full((len(features), exp.width), np.nan, np.float32)
                selection_features[old_ids] = embedding
            memory = select_replay(old, ws.bags, ws.label, ws.index, selection_features, spec.policy,
                                   exp.memory_per_class, self.seed + 1)
        feature_bytes = memory.feature_bytes(dim)

        joint = exp.method == "joint"
        available = {**(old_bags if joint else memory.bags), **new_bags}
        mean, std = _fit_scaler(raw, ws.bags, self.spec.train) if joint else (mu, sd)
        update_ids = np.concatenate(list(available.values()))
        update_z = torch.from_numpy((raw[update_ids] - mean) / std)
        current, offset = {}, 0
        for record, ids in available.items():
            current[record] = np.arange(offset, offset + len(ids))
            offset += len(ids)
        update_aug = None
        if augmented:
            update_aug = torch.empty_like(update_z)
            for r in new_bags:
                update_aug[current[r]] = torch.from_numpy((self.views[exp.representation, "train_aug"][ws.bags[r]] - mean) / std)

        active = [0, 1, 2]
        if joint:
            net = build_network(dim, exp.width, self.seed)
            fit2 = fit_model(net, update_z, current, ws.label, active, exp.objective, 2 * tc.steps,
                             self.seed + 1, tc, bag_size=self.bag_size)
        else:
            net = copy.deepcopy(stage1)
            ewc = fisher_diagonal(stage1, features, old_bags, ws.label, self.old_classes, self.bag_size) \
                if exp.method == "ewc" else None
            distill = Distillation(stage1, self.old_classes, {r: current[r] for r in memory.bags},
                                   spec.kd == "gated", spec.kd_lambda) if spec.kd != "none" and memory.bags else None
            lwf = Distillation(stage1, self.old_classes, {}, False, tc.lwf_lambda) if exp.method == "lwf" else None
            fit2 = fit_model(net, update_z, current, ws.label, active, exp.objective, tc.steps, self.seed + 1, tc,
                             distill, bag_size=self.bag_size, lwf=lwf, ewc=ewc, aug=update_aug,
                             augment_records=new_bags, ace_new=new_class if exp.method == "er_ace" else None)
            if exp.method == "weight_align":
                with torch.no_grad():
                    weight = net.net[-1].weight
                    weight[new_class].mul_(weight[self.old_classes].norm(dim=1).mean()
                                           / weight[new_class].norm().clamp_min(1e-12))
        update_s = time.perf_counter() - update_start

        rotations = ["original"] if self.validation else [
            rot for rep, rot in self.views if rep == exp.representation and rot != "train_aug"]
        predictions = []
        for rotation in rotations:
            if not self.validation:
                predictions += self._measure(stage1, self.old_classes, 1, rotation, 0, fit1.train_seconds, mu, sd)
            predictions += self._measure(net, active, 2, rotation, feature_bytes, fit2.train_seconds, mean, std)
        if self.validation:
            return predictions

        params, weight_bytes = parameter_counts(net)
        out = TrialOutput(predictions=predictions)
        out.resources = {
            "run_seed": self.seed, "selected_lr": tc.lr, "incremental_update_s": update_s, **self.meta,
            "buffer_feature_bytes": feature_bytes, "buffer_metadata_bytes": memory.metadata_bytes(),
            "model_parameter_count": params, "model_weight_bytes": weight_bytes,
            "stage1_train_s": fit1.train_seconds, "stage2_train_s": fit2.train_seconds,
        }
        out.replay_selection = [{**out.resources, "class_name": c, "record": r, "window_index": i,
                                 "peak_accel": float(ws.index.peak_accel.iat[i]), "descriptor_value": s,
                                 "policy": pol} for c, r, i, s, pol in memory.selection]
        out.losses = self._loss_rows(fit1, fit2)
        out.gates = [{**self.meta, "step": s, "teacher_confidence": c, "gate_weight": g, "kd_per_bag": k}
                     for s, c, g, k in fit2.gate_trace]
        out.replay_diagnostics = [
            {**self.meta, "record": r, "window_index": int(ix), "class_name": CLASSES[ws.label[r]],
             "peak_accel": float(ws.index.peak_accel.iat[ix])}
            for r, ids in memory.bags.items() for ix in ids
        ]
        out.resources.update(self._latency(net, mean, std))
        out.resources["process_lifetime_peak_rss_bytes"] = _process_peak_rss()

        first_cell = self.spec.fold == 0 and self.spec.seed_id == 0 and self.spec.split_repeat == 0
        if cfg.protocol.checkpoint and first_cell and exp.suite == "main":
            out.checkpoint = {
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "mean": mean.tolist(), "std": std.tolist(), "classes": CLASSES, "axes": AXES,
                "window": cfg.data.window, "stride": cfg.data.stride, "representation": exp.representation,
                "method": exp.method, "first_anomaly": CLASSES[self.spec.first], "width": exp.width,
                "timing_record": self.spec.test[0], "fs": cfg.data.fs, "time_tol": cfg.data.time_tol,
                "split": "cv_training_fold_0",
            }
        return out

    def _latency(self, net, mean, std) -> dict:
        """Time features, scaling, MLP and pooling for the first held-out recording."""
        record = self.spec.test[0]
        sample = self.raw_windows[self.ws.bags[record]]

        def classify():
            z = torch.from_numpy((window_features(sample, self.exp.representation) - mean) / std)
            return torch.softmax(bag_pool(net(z), dim=0), dim=0).argmax()

        with torch.inference_mode():
            classify()
            start = time.perf_counter()
            for _ in range(self.cfg.protocol.latency_repeats):
                classify()
        return {"recording_latency_ms": (time.perf_counter() - start) * 1000 / self.cfg.protocol.latency_repeats,
                "timing_record": record, "timing_windows": len(sample), "platform": platform.platform(),
                "scaler_bytes": mean.nbytes + std.nbytes}

    def _loss_rows(self, fit1: FitResult, fit2: FitResult) -> list[dict]:
        return [{**self.meta, "stage": stage, "step": int(step), "task_loss": task, "total_loss": total,
                 "kd_unweighted": kd, "gate_mean": gate}
                for stage, fit in ((1, fit1), (2, fit2)) for step, task, total, kd, gate in fit.loss_trace]
