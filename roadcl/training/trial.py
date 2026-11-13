from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from roadcl.config import CLASSES, AXES, Config
from roadcl.data.replay import ReplayMemory, select_replay
from roadcl.data.windows import WindowSet
from roadcl.models.mlp import build_network, parameter_counts
from roadcl.training.evaluate import cpu_latency_ms, predict_recording
from roadcl.training.experiments import Experiment, method_spec
from roadcl.training.trainer import Distillation, FitResult, fit_model

TRACE_KEYS = ["split_repeat", "fold", "seed_id", "first_anomaly", "suite", "method", "representation",
              "objective", "memory_per_class", "kd_lambda", "width"]


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


def standardized_views(views: dict, representation: str, mean: np.ndarray, std: np.ndarray) -> dict[str, torch.Tensor]:
    return {rotation: torch.from_numpy(np.ascontiguousarray((z - mean) / std))
            for (rep, rotation), z in views.items() if rep == representation}


def _fit_scaler(raw: np.ndarray, bags: dict[str, np.ndarray], records: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sample = raw[np.concatenate([bags[r] for r in records])]
    return sample.mean(0), sample.std(0).clip(1e-5)


class Trial:
    """One fold, seed and arrival order of one experiment: stage 1, replay selection, stage 2."""

    def __init__(self, exp: Experiment, spec: TrialSpec, windows: WindowSet, views: dict, cfg: Config) -> None:
        self.exp, self.spec, self.ws, self.views, self.cfg = exp, spec, windows, views, cfg
        self.old_classes = [0, spec.first]
        self.seed = spec.run_seed(cfg.protocol.seed)
        self.meta = {
            "split_repeat": spec.split_repeat, "fold": spec.fold, "seed_id": spec.seed_id,
            "first_anomaly": CLASSES[spec.first], "suite": exp.suite, "method": exp.method,
            "representation": exp.representation, "objective": exp.objective,
            "memory_per_class": exp.memory_per_class, "kd_lambda": exp.kd_lambda, "width": exp.width,
        }

    def _measure(self, net, views, active, stage, rotation, payload, elapsed) -> list[dict]:
        rows = []
        for record in self.spec.test:
            if self.ws.label[record] not in active:
                continue
            pred, probs = predict_recording(net, views[rotation], self.ws.bags[record], active)
            rows.append({
                **self.meta, "stage": stage, "rotation": rotation, "record": record,
                "truth": CLASSES[self.ws.label[record]], "prediction": CLASSES[pred],
                "confidence": float(probs.max()), "buffer_feature_bytes": payload, "train_seconds": elapsed,
                **{f"prob_{CLASSES[c].lower()}": float(probs[active.index(c)]) if c in active else np.nan
                   for c in range(3)},
            })
        return rows

    def run(self) -> TrialOutput:
        exp, ws, cfg = self.exp, self.ws, self.cfg
        old = [r for r in self.spec.train if ws.label[r] in self.old_classes]
        new = [r for r in self.spec.train if ws.label[r] == 3 - self.spec.first]
        old_bags = {r: ws.bags[r] for r in old}
        new_bags = {r: ws.bags[r] for r in new}
        raw = self.views[exp.representation, "original"]
        dim = raw.shape[1]

        mean, std = _fit_scaler(raw, ws.bags, old)
        views = standardized_views(self.views, exp.representation, mean, std)
        features = views["original"]

        stage1 = build_network(dim, exp.width, self.seed)
        fit1 = fit_model(stage1, features, old_bags, ws.label, self.old_classes, exp.objective,
                         cfg.training.steps, self.seed, cfg.training)
        out = TrialOutput()
        for rotation in views:
            out.predictions += self._measure(stage1, views, self.old_classes, 1, rotation, 0, fit1.train_seconds)

        spec = method_spec(exp)
        memory = ReplayMemory({}, [])
        if spec.policy:
            memory = select_replay(old, ws.bags, ws.label, ws.index, features.numpy(), spec.policy,
                                   exp.memory_per_class, self.seed + 1)
        feature_bytes = memory.feature_bytes(dim)

        active = [0, 1, 2]
        if exp.method == "joint":
            mean, std = _fit_scaler(raw, ws.bags, self.spec.train)
            stage2_views = standardized_views(self.views, exp.representation, mean, std)
            net = build_network(dim, exp.width, self.seed)
            fit2 = fit_model(net, stage2_views["original"], {**old_bags, **new_bags}, ws.label, active,
                             exp.objective, 2 * cfg.training.steps, self.seed + 1, cfg.training)
        else:
            stage2_views = views
            net = copy.deepcopy(stage1)
            distill = None
            if spec.kd != "none" and memory.bags:
                distill = Distillation(stage1, self.old_classes, memory.bags, spec.kd == "gated", spec.kd_lambda)
            fit2 = fit_model(net, features, {**memory.bags, **new_bags}, ws.label, active, exp.objective,
                             cfg.training.steps, self.seed + 1, cfg.training, distill)
        for rotation in stage2_views:
            out.predictions += self._measure(net, stage2_views, active, 2, rotation, feature_bytes, fit2.train_seconds)

        params, weight_bytes = parameter_counts(net)
        out.resources = {
            **self.meta, "buffer_feature_bytes": feature_bytes, "buffer_metadata_bytes": memory.metadata_bytes(),
            "model_parameter_count": params, "model_weight_bytes": weight_bytes,
            "stage1_train_s": fit1.train_seconds, "stage2_train_s": fit2.train_seconds,
        }
        out.replay_selection = [{**out.resources, "class_name": c, "record": r, "window_index": i,
                                 "peak_accel": p, "policy": pol} for c, r, i, p, pol in memory.selection]
        out.losses = self._loss_rows(fit1, fit2)
        out.gates = [{**self.meta, "step": s, "teacher_confidence": c, "gate_weight": g, "kd_per_bag": k}
                     for s, c, g, k in fit2.gate_trace]
        out.replay_diagnostics = [
            {**self.meta, "record": r, "window_index": int(ix), "class_name": CLASSES[ws.label[r]],
             "peak_accel": float(ws.index.peak_accel.iat[ix])}
            for r, ids in memory.bags.items() for ix in ids
        ]
        batch = stage2_views["original"][:min(cfg.protocol.latency_windows, len(features))]
        out.resources["inference_ms_per_window_cpu"] = cpu_latency_ms(net, batch, cfg.protocol.latency_repeats)

        first_cell = self.spec.fold == 0 and self.spec.seed_id == 0 and self.spec.split_repeat == 0
        if cfg.protocol.checkpoint and first_cell and exp.suite == "main":
            out.checkpoint = {
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "mean": mean, "std": std, "classes": CLASSES, "axes": AXES,
                "window": cfg.data.window, "stride": cfg.data.stride, "fs": cfg.data.fs,
                "representation": exp.representation, "method": exp.method,
                "first_anomaly": CLASSES[self.spec.first], "width": exp.width, "split": "cv_training_fold_0",
            }
        return out

    def _loss_rows(self, fit1: FitResult, fit2: FitResult) -> list[dict]:
        return [{**self.meta, "stage": stage, "step": int(step), "task_loss": task, "total_loss": total,
                 "kd_unweighted": kd, "gate_mean": gate}
                for stage, fit in ((1, fit1), (2, fit2)) for step, task, total, kd, gate in fit.loss_trace]
