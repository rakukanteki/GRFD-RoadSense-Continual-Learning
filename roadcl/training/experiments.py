from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    suite: str
    method: str
    representation: str = "axes"
    objective: str = "bag"
    memory_per_class: int = 24
    kd_lambda: float = 1.0
    width: int = 32
    bag_size: int = 0


@dataclass(frozen=True)
class MethodSpec:
    policy: str | None
    kd: str
    kd_lambda: float


NO_MEMORY = ("fine_tune", "ewc", "lwf", "joint")
REPLAY_ONLY = ("global_random", "reservoir", "stratified_rms", "random", "recording_balanced", "impact_only",
               "stratified", "diverse", "herding", "kcenter")
RANDOM_REPLAY = ("er_ace", "weight_align")
DISTILLED = {
    "fixed_kd": ("diverse", "fixed"),
    "gated_kd": ("diverse", "gated"),
    "stratified_fixed_kd": ("stratified", "fixed"),
    "stratified_gated_kd": ("stratified", "gated"),
}
ABLATION = ["random", "recording_balanced", "impact_only", "stratified"]


def method_spec(exp: Experiment) -> MethodSpec:
    if exp.method in NO_MEMORY:
        return MethodSpec(None, "none", 0.0)
    if exp.method in RANDOM_REPLAY:
        return MethodSpec("random", "none", 0.0)
    if exp.method in REPLAY_ONLY:
        return MethodSpec(exp.method, "none", 0.0)
    if exp.method in DISTILLED:
        policy, kd = DISTILLED[exp.method]
        return MethodSpec(policy, kd, exp.kd_lambda)
    raise ValueError(f"unknown method {exp.method!r}")


def build_suite(profile: str) -> list[Experiment]:
    """Review comparisons; the full profile adds the capacity, representation, memory and gate sweeps."""
    if profile == "smoke":
        return [Experiment("component_ablation", m) for m in ABLATION]
    suite = [Experiment("main", m) for m in
             ["fine_tune", "random", "stratified", "diverse", "fixed_kd", "gated_kd", "joint"]]
    suite += [Experiment("regularization", m) for m in ("ewc", "lwf")]
    suite += [Experiment("component_ablation", m) for m in ABLATION]
    suite += [Experiment("main", m) for m in
              ["global_random", "reservoir", "er_ace", "weight_align", "herding", "stratified_gated_kd"]]
    suite += [Experiment("descriptor", "stratified_rms"), Experiment("augmented", "stratified_gated_kd")]
    if profile == "review":
        return suite
    suite += [Experiment("main", "stratified_fixed_kd")]
    suite += [Experiment("bag_size", "stratified", bag_size=n) for n in (1, 4, 16, 32)]
    suite += [Experiment("stratified_supervision", "stratified", "axes", "window"),
              Experiment("stratified_supervision", "stratified_gated_kd", "axes", "window")]
    suite += [Experiment("stratified_memory", "stratified_gated_kd", memory_per_class=n) for n in (8, 16, 48, 96)]
    suite += [Experiment("stratified_gate_weight", "stratified_gated_kd", kd_lambda=v) for v in (0.0, 0.25, 2.0)]
    suite += [Experiment("replay_policy", m) for m in ("herding", "kcenter")]
    suite += [Experiment("representation", "gated_kd", rep) for rep in ("gravity_relative", "hybrid")]
    suite += [Experiment("supervision", "gated_kd", "axes", "window")]
    suite += [Experiment("gate_ablation", "gated_kd", kd_lambda=v) for v in (0.0, 0.25, 2.0)]
    suite += [Experiment("memory_sweep", "gated_kd", memory_per_class=n) for n in (8, 16, 48, 96)]
    suite += [Experiment("model_capacity", "gated_kd", width=w) for w in (16, 64)]
    return suite
