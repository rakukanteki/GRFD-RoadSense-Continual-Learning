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


@dataclass(frozen=True)
class MethodSpec:
    policy: str | None
    kd: str
    kd_lambda: float


REPLAY_ONLY = ("random", "stratified", "diverse", "herding", "kcenter")
DISTILLED = {
    "fixed_kd": ("diverse", "fixed"),
    "gated_kd": ("diverse", "gated"),
    "stratified_fixed_kd": ("stratified", "fixed"),
    "stratified_gated_kd": ("stratified", "gated"),
}


def method_spec(exp: Experiment) -> MethodSpec:
    if exp.method in ("fine_tune", "joint"):
        return MethodSpec(None, "none", 0.0)
    if exp.method in REPLAY_ONLY:
        return MethodSpec(exp.method, "none", 0.0)
    if exp.method in DISTILLED:
        policy, kd = DISTILLED[exp.method]
        return MethodSpec(policy, kd, exp.kd_lambda)
    raise ValueError(f"unknown method {exp.method!r}")


def build_suite(profile: str) -> list[Experiment]:
    """Main comparison plus the ablation suites reported in the paper."""
    if profile == "smoke":
        return [Experiment("main", m) for m in
                ["fine_tune", "stratified", "stratified_fixed_kd", "stratified_gated_kd", "joint"]]
    suite = [Experiment("main", m) for m in
             ["fine_tune", "random", "stratified", "diverse", "fixed_kd", "gated_kd", "joint",
              "stratified_fixed_kd", "stratified_gated_kd"]]
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
