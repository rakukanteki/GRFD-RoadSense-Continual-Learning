from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from roadcl.config import CLASSES
from roadcl.metrics.scores import GROUP_KEYS

PAIRED_SUITES = ("main", "component_ablation", "augmented", "descriptor")


def session_weights(pred: pd.DataFrame, iterations: int, seed: int) -> tuple[np.ndarray, dict]:
    """Bootstrap weights over source groups, resampled within each class-composition stratum; row 0 is the observed sample."""
    groups = sorted(pred.source_group.unique())
    index = {g: i for i, g in enumerate(groups)}
    weights = np.zeros((iterations + 1, len(groups)), dtype=int)
    weights[0] = 1
    rng = np.random.default_rng(seed + 951)
    strata = pred[["source_group", "truth"]].drop_duplicates().groupby("source_group").truth.agg(
        lambda v: tuple(sorted(v)))
    for signature in sorted(set(strata)):
        ix = [index[g] for g, labels in strata.items() if labels == signature]
        weights[1:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), iterations)
    return weights, index


def clustered_summary(pred: pd.DataFrame, out: Path, iterations: int, seed: int) -> pd.DataFrame:
    """Pooled out-of-fold recording metrics per configuration with shared session-bootstrap intervals."""
    weights, index = session_weights(pred, iterations, seed)
    keys = GROUP_KEYS[3:]
    draws, summary, confusions = {}, [], []
    for values, g in pred.groupby(keys, sort=True):
        config = dict(zip(keys, values))
        old = [0, CLASSES.index(config["first_anomaly"])]
        new = 3 - old[1]
        metrics = defaultdict(list)
        for _, rep in g.groupby(["split_repeat", "seed_id"]):
            recalls = {}
            for stage, part in rep.groupby("stage"):
                active = old if stage == 1 else [0, 1, 2]
                counts = np.zeros((len(index), 3, 3), dtype=int)
                np.add.at(counts, (part.source_group.map(index).to_numpy(), part.truth.map(CLASSES.index).to_numpy(),
                                   part.prediction.map(CLASSES.index).to_numpy()), 1)
                cm = (weights @ counts.reshape(-1, 9)).reshape(-1, 3, 3)[:, active][:, :, active]
                diag = cm.diagonal(axis1=1, axis2=2)
                recall = diag / cm.sum(axis=2)
                precision = diag / np.maximum(cm.sum(axis=1), 1)
                f1 = 2 * diag / np.maximum(cm.sum(axis=1) + cm.sum(axis=2), 1)
                metrics[f"macro_f1_s{stage}"].append(f1.mean(axis=1))
                metrics[f"balanced_accuracy_s{stage}"].append(recall.mean(axis=1))
                for j, k in enumerate(active):
                    metrics[f"{CLASSES[k]}_precision_s{stage}"].append(precision[:, j])
                    metrics[f"{CLASSES[k]}_recall_s{stage}"].append(recall[:, j])
                    for l, h in enumerate(active):
                        confusions.append(dict(config, stage=stage, split_repeat=part.split_repeat.iat[0],
                                               seed_id=part.seed_id.iat[0], truth=CLASSES[k],
                                               prediction=CLASSES[h], recordings=cm[0, j, l]))
                recalls[stage] = recall
            transfer = recalls[2][:, old].mean(axis=1) - recalls[1].mean(axis=1)
            metrics["backward_transfer"].append(transfer)
            metrics["old_forgetting_pp"].append(-100 * transfer)
            metrics["new_recall"].append(recalls[2][:, new])
        for metric, arrays in metrics.items():
            draw = np.mean(arrays, axis=0)
            lo, hi = np.quantile(draw[1:], [0.025, 0.975])
            summary.append(dict(config, metric=metric, estimate=draw[0], lower=lo, upper=hi))
            if config["suite"] in PAIRED_SUITES and metric == "macro_f1_s2":
                draws[config["first_anomaly"], config["rotation"], config["suite"], config["method"]] = draw
    table = pd.DataFrame(summary)
    table.to_csv(out / "scores_summary.csv", index=False)
    pd.DataFrame(confusions).to_csv(out / "confusion_matrices.csv", index=False)
    paired_differences(draws).to_csv(out / "paired_method_comparisons.csv", index=False)
    return table


def paired_differences(draws: dict) -> pd.DataFrame:
    """Macro-F1 differences on shared bootstrap draws against the suite reference and every method sorting before it."""
    rows = []
    for (order, view, suite, method), draw in draws.items():
        ref = ("main", "stratified_gated_kd") if suite == "augmented" else \
            ("main", "stratified") if suite == "descriptor" else (suite, "random")
        refs = [ref]
        if suite in ("main", "component_ablation") and method != "random":
            refs += [(suite, m) for o, v, s, m in draws if (o, v, s) == (order, view, suite) and m != "random" and m < method]
        for comparison in refs:
            reference = draws.get((order, view, *comparison))
            if reference is None or (suite, method) == comparison:
                continue
            diff = draw - reference
            lo, hi = np.quantile(diff[1:], [0.025, 0.975])
            rows.append(dict(first_anomaly=order, rotation=view, suite=suite, method=method,
                             reference="/".join(comparison), macro_f1_difference=diff[0], lower=lo, upper=hi))
    return pd.DataFrame(rows)
