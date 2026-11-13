from __future__ import annotations

import numpy as np
import pandas as pd

SUMMARY_KEYS = ["suite", "method", "first_anomaly", "representation", "objective", "memory_per_class",
                "kd_lambda", "width", "rotation"]
SUMMARY_METRICS = ["macro_f1_s2", "old_forgetting_pp", "new_recall", "ece_s2", "buffer_feature_bytes",
                   "stage2_train_s"]
PAIR_KEYS = ["split_repeat", "fold", "seed_id", "first_anomaly", "representation", "objective"]
COMPARISONS = [("diverse", "random"), ("diverse", "stratified"), ("gated_kd", "fixed_kd"),
               ("gated_kd", "diverse"), ("gated_kd", "joint"), ("stratified_gated_kd", "stratified_fixed_kd"),
               ("stratified_gated_kd", "stratified"), ("stratified_fixed_kd", "stratified"),
               ("stratified", "random"), ("stratified", "fine_tune"), ("stratified", "joint")]
PAIR_NOTE = ("Descriptive paired fold/seed bootstrap; trials share recordings, so these are not "
             "confidence bounds over independent recordings")


def bootstrap_means(values: np.ndarray, iterations: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)


def summarize(scores: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    """Mean, SD and percentile bootstrap interval of every metric per configuration."""
    if scores.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed + 900)
    rows = []
    for key, group in scores.groupby(SUMMARY_KEYS):
        row = dict(zip(SUMMARY_KEYS, key), n_trials=len(group))
        for col in SUMMARY_METRICS:
            values = group[col].dropna().to_numpy(np.float64)
            if not len(values):
                continue
            means = bootstrap_means(values, iterations, rng)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{col}_ci_low"], row[f"{col}_ci_high"] = map(float, np.quantile(means, [0.025, 0.975]))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(scores: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    """Matched-trial differences between methods of the main suite, per arrival order."""
    if scores.empty:
        return pd.DataFrame()
    base = scores[(scores.suite == "main") & (scores.rotation == "original")]
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for challenger, reference in COMPARISONS:
        merged = base[base.method == challenger].merge(base[base.method == reference], on=PAIR_KEYS,
                                                       suffixes=("_a", "_b"))
        for order, group in merged.groupby("first_anomaly"):
            for col in ("macro_f1_s2", "old_forgetting_pp", "new_recall"):
                delta = (group[f"{col}_a"] - group[f"{col}_b"]).to_numpy()
                if not len(delta):
                    continue
                means = bootstrap_means(delta, iterations, rng)
                rows.append({
                    "challenger": challenger, "reference": reference, "first_anomaly": order, "metric": col,
                    "paired_trials": len(delta), "mean_difference": float(delta.mean()),
                    "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
                    "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()), "note": PAIR_NOTE,
                })
    return pd.DataFrame(rows)
