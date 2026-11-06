from __future__ import annotations

import logging
import time

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from grfd.config import Config
from grfd.metrics import label_metrics, macro_auc
from grfd.models import aligned_proba

from .context import FoldContext, Recorder

log = logging.getLogger(__name__)


def build_baselines(cfg: Config) -> dict:
    seed = cfg.experiment.seed
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=cfg.baselines.lr_max_iter, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.baselines.rf_trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=cfg.teacher.n_jobs,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=cfg.baselines.hgb_max_iter, random_state=seed),
    }


def run_baselines(ctx: FoldContext, cfg: Config, rec: Recorder) -> None:
    X_train, y_train = ctx.X[ctx.split.train], ctx.y[ctx.split.train]
    X_test, y_test = ctx.X_test, ctx.y_test

    for name, est in build_baselines(cfg).items():
        start = time.perf_counter()
        est.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        metrics = label_metrics(y_test, est.predict(X_test), ctx.class_names)
        metrics["auc_ovr"] = macro_auc(y_test, aligned_proba(est, X_test, ctx.n_classes), ctx.n_classes)
        rec.add("baseline_model_comparison", {"fold": ctx.fold, "model": name, "fit_seconds": fit_seconds, **metrics})

        if name == "random_forest":
            joblib.dump(est, ctx.fold_dir / "baseline_random_forest.joblib")
        log.info("BASELINE %-22s acc=%.4f macro_f1=%.4f", name, metrics["accuracy"], metrics["macro_f1"])
