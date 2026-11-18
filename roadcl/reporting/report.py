from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadcl.config import Config
from roadcl.metrics import aggregate_scores, clustered_summary
from roadcl.reporting.figures import make_figures
from roadcl.reporting.tables import dataset_tables, research_report, resource_and_descriptor_tables, write_diagnostics


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def report_run(run_dir: Path, cfg: Config, figures: bool = True) -> None:
    """Score saved predictions and write every summary table, report and, optionally, figure of a run."""
    predictions = pd.read_csv(run_dir / "predictions.csv")
    aggregate_scores(predictions).to_csv(run_dir / "scores_by_fold_seed.csv", index=False)
    summary = clustered_summary(predictions, run_dir, cfg.protocol.bootstrap_iterations, cfg.protocol.seed)
    if figures:
        make_figures(summary, read_optional(run_dir / "training_curves.csv"), run_dir)
    resource_and_descriptor_tables(run_dir, summary)
    dataset_tables(run_dir, Path(cfg.data.data_dir))
    research_report(run_dir, cfg, summary)
    write_diagnostics(run_dir)
