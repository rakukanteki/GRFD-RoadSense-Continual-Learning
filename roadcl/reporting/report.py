from __future__ import annotations

from pathlib import Path

import pandas as pd

from roadcl.reporting.figures import make_figures
from roadcl.reporting.tables import build_tables


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def report_run(run_dir: Path, iterations: int, seed: int, figures: bool = True) -> None:
    """Write every summary table of a finished run and, optionally, its figures."""
    scores, summary = build_tables(run_dir, iterations, seed)
    if figures:
        make_figures(scores, summary, read_optional(run_dir / "training_curves.csv"),
                     read_optional(run_dir / "gate_diagnostics.csv"), run_dir / "figures")
