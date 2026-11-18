from __future__ import annotations

import argparse
from pathlib import Path

from roadcl.config import load_config
from roadcl.reporting import report_run
from roadcl.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild tables and figures from a finished run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.run_dir / "config.yaml")
    report_run(args.run_dir, cfg, figures=not args.no_figures)


if __name__ == "__main__":
    main()
