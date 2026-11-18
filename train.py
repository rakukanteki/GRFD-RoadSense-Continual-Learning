from __future__ import annotations

import argparse

from roadcl.config import load_config
from roadcl.reporting import report_run
from roadcl.training import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stage class-incremental road-anomaly benchmark.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", help="folder with the three per-class CSVs (overrides data.data_dir)")
    parser.add_argument("--output", help="parent directory for runs (overrides experiment.output_dir)")
    parser.add_argument("--name", help="run name (overrides experiment.name)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config field, e.g. --set training.steps=50")
    parser.add_argument("--no-report", action="store_true", help="skip summary tables and figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    for value, key in ((args.data, "data.data_dir"), (args.output, "experiment.output_dir"),
                       (args.name, "experiment.name")):
        if value:
            overrides.append(f"{key}={value}")

    cfg = load_config(args.config, overrides)
    run_dir = run_benchmark(cfg)
    if not args.no_report:
        report_run(run_dir, cfg)


if __name__ == "__main__":
    main()
