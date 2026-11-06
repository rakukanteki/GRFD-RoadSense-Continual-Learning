from __future__ import annotations

import argparse

from grfd.config import load_config
from grfd.reporting import make_figures, summarize_run
from grfd.training import run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRFD students with cross-validation and continual experiences.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", help="path to all_labeled_data.csv (overrides data.csv_path)")
    parser.add_argument("--output", help="parent directory for runs (overrides experiment.output_dir)")
    parser.add_argument("--name", help="run name (overrides experiment.name)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config field, e.g. --set training.epochs=5")
    parser.add_argument("--no-report", action="store_true", help="skip summary tables and figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if args.data:
        overrides.append(f"data.csv_path={args.data}")
    if args.output:
        overrides.append(f"experiment.output_dir={args.output}")
    if args.name:
        overrides.append(f"experiment.name={args.name}")

    cfg = load_config(args.config, overrides)
    run_dir = run_cross_validation(cfg)
    if not args.no_report:
        summarize_run(run_dir)
        make_figures(run_dir)


if __name__ == "__main__":
    main()
