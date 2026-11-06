from __future__ import annotations

import argparse

from grfd.reporting import make_figures, summarize_run
from grfd.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild tables and figures from a finished run directory.")
    parser.add_argument("run_dir")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    setup_logging()
    summarize_run(args.run_dir)
    if not args.no_figures:
        make_figures(args.run_dir)


if __name__ == "__main__":
    main()
