from __future__ import annotations

import argparse

import pandas as pd

from grfd.inference import predict_frame, report_against_labels
from grfd.utils import resolve_device, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a sensor CSV with a trained GRFD student.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True, help="CSV with the same sensor columns used in training")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--variant", default="replay_lwf", help="joint, naive, replay, lwf, replay_lwf, adaptive_kd_750, ...")
    parser.add_argument("--task", type=int, default=5, help="experience whose checkpoint to load (ignored for joint)")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output", help="where to write predictions as CSV")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    setup_logging()
    frame = pd.read_csv(args.input, low_memory=False)
    frame.columns = frame.columns.str.strip()
    result = predict_frame(frame, args.run_dir, args.fold, args.variant, args.task, resolve_device(args.device))

    if args.label_col in frame.columns:
        classes = [c.removeprefix("prob_") for c in result.columns if c.startswith("prob_")]
        report_against_labels(result.predicted_label, frame[args.label_col], classes)
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"wrote {len(result):,} predictions to {args.output}")
    else:
        print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
