from __future__ import annotations

import argparse

import pandas as pd

from roadcl.inference import predict_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify recordings in a sensor CSV with a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="e.g. runs/road_cl/checkpoints/stratified_bump_first_fold0.pt")
    parser.add_argument("--input", required=True, help="CSV with seconds_elapsed and the nine sensor columns")
    parser.add_argument("--group-col", default="source_file", help="column that identifies a recording")
    parser.add_argument("--output", help="where to write one row per recording")
    args = parser.parse_args()

    result = predict_frame(pd.read_csv(args.input), args.checkpoint, args.group_col)
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"wrote {len(result)} recording predictions to {args.output}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
