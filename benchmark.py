from __future__ import annotations

import argparse
from pathlib import Path

import torch

from roadcl.config import load_config
from roadcl.inference import benchmark_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Time one recording through a saved checkpoint on this device.")
    parser.add_argument("--checkpoint", required=True, help="e.g. runs/road_cl/checkpoints/stratified_gated_kd_Bump_fold0.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", help="folder with the three per-class CSVs (overrides data.data_dir)")
    parser.add_argument("--record", help="Class/source_file; defaults to the checkpoint's timing record")
    parser.add_argument("--out", type=Path, default=Path("device_measurements"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config, [f"data.data_dir={args.data}"] if args.data else [])
    torch.set_num_threads(args.threads)
    result = benchmark_checkpoint(args.checkpoint, cfg.data, args.out, args.repeats, args.threads, args.record)
    print(f"median {result.ms.median():.3f} ms over {len(result)} runs; wrote {args.out / 'device_latency.csv'}")


if __name__ == "__main__":
    main()
