# Road-anomaly class-incremental learning

Recognizes normal road, bumps and potholes from smartphone inertial recordings in two stages. Stage 1 learns Normal and one anomaly. Stage 2 adds the other anomaly using a small replay memory and, optionally, confidence-gated distillation. Outer and inner folds are grouped by source recording, and the inner folds pick the learning rate.

This is the modular version of `road_cl_experiments.py`. With the same data and settings it writes the same tables, predictions and checkpoints.

## Running

```
pip install -r requirements.txt
python train.py                                   # full profile
python train.py --config configs/review.yaml      # reviewer run from the notebook
python train.py --config configs/smoke.yaml       # quick check
python train.py --set training.steps=50 --name short
python report.py runs/road_cl
python predict.py --checkpoint runs/road_cl/checkpoints/stratified_gated_kd_Bump_fold0.pt --input trip.csv
python benchmark.py --checkpoint runs/road_cl/checkpoints/stratified_gated_kd_Bump_fold0.pt --threads 1 --repeats 50
```

Put the three RoadSens-4M class files from `configs/default.yaml` in `data/Road Anomalies/`, or pass `--data`. They need `source_file`, `segment_id`, `seconds_elapsed`, `label` and accelerometer, gyroscope and gravity x/y/z. To group by verified trips instead of file names, set `data.session_manifest` to a CSV with `record,session_id`.

Profiles:

- `smoke`: replay ablation only.
- `review`: fine-tuning, EWC, LwF, the replay baselines (random, uniform, reservoir, ER-ACE, Weight Aligning, herding, diverse), fixed and gated KD, joint training, the RMS impact descriptor and augmented training.
- `full`: `review` plus the bag-size, supervision, memory, gate-weight, replay-policy, representation and width sweeps.

Each run goes to `runs/<name>/`. `scores_summary.csv` has the bootstrap intervals, `results_report.md` the main table, `verification.json` the leakage checks and `checkpoints/` the fold-0 models. `benchmark.py` writes `device_latency.csv`.

## Layout

```
roadcl/
├── config.py      config dataclasses, YAML, KEY=VALUE overrides
├── data/          loading, windows, features, grouped folds, replay policies
├── models/        FeatureMLP and log-mean-exp bag pooling
├── losses/        bag loss and fixed or gated distillation
├── training/      suites, trainer, one trial, pipeline with LR selection and checks
├── metrics/       per-trial scores and session bootstrap
├── reporting/     figures, tables, results report
├── inference/     prediction and device benchmark
└── utils/         seeding, hashing, logging, RSS sampling
```

## Raspberry Pi 5 simulation

QEMU has no Pi 5 machine, so `../simulations/` copies the platform instead of the speed. It runs an arm64 Debian bookworm container (the base of 64-bit Raspberry Pi OS) on an Apple-silicon Mac, limited by Docker to 4 cores and 4 GB RAM with no swap. `simulations/pi5.env` holds that profile, and each run saves it with the limits read inside the container. Times are scaled by 3.97, the Geekbench 6 single-core ratio of Apple M3 (3125) to Pi 5 (788), into `*_pi5_est` columns, which are estimates. Container predictions are checked against a host run. `benchmark.py` runs unchanged in the container or on a real Pi 5.
