# Road-anomaly class-incremental learning

Two-stage class-incremental recognition of normal road, bumps and potholes from smartphone inertial recordings. Stage 1 learns Normal and one anomaly; stage 2 adds the other anomaly with a small replay memory and optional confidence-gated distillation. Every outer and inner split is grouped by source recording, and each outer fold, method and arrival order selects its learning rate by inner grouped cross-validation.

This package is the modular form of `road_cl_experiments.py` (the script behind `TVT_Continual_Learning_Reviewer_Experiments.ipynb`). On the same data and settings it writes the same tables, predictions and checkpoints.

## Running

```
pip install -r requirements.txt
python train.py                                   # full profile: review comparisons plus every sweep
python train.py --config configs/review.yaml      # the reviewer run in the notebook (5 folds, 2 seeds, 2000 bootstrap draws)
python train.py --config configs/smoke.yaml       # quick check: replay ablation only
python train.py --set training.steps=50 --set protocol.seeds=3 --name short
python report.py runs/road_cl                     # rebuild tables, figures and the results report
python predict.py --checkpoint runs/road_cl/checkpoints/stratified_gated_kd_Bump_fold0.pt --input trip.csv
python benchmark.py --checkpoint runs/road_cl/checkpoints/stratified_gated_kd_Bump_fold0.pt --threads 1 --repeats 50
```

`data/Road Anomalies/` must hold the three RoadSens-4M class files named in `configs/default.yaml` (`--data` points elsewhere). Each needs `source_file`, `segment_id`, `seconds_elapsed`, `label` and the accelerometer, gyroscope and gravity x/y/z columns. A verified trip mapping (`record,session_id`) can replace the filename session proxy through `data.session_manifest`.

Profiles: `smoke` runs the four-way replay ablation; `review` runs the main baselines (fine-tuning, EWC, LwF, random, uniform random, reservoir, ER-ACE, Weight Aligning, herding, diverse, fixed and gated KD, joint), the ablation, the RMS impact descriptor and augmented training; `full` adds the bag-size, supervision, memory, gate-weight, replay-policy, representation and width sweeps.

A run writes to `runs/<name>/`: `config.yaml`, `train.log`, input, trace, recording, source and fold audits, `inner_assignments.csv` and `inner_selection.csv`, `predictions.csv`, `scores_by_fold_seed.csv`, `scores_summary.csv` (session-bootstrap intervals), `paired_method_comparisons.csv`, `confusion_matrices.csv`, replay, gate and resource tables, `baseline_*.svg`, `replay_ablation_*.svg`, `robustness_*.svg`, `results_report.md`, `verification.json` and `checkpoints/`. `benchmark.py` writes `device_latency.csv`.

## Architecture

```
roadcl/
├── config.py          dataclass config, YAML loading, KEY=VALUE overrides
├── data/
│   ├── loading.py     per-class CSVs, duplicate-trace removal, source-group aliases, session manifest
│   ├── windows.py     90-sample windows, stride 45, never across time gaps; peak and RMS impact
│   ├── features.py    axis, gravity-relative and hybrid statistics; rotated, noisy, gyro-free and augmented views
│   ├── splits.py      class-complete StratifiedGroupKFold over source groups (outer and inner)
│   └── replay.py      class-quota, uniform, reservoir, recording-only, impact-only, stratified, diverse, herding, k-center
├── models/mlp.py      FeatureMLP (d→32→3) and log-mean-exp bag pooling
├── losses/            task loss and fixed or entropy-gated bag distillation
├── training/
│   ├── experiments.py smoke, review and full suites
│   ├── trainer.py     bag-sampled Adam with replay KD, LwF, EWC, augmentation or ER-ACE masking
│   ├── trial.py       one fold, seed and order: stage 1, replay, stage 2, evaluation, timing, checkpoint
│   └── pipeline.py    nested learning-rate selection, outer fits, RAM sampling, output verification
├── metrics/           per-trial scores; pooled recording metrics with session-clustered bootstrap and paired draws
├── reporting/         figures, dataset and resource tables, diagnostics, results report
├── inference/         per-recording prediction and on-device latency benchmark
└── utils/             seeding, hashing, logging, RSS sampling
```

## Raspberry Pi 5 simulation

`../simulations/` runs the saved checkpoints in a Raspberry Pi 5 profile. No public emulator reproduces Pi 5 speed (QEMU has no Pi 5 machine), so we match the platform instead: a native arm64 Debian bookworm container (the base of 64-bit Raspberry Pi OS) on an Apple-silicon host, pinned by Docker cgroups to 4 cores and 4 GB RAM with no swap. The profile is in `simulations/pi5.env`, and each run records it and the limits measured inside the container in `pi5_config_used.json` and `environment.csv`. The container cannot slow the CPU down, so every timing CSV also has a `*_pi5_est` column: the measured time times 3.97, the Geekbench 6 single-core ratio of Apple M3 (3125) to Pi 5 (788). Those columns are estimates; a run on a real Pi 5 replaces them. A correctness gate checks that the container's recording predictions match a host run. `benchmark.py` runs unchanged in the same container or on a real Pi 5 (see `simulations/README.md`).
