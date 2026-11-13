# Road-anomaly class-incremental learning

Two-stage class-incremental recognition of normal road, bumps and potholes from smartphone inertial recordings. Stage 1 learns Normal and one anomaly; stage 2 adds the other anomaly with a small replay memory and optional confidence-gated distillation. Every split is recording-disjoint.

## Running

```
pip install -r requirements.txt
python train.py                                   # full paper suite, configs/default.yaml
python train.py --config configs/smoke.yaml       # quick check
python train.py --set training.steps=50 --set protocol.seeds=3 --name short
python report.py runs/road_cl                     # rebuild tables and figures
python predict.py --checkpoint runs/road_cl/checkpoints/stratified_bump_first_fold0.pt --input trip.csv
```

`data/Road Anomalies/` must hold the three RoadSens-4M class files named in `configs/default.yaml` (`--data` points elsewhere). Each needs `source_file`, `segment_id`, `seconds_elapsed`, `label` and the accelerometer, gyroscope and gravity x/y/z columns.

A run writes to `runs/<name>/`: `config.yaml`, `train.log`, audits of the inputs, windows and folds, `predictions.csv`, `scores_by_fold_seed.csv`, `scores_summary.csv`, `paired_method_comparisons.csv`, replay, gate and resource tables, `confusion/`, `figures/` and `checkpoints/`.

## Architecture

```
roadcl/
├── config.py          dataclass config, YAML loading, KEY=VALUE overrides
├── data/
│   ├── loading.py     per-class CSVs, duplicate-trace removal, source-group aliases
│   ├── windows.py     90-sample windows, stride 45, never across time gaps
│   ├── features.py    axis, gravity-relative and hybrid statistics; rotated and noisy test views
│   ├── splits.py      StratifiedGroupKFold over source groups
│   └── replay.py      random, impact-stratified, diverse, herding and k-center memories
├── models/mlp.py      FeatureMLP (d→32→3) and log-mean-exp bag pooling
├── losses/            task loss and fixed or entropy-gated bag distillation
├── training/
│   ├── experiments.py main suite and ablations (memory, representation, supervision, gate, width)
│   ├── trainer.py     bag-sampled Adam training with optional replay distillation
│   ├── trial.py       one fold, seed and class order: stage 1, replay, stage 2, evaluation
│   └── pipeline.py    loops over repeats, folds, seeds and orders; writes the run
├── metrics/           macro-F1, forgetting, new-class recall, ECE, Brier; bootstrap and paired tables
├── reporting/         summary tables and figures
├── inference/         per-recording prediction from a checkpoint
└── utils/             seeding, hashing, logging
```

Each trial trains stage 1 on Normal plus the first anomaly, selects the replay memory from those training recordings, then trains stage 2 on the new anomaly together with the memory. `fine_tune` uses no memory and `joint` retrains on all classes as the upper reference. Predictions are pooled per recording and scored on the original and perturbed test views.
