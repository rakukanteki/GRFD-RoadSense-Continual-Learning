# GRFD continual learning

Gated Random Forest distillation for road-condition classification (normal, bump, pothole) from smartphone sensor data, trained jointly and over sequential experiences.

## Architecture

```
GRFD-Continual-Learning/
├── configs/
│   ├── default.yaml        full experiment
│   └── quick.yaml          small smoke test
├── data/                   all_labeled_data.csv
├── grfd/
│   ├── config.py           dataclass config, YAML loading, CLI overrides
│   ├── data/               CSV loading, per-fold preprocessing, splits, replay buffer
│   ├── models/             gated student, RF teacher, OOB reliability calibrator
│   ├── losses/             softened KL, weighted KL, combined objective
│   ├── metrics/            classification, continual (AIA/AF/BWT/FWT/FAA), size and latency
│   ├── training/           trainer loop, joint run, baselines, continual variants, CV driver
│   ├── reporting/          summary tables and figures
│   ├── inference/          scoring new CSVs with a saved fold
│   └── utils/              seeding, device, logging
├── train.py
├── report.py
├── predict.py
└── requirements.txt
```

The student is a shared encoder feeding three expert heads mixed by a softmax gate, plus a linear skip head. It is trained with

```
(1 - kd_weight - lwf_weight) * CE + kd_weight * KL(RF teacher) + lwf_weight * KL(previous student)
```

Continual variants are set in `configs/default.yaml` under `continual.variants`:

| variant | replay | LwF | adaptive KD |
|---|---|---|---|
| naive | 0 | no | no |
| replay | 750 | no | no |
| lwf | 0 | yes | no |
| replay_lwf | 750 | yes | no |
| fixed_kd_750 | 750 | no | no |
| adaptive_kd_750 | 750 | no | yes |
| adaptive_kd_0 | 0 | no | yes |
| adaptive_kd_3000 | 3000 | no | yes |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ln -s /path/to/all_labeled_data.csv data/all_labeled_data.csv
```

## Run

Smoke test:

```bash
python train.py --config configs/quick.yaml
```

Full experiment:

```bash
python train.py --config configs/default.yaml
```

Override any config field:

```bash
python train.py --data /path/to/all_labeled_data.csv --name my_run \
    --set training.epochs=10 --set "continual.run=[naive,replay_lwf]"
```

Rebuild tables and figures from a finished run:

```bash
python report.py runs/grfd_cv_continual
```

Predict on a new CSV:

```bash
python predict.py --run-dir runs/grfd_cv_continual --input new_trip.csv \
    --fold 1 --variant replay_lwf --task 5 --output predictions.csv
```

## Outputs

```
runs/<name>/
├── config.yaml, train.log
├── *.csv                   metrics, continual, efficiency, C1/C3 tables
├── figures/
├── predictions/            last-fold test probabilities
└── fold_XX/                preprocessor, splits, student and teacher checkpoints per variant
```
