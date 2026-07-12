# TeraCyte Live/Dead Cell Classification

Binary classifier for brightfield microscopy images (live vs dead).

## Setup

```bash
pip install -r requirements.txt
```

Dataset layout (from assignment):

```
images/
  metadata.csv
  images/*.png
split.csv
```

## Environment

Works locally and on a single cloud GPU (tested on a Nebius L4/L40S-class VM). Set the project root if needed:

```bash
export DATA_ROOT=/path/to/TeraCyte_assignment
```

## Train

```bash
python train.py --epochs 30 --batch-size 512 --num-workers 12 --output-dir checkpoints
```

The larger batch size keeps the GPU well utilized while fitting comfortably in memory. Options: `--batch-size`, `--num-workers`, `--lr 1e-4`, `--data-root .`

Outputs:
- `checkpoints/best.pt` — best model by validation F1
- `checkpoints/history.json` — per-epoch metrics for plotting

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --split test
```

Outputs:
- Prints overall, per-experiment, and per-assay metrics
- Saves `checkpoints/eval_results.json`

## Notebook (recommended on Nebius GPU VM)

```bash
pip install -r requirements.txt
export DATA_ROOT=$(pwd)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

SSH tunnel from your machine: `ssh -L 8888:localhost:8888 itiel@<VM_IP>`

Part 3 streams live logs from `train.py` / `evaluate.py`, plots GPU usage from `history.json`, and reports PR-AUC + inference throughput on the test set.

## Notes

- Class imbalance handled via weighted cross-entropy (computed from train split)
- Noisy samples (`split=noisy`) excluded from training and evaluation
- Primary metrics: macro F1, balanced accuracy, per-assay breakdown
