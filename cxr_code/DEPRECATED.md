# cxr_code — Deprecated Legacy Module

This directory contains the original CheXNet-style training code for the NIH ChestX-ray14 dataset. It has been superseded by the scripts in `scripts/` and the models in `model/`.

## Why it is preserved

**The CSV files in `cxr_code/data/` are the authoritative dataset splits** used by the active training pipeline:

| File | Purpose |
|------|---------|
| `data/train.csv` | NIH ChestX-ray14 training split (~98,637 samples) |
| `data/valid.csv` | NIH ChestX-ray14 validation split (~6,350 samples) |
| `data/train_relabeled.csv` | Relabeled training variant |
| `data/valid_relabeled.csv` | Relabeled validation variant |

`scripts/train_nih.py` reads `cxr_code/data/train.csv` and `cxr_code/data/valid.csv` directly.  
**Do not delete this directory.**

## What replaced it

| Legacy | Replacement |
|--------|------------|
| `cxr_code/train.py` | `scripts/train_nih.py` (GPU-optimized, AMP, reproducible) |
| `cxr_code/predict.py` | `scripts/predict_nih.py` |
| `cxr_code/densenet.py` | `torchvision.models.densenet121` (used via `model/image_only_model.py`) |
| `cxr_code/util.py` | `utils/nih_dataset.py`, `utils/train.py` |
| `cxr_code/get_best_model.py` | Checkpoint saved automatically in `scripts/train_nih.py` |

## Known issues in legacy code (fixed)

- `util.py`: `preds` NameError in closure — fixed
- `util.py`: deprecated `.as_matrix()` — fixed to `.values`
- `train.py`: deprecated `.data[0]` — fixed to `.item()`
- `train.py`: deprecated `volatile` kwarg — removed
- `predict.py`: unclosed file handle — fixed with context manager
