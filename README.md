# ReliableChange

Implementation of **Reliability-Aware Missing-Modality SAR-Optical Change Detection for Rapid Disaster Damage Mapping**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate synthetic seven-channel images, masks and availability/quality arrays. These are generated image controls, not SAR measurements.

```bash
reliable-change demo --output data/generated.npz --count 96 --config config.json --seed 13
reliable-change prepare --input data/generated.npz --output data/prepared
```

`x[N,7,H,W]`: pre RGB, post RGB and post SAR; `y[N,1,H,W]`: change mask; `available[N,2]`: post optical/SAR availability (at least one present). Optional `quality[N,2,H,W]` holds known corruption quality.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
reliable-change train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
reliable-change evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_alignment`, `no_quality`, `no_mask_consistency`. Reuse the prepared data and training seed:

```bash
reliable-change train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
reliable-change evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
reliable-change plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
