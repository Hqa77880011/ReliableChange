# Reliability-Aware Missing-Modality SAR-Optical Change Detection for Rapid Disaster Damage Mapping

Implementation of **Reliability-Aware Missing-Modality SAR-Optical Change Detection for Rapid Disaster Damage Mapping**.

Optical-before, optical-after, and SAR-after branches share a projected feature space. Bounded local correlation aligns each available post-event branch, and availability-normalized quality weights fuse temporal differences. Region, boundary, unchanged/changed feature separation, full-to-masked consistency, and optional corruption-quality supervision use only acquired modalities. The region objective uses BCE plus the Lovasz binary IoU surrogate.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,7,H,W]` concatenates pre-event RGB, post-event RGB, and post-event SAR. `y[N,1,H,W]` is change ground truth; `available[N,2]` indicates post optical/SAR presence and cannot be both zero. Optional `quality[N,2,H,W]` supplies known synthetic corruption maps. Use geo-disjoint paired SEN12-type/SAR-optical data with an explicit common registration and label definition.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
reliable-change demo --output data/example.npz --count 96 --config config.json
reliable-change prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
reliable-change train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
reliable-change evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
reliable-change plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
reliable-change calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_alignment`, `no_quality`, `no_mask_consistency`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
reliable-change train --input data/prepared --output runs/baseline --config config.json --variant baseline
reliable-change evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Perceptual and saliency endpoints

```bash
python -m pip install -e ".[image-metrics]"
python -m reliable_change.evaluate_images --predictions runs/full/evaluation/predictions.npz --reference data/prepared/test.npz --metric saliency --output runs/additional_metrics.json
```

LPIPS uses the [published perceptual metric implementation](https://github.com/richzhang/PerceptualSimilarity) with RGB values mapped to [-1,1]. Saliency endpoints use [PySODMetrics](https://github.com/lartpang/PySODMetrics) for adaptive/maximum F-measure, E-measure, S-measure, and weighted F-measure; their threshold conventions remain those of the metric package.
