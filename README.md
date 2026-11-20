# ProjetoIA025

Flood segmentation in Porto Alegre using geospatial patches and SFINCS labels.
The active pipeline lives in `pipeline/`. The model factory registers
architectures, the metrics module scores valid pixels, and the orchestrator
runs training and evaluation for both root entry points.

## Setup

Use a CUDA-enabled PyTorch environment for the full benchmark. Install the
pretrained encoder dependency in the same Python environment. The first
ResNet-34 run downloads ImageNet weights.

```bash
python -m pip install segmentation-models-pytorch
```

The dataset directory contains `manifest.json` and `patches/*.npz`. Each patch
contains input tensor `X`, binary target `y`, and validity mask `valido`.
The trainer excludes invalid pixels from losses and metrics. Dataset lookup
supports the default `DATASET_UNET/` layout. Pass `--dataset` to use a directory
elsewhere and `--output` to change the results directory.

## Run the benchmark

```bash
python benchmark.py --smoke
python benchmark.py
```

If needed, use `--batch-size 4` for a smaller GPU or
`--dataset /path/to/DATASET_UNET` for an external dataset. The default
configuration is `pipeline/config/exp_mateus.json`. To run an individual
architecture through the same pipeline, use:

```bash
python pipeline.py --config pipeline/config/exp_mateus.json --model unet_resnet34
```

Share `comparison.csv` and `split.json` from the new benchmark results folder.
Each model also saves its training history and best checkpoint.
Use `--models` or the configuration's `benchmark_models` list to choose
architectures already registered in the pipeline.

## Experiment plan

1. **Smoke test on the target GPU.** One epoch and four patches per split check
   dependency loading, CUDA mixed precision, masked training, checkpoints,
   and the report before committing to a long run.
2. **Matched full-budget U-Net comparison.** Train the plain U-Net and
   ImageNet-pretrained ResNet-34 U-Net on the same date-and-region split,
   with masked BCE + Dice and up to 80 epochs. The CPU pilot suggested a
   temporal gain and a spatial loss for ResNet-34. This comparison tests
   whether that tradeoff persists with the full training set. The pipeline
   selects each cutoff by validation IoU and fixes it for all holdouts.
   Read AP against prevalence and check Recall, F2, and predicted area at
   that cutoff. Interpret temporal, spatial, and combined holdouts separately.
3. **Repeat and verify a promising result.** If a candidate gains on the
   holdout relevant to deployment, repeat the matched comparison with
   `--seed 20261023` and `--seed 20261024`. This checks whether the result
   depends on initialization or batch sampling. Before claiming performance
   on independent storms, group multi-day events and retest. The current
   date split can place consecutive dates from one storm in different sets.
