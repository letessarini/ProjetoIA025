# ProjetoIA025

This repository contains a flood-segmentation project based on geospatial patches and a U-Net model.

## Project goal

The model receives a patch with multiple spatial channels (such as rainfall, imperviousness, slope, susceptibility, etc.) and predicts a binary mask indicating where flooding is present.

## Current structure

- `nlp_34_treinar_unet.py` — legacy training script used as a reference implementation
- `utils/` — legacy utilities for patch validation, splitting, and visualization
- `pipeline/` — clean, modular structure for a reusable pipeline
  - `pipeline/data_preparation/` — dataset loader and split utilities
  - `pipeline/models/` — future model/training code
  - `pipeline/metrics/` — future evaluation/report code
  - `pipeline/orchestrator/` — orchestration entry point
- `DATASET_UNET/` — dataset folder with manifest and patch files

## Dataset format

The dataset is expected to be a folder containing:

- `manifest.json`
- `patches/*.npz`

Each patch is expected to include:

- `X`: input channels, usually shaped like `(C, H, W)`
- `y`: target flood mask, usually shaped like `(H, W)`
- `valido`: valid-pixel mask, usually shaped like `(H, W)`
- optional metadata: `date`, `precip_val`, `frac_valida`

## Pipeline status

The repository is currently in a transition phase:

- the original project scripts are still available and kept as a reference
- a cleaner modular structure has been created under `pipeline/`
- the low-level dataset loader is now ready to be reused by the upgraded workflow

## Important note

The repo was originally built with hardcoded personal paths. Those have been removed in the new modular structure and replaced with project-relative logic.

## Suggested next step

Use the modules under `pipeline/data_preparation/` as the first reusable block for dataset discovery and splitting, and then build the model/training layer on top of that without disturbing the old scripts.

## GPU benchmark

From the repository root, run in a CUDA-enabled PyTorch environment with
NumPy and `segmentation-models-pytorch` installed. The first ResNet-34 run
downloads its ImageNet weights.

```bash
python benchmark.py --smoke
python benchmark.py
```

Pass `--dataset` followed by the path to the directory containing
`manifest.json` and `patches/` if the dataset is elsewhere. Use
`--batch-size 4` if the GPU needs a smaller batch, and `--output` to select
a different results directory.

The root script uses the same training, evaluation, and logging as
`pipeline.py`. The default configuration (`pipeline/config/exp_mateus.json`)
compares plain U-Net and pretrained ResNet-34 U-Net with masked BCE + Dice, the
same date-and-region splits, and up to 80 epochs. The pipeline selects the
checkpoint and cutoff on validation IoU and evaluates the temporal, spatial,
and combined holdouts with that cutoff. Consecutive dates from one storm can
still cross the date split. `--smoke` uses four patches per split and one
epoch to check the setup.

Share the generated `comparison.csv` and `split.json` from
`results/gpu_benchmark/`. Each model also writes the usual
`run_metadata.json` and best checkpoint. To compare other models, pass
`--models` or edit `benchmark_models` in the pipeline JSON configuration.
