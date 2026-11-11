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
