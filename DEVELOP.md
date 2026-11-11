# Development notes

## Workflow rule

- Important pending work belongs in `to-do.md`.
- Completed work, validated decisions, and design notes belong here in `DEVELOP.md`.
- This repository keeps both active and historical context simultaneously without deleting the original reference implementation.

## Current pipeline goal

This repository is being refactored into a modular pipeline for flood segmentation using U-Net-style models with geospatial patch data.

## Data contract

The expected dataset structure is:

- `DATASET_UNET/manifest.json`
- `DATASET_UNET/patches/*.npz`

Each patch is expected to contain:

- `X`: feature channels with shape `(C, H, W)`
- `y`: flood mask with shape `(H, W)`
- `valido`: valid-pixel mask with shape `(H, W)`
- optional metadata such as `date`, `precip_val`, and `frac_valida`

In this project, the actual dataset is organized as spatial tile patches named with coordinate metadata, such as `20130107_y0000_x0256.npz`, so reconstruction must be performed per date and per tile coordinate rather than by merging all patches indiscriminately.

## Current modular layout

- `pipeline/data_preparation/dataset_loader.py`
  - locates the dataset and loads metadata
- `pipeline/data_preparation/dataset_splitter.py`
  - builds train/val/test splits while preventing leakage
- `pipeline/data_preparation/prepare_dataset.py`
  - validates and preprocesses patches for training
- `pipeline/models/`
  - model definitions and training logic
- `pipeline/metrics/`
  - evaluation and reporting
- `pipeline/orchestrator/`
  - final end-to-end runner

## Preprocessing policy

The project already stores preprocessing metadata inside `manifest.json`:

- `preenchimento_nodata`
- `normalizacao`

The preprocessing layer should respect these existing rules before training. The current implementation does two things:

1. fill missing values with the dataset-specified default values
2. normalize channels using the range declared in the manifest

Optional augmentation can be enabled during training or patch inspection, but should remain explicit and not be forced on all runs.

## Model architecture and tunable knobs

The current U-Net is configurable and exposes the main editable architecture parameters in `pipeline/models/unet_model.py`:

- `in_channels`
- `base_channels`
- `depth`
- `padding`
- `use_batch_norm`
- `dropout`
- `alpha`
- `beta`

The default runtime config in `pipeline/config/config.json` is:

- `model`: `unet`
- `epochs`: `3`
- `batch_size`: `4`
- `seed`: `20261022`
- `output_dir`: `results`
- `checkpoint_name`: `last_model.pt`
- `model_config`: `in_channels=10`, `base_channels=64`, `depth=4`, `padding=true`, `use_batch_norm=true`, `dropout=0.1`, `alpha=0.3`, `beta=0.7`

The model name is normalized in `pipeline/models/factory.py`, so `unet`, `u-net`, and `u_net` resolve to the same architecture.

## Training runtime policy

- `limit` is a runtime data-subset control, not a model architecture parameter.
- A missing or negative `limit` means use the full dataset.
- CLI arguments override config values when explicitly provided.
- The default behavior is to load the config file and apply CLI overrides only when the user passes them.

## Results and output handling

The pipeline saves metrics, checkpoints, and prediction outputs under the repo `results/` directory and includes a timestamped run folder with metadata.

This is currently used for training runs and prediction export, with the metadata stored in `run_metadata.json`.

## Notebook and reconstruction understanding

The result notebook and data analysis work clarified that:

- targets are sparse, so many tiles appear mostly black in raw plots
- `valido` is the valid-pixel mask and should be respected in loss and metrics
- patches are tile-based and must be reconstructed by date and by spatial coordinate
- full-scene reconstruction is not done by concatenating all patches from all dates into one array

## Loss design and imbalance handling

The training objective is now configurable and is no longer hard-coded to a single binary cross-entropy setup.

Current behavior:

- default loss: `weighted_bce`
- alternative options: `bce`, `weighted_bce`, `focal`
- CLI usage: `--loss weighted_bce` and `--loss-config '{"pos_weight": 3.0}'`
- config file entry: `loss_name` and `loss_config`

Why this matters:

- the flood masks are typically sparse and strongly imbalanced
- when most pixels are background, plain BCE can push the model toward trivial predictions that maximize the average loss without learning the rare foreground class
- in segmentation tasks this often appears as near-perfect recall with very low precision and a flat F1 score

The implemented loss options are:

1. `bce` / `binary_cross_entropy`
   - standard masked binary cross-entropy
   - good baseline, but does not compensate for rare positive pixels

2. `weighted_bce`
   - masks invalid pixels and reweights positive examples via `pos_weight`
   - recommended default for class imbalance in this project
   - helpful when foreground pixels are much rarer than background pixels
   - the weight can be fixed manually or estimated from the dataset prevalence

3. `tversky`
   - uses the Tversky index, which balances TP, FP, and FN explicitly
   - a good fit for sparse segmentation and flood masks
   - beta > alpha makes false negatives more expensive than false positives
   - this aligns with the design of the original legacy workflow, which already used alpha/beta values of 0.3 and 0.7

4. `focal`
   - down-weights easy examples and emphasizes hard-to-classify pixels
   - useful when the foreground is extremely sparse and difficult to detect
   - more aggressive than weighted BCE and often needs more careful tuning

The loss is selected in the trainer via `build_loss(...)`, and the config/CLI override path is used so the same training runner can compare objective choices without code changes.

## Implemented evaluation metrics

The pipeline now computes and reports epoch-level segmentation metrics after training and validation, using the valid-pixel mask to ignore invalid areas.

The implemented metrics are:

- IoU (Intersection over Union)
  - measures how much the predicted positive region overlaps the true positive region
  - sensitive to both false positives and false negatives
  - useful as the main segmentation overlap score

- Precision
  - `TP / (TP + FP)`
  - tells us how many predicted positives are actually correct
  - low precision means the model is predicting too many false alarms

- Recall
  - `TP / (TP + FN)`
  - tells us how many true positives were recovered
  - high recall with low precision is a classic sign of class imbalance or a collapsed prediction rule

- F1 score
  - harmonic mean of precision and recall
  - balances false positives and false negatives
  - useful as the threshold-selection target when the model is not perfectly calibrated

- Accuracy
  - overall fraction of correct pixels among all valid pixels
  - can be misleading when the foreground is sparse, because background dominates the score

- Prevalence / predicted fraction metrics
  - help diagnose whether the model is predicting mostly background or mostly foreground
  - useful when interpreting collapsing metrics during early training

The training loop also runs a threshold sweep across values such as `0.3`, `0.5`, and `0.7`, and selects the threshold that maximizes F1 on the validation set. This is useful because a model can look better or worse depending on how the probability map is binarized.

This combination of imbalance-aware loss and threshold-aware evaluation is now part of the standard training flow and directly supports debugging model collapse during early epochs.

## Completed project phases

### Phase 1 - organization and portability
- [x] Create the `pipeline/` folder structure and modular entry points
- [x] Keep the original scripts as reference and fallback
- [x] Remove machine-specific path assumptions and use repo-relative logic
- [x] Add portable dataset discovery and patch validation

### Phase 2 - configuration and model generalization
- [x] Add a JSON config-driven runner
- [x] Add CLI override support for `--config`, `--model`, `--epochs`, `--batch-size`, `--limit`, and `--seed`
- [x] Generalize the U-Net architecture using editable knobs
- [x] Add automatic CUDA detection with CPU fallback

### Phase 3 - training and result output
- [x] Prepare patch datasets and keep a smoke-test path working
- [x] Create a results folder structure with timestamped run metadata
- [x] Save predictions, checkpoints, and model config metadata
- [x] Build notebook-based visual inspection for prediction vs target vs valid mask

## Recommended future steps

1. Keep the legacy `nlp_34_treinar_unet.py` script as the reference implementation.
2. Move training logic to `pipeline/models/` once the preparation pipeline is stable.
3. Add evaluation logic under `pipeline/metrics/`.
4. Create the final orchestrator script that calls the stages in order.
5. Add a small test notebook to inspect augmentation and batch loading.
6. Run a full-data benchmark and compare model variants using the tuned U-Net parameters.

## Important note

The project was originally built with hardcoded user-specific paths. The new pipeline uses repo-relative paths and should work when the repository is cloned and the dataset is downloaded into the expected folder structure.

## Tuning guide

A dedicated tuning guide for the U-Net parameters is available in `u-net-tuning-guide.md`.

The primary knobs to experiment with are:

1. `base_channels`
2. `depth`
3. `learning_rate`
4. `dropout`
5. `limit`

The main recommendation is to first ensure that `limit` is set to a meaningful value, then tune the model capacity and regularization before changing the loss weights.
