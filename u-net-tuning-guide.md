# U-Net tuning guide

This file documents the architecture parameters that can be edited in the project U-Net and explains how each one affects model behavior and results.

## Scope

The model factory currently supports a configurable U-Net built by `pipeline/models/unet_model.py` and exposed through `pipeline/models/factory.py`.

The most relevant editable parameters are:

- `in_channels`
- `base_channels`
- `depth`
- `padding`
- `use_batch_norm`
- `dropout`
- `alpha`
- `beta`
- `learning_rate`
- `weight_decay`
- `batch_size`
- `epochs`
- `limit`

## Priority ranking for tuning

### 1) `base_channels` — highest impact on capacity and training cost

Type: integer
Default in config: `64`

What it does:
- controls the width of the first encoder block
- scales all subsequent feature maps in the encoder/decoder

Effect on results:
- larger values increase capacity and usually improve segmentation quality when training data is sufficient
- too large causes overfitting, slower training, and higher VRAM usage
- too small can underfit small flood regions and weak boundaries

Recommended range:
- conservative: `32`
- strong default: `64`
- bigger experiments: `96` or `128`

Use this first when you want a bigger or smaller model without changing architecture.

### 2) `depth` — very high impact on representational power

Type: integer
Default in config: `4`

What it does:
- determines how many encoder/decoder stages the model has
- increases receptive field and abstraction level

Effect on results:
- deeper networks can capture larger spatial structure and long-range context
- too deep can hurt convergence and overfit, especially with limited data
- deeper models are more expensive and require more memory

Recommended range:
- small dataset: `3` to `4`
- regular compromise: `4`
- stronger but heavier: `5`

This is one of the most important architectural knobs after width.

### 3) `dropout` — important for regularization

Type: float
Default in config: `0.1`

What it does:
- randomly deactivates activations during training to reduce overfitting

Effect on results:
- too low: model may overfit and memorize patch patterns
- too high: model may underfit and become unstable
- helpful when the dataset is modest and many patches are similar

Recommended range:
- low-data regime: `0.1` to `0.2`
- very small dataset: `0.2` to `0.3`
- disable only if overfitting is not an issue

### 4) `learning_rate` — critical for convergence quality

Type: float
Default in `TrainingConfig`: `3e-4`

What it does:
- controls optimizer step size

Effect on results:
- too high: unstable training, loss spikes, poor convergence
- too low: slow learning, under-trained model
- strongly affects how fast the U-Net learns to detect small flood regions

Recommended range:
- standard: `1e-4` to `3e-4`
- aggressive: `5e-4` with careful monitoring

This is usually the first parameter to tune after model size.

### 5) `batch_size` — affects optimization stability and memory

Type: integer
Default in config: `4`

What it does:
- controls how many patches are processed in one training step

Effect on results:
- larger batches are more stable but less noisy
- smaller batches can generalize better in some tasks but are slower
- too large may reduce effective learning quality on small datasets

Recommended range:
- small GPU: `2` to `4`
- mid GPU: `4` to `8`
- larger GPU: `8` to `16`

### 6) `use_batch_norm` — medium impact

Type: boolean
Default in config: `true`

What it does:
- normalizes intermediate feature maps per batch

Effect on results:
- usually more stable training and faster convergence
- can be less effective with very small batches or highly variable data
- may hurt if batch size is too small or data distribution shifts strongly

Recommended use:
- keep `true` unless you are intentionally testing a lightweight variant

### 7) `padding` — low to medium impact

Type: boolean
Default in config: `true`

What it does:
- preserves spatial size through convolutions

Effect on results:
- keeps segmentation output aligned with input patch size
- important for patch-level segmentation consistency
- disabling it can reduce output alignment and create border artifacts

Recommendation:
- keep `true` unless you want a deliberately different architecture

### 8) `epochs` — training duration control

Type: integer
Default in config: `3`

What it does:
- determines how many full passes over the training set occur

Effect on results:
- more epochs usually help if the model is still learning
- too many can overfit or waste compute
- with sparse flood labels, early stopping is often useful

Recommended range:
- smoke tests: `1` to `5`
- real training: `20` to `50`
- very small data: monitor validation loss carefully

### 9) `limit` — influences dataset exposure, not architecture

Type: integer or null
Default in config historically: `8`

What it does:
- limits how many patches are loaded and used in training

Effect on results:
- lower values create smoke tests and unstable comparison runs
- higher values give better coverage of the dataset, but longer runtime
- this is often the biggest reason training appears weak or inconsistent

Recommendation:
- for quick tests: `8`, `20`, or `50`
- for a real run: omit it or set it to `None` to use the full dataset

This is not an architecture parameter, but it can dominate the apparent model quality.

### 10) `in_channels` — data contract parameter

Type: integer
Default in config: `10`

What it does:
- sets how many feature channels the model expects as input

Effect on results:
- must match the dataset patch channel count
- wrong value will break the forward pass or distort representation

Recommendation:
- keep it equal to the actual dataset channel count
- for this project, it should remain `10` unless the dataset changes

### 11) `alpha` and `beta` — loss function weighting

Type: float
Default in config: `alpha=0.3`, `beta=0.7`

What they do:
- control Tversky loss weighting between false positives and false negatives

Effect on results:
- `alpha` increases penalty on false positives
- `beta` increases penalty on false negatives
- useful for class imbalance and flood detection where missing positives is costly

Recommended range:
- `alpha=0.3`, `beta=0.7` is a sensible default
- if false negatives are worse, increase `beta`
- if false positives are worse, increase `alpha`

### 12) `weight_decay` — secondary regularizer

Type: float
Default in `TrainingConfig`: `1e-5`

What it does:
- regularizes weights and reduces overfitting

Effect on results:
- small values are usually safe
- too large may underfit and slow learning

Recommended range:
- `1e-5` to `1e-4`

## Practical tuning order

If you want a sensible tuning workflow, use this order:

1. `limit` — ensure you are training on the intended data size
2. `base_channels`
3. `depth`
4. `learning_rate`
5. `dropout`
6. `batch_size`
7. `alpha` and `beta`
8. `weight_decay`
9. `epochs`

## Current project defaults

The current config file uses:

- `model`: `unet`
- `epochs`: `3`
- `batch_size`: `4`
- `limit`: `8`
- `base_channels`: `64`
- `depth`: `4`
- `padding`: `true`
- `use_batch_norm`: `true`
- `dropout`: `0.1`
- `alpha`: `0.3`
- `beta`: `0.7`

These are a reasonable baseline, but for serious experiments you should usually:

- increase `epochs`
- avoid `limit=8` for final comparisons
- increase `base_channels` or `depth` if training data is enough
- keep `dropout` in the `0.1` to `0.2` range unless you are intentionally testing a more aggressive or stable model

## Good starting points

### Small/fast experiment
- `base_channels=32`
- `depth=4`
- `dropout=0.1`
- `batch_size=4`
- `learning_rate=1e-4`
- `epochs=10`
- `limit=50`

### Strong baseline
- `base_channels=64`
- `depth=4`
- `dropout=0.1`
- `batch_size=4`
- `learning_rate=3e-4`
- `epochs=30`
- `limit=None` or full dataset

### Larger model
- `base_channels=64`
- `depth=5`
- `dropout=0.1`
- `batch_size=4`
- `learning_rate=1e-4`
- `epochs=30`
- `limit=None`

## Summary

The most important knobs are:

1. `base_channels`
2. `depth`
3. `learning_rate`
4. `dropout`
5. `limit`

The best way to improve results in this project is usually to:

- first remove or raise `limit`
- then increase width/depth if the model is underfitting
- then tune learning rate and regularization
- finally tune the loss weights and training duration
