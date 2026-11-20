# GPU benchmark

Run the smoke test first. The full comparison uses the same ten inputs and
date-and-region splits for all five models. Read the temporal, spatial, and
combined holdouts separately.

```bash
python -m pip install "segmentation-models-pytorch==0.6.0" "nnunetv2==2.6.4"
python benchmark.py --smoke
python benchmark.py
```

Use a CUDA-enabled PyTorch environment. The first pretrained run downloads
ImageNet weights. Pass `--dataset` and `--output` for custom locations or
`--models` to select a subset. `--batch-size` controls the native models.
nnU-Net derives its own batch size from the plans.

| Model | Architecture and parameters | Why it is in the benchmark |
| --- | --- | --- |
| `unet` | Three downsampling stages with widths 16, 32, 64 and a 128-channel bottleneck. Padded 3×3 convolutions, BatchNorm, ReLU, no dropout, bilinear upsampling, and matching-scale concatenation. One 1×1 logit head. | Small control for temporal and spatial generalization. |
| `unetpp` | Same encoder widths, depth, convolutions, activations, and logit head as the small U-Net. Nested dense skips concatenate intermediate features across scales. The final head supplies the loss. | Tests whether nested skips help recover small flooded pockets with the compact encoder. |
| `unet_resnet34` | Five-stage ImageNet-pretrained ResNet-34 with a 7?7 stride-2 stem adapted to ten inputs. Residual ReLU encoder, decoder widths 256, 128, 64, 32, 16, BatchNorm, ReLU, nearest-neighbor upsampling, and matching-scale concatenation. One logit head. | Pretrained control. The short pilot improved temporal IoU and lost spatial IoU. The full run checks that tradeoff. |
| `unetpp_resnet34` | Same pretrained encoder, decoder width schedule, normalization, activations, and upsampling as the ResNet-34 U-Net. Dense nested decoder skips and one final logit head. | Tests the decoder change at fixed encoder and training recipe. Look for better AP and Recall at comparable Precision. |
| `nnunet_2d` | Standard 2D PlainConvUNet with encoder-decoder skips, InstanceNorm, LeakyReLU, deep supervision, and two class logits. The training-only plans set stage widths, kernels, strides, patch size, and batch size. The actual plans are saved with the run. | Tests a biomedical segmentation pipeline end to end. Differences also reflect its preprocessing, loss, sampling, and schedule. |

All five models see 256?256 patches with nine physical bands and a binary validity band. In every
model, invalid pixels are excluded from loss and reported metrics. Native
models use **masked BCE + Dice** with equal weights. BCE takes logits and
sigmoid supplies probabilities for Dice and evaluation. The shared recipe uses
AdamW at 0.0003, weight decay 0.00001, cosine decay, batch size 8, a 70% target
fraction of wet patches, and up to 80 epochs with patience 15. The paired controls share these settings
to isolate the change in skip connections as closely as possible.

nnU-Net applies **Dice + cross-entropy** with the upstream ignore label and
deep supervision. It uses Nesterov SGD starting at 0.01, polynomial decay,
foreground oversampling, and augmentation for the configured epoch budget,
with 250 training iterations per full epoch.
The nine physical bands use nnU-Net
normalization and the binary validity band uses no normalization. Planning
uses training cases only. The smoke plan used seven stages of width 32, 64,
128, 256, 512, 512, 512, with two 3?3 convolutions per stage. Fold 0 uses
the shared validation IDs for checkpoint
selection by nnU-Net's foreground Dice proxy. Native models select checkpoints
by validation IoU. At evaluation, all models select their probability cutoff
by validation IoU and apply it to every test holdout. nnU-Net exports flood
softmax probabilities and uses inference without mirroring.

Compare AP against prevalence, and compare F2, Recall, Precision, and flagged
area at the fixed cutoff. Repeat promising runs with two additional seeds.
Group multi-day storms before making claims about independent events.

Architecture references [U-Net++](https://arxiv.org/abs/1912.05074) and
[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z).
