from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipeline.models.activations import DEFAULT_HIDDEN_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION, build_activation

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency during import
    torch = None
    nn = None


@dataclass
class ModelConfig:
    in_channels: int = 10
    base_channels: int = 32
    depth: int = 4
    padding: bool = True
    use_batch_norm: bool = True
    dropout: float = 0.0
    alpha: float = 0.3
    beta: float = 0.7
    activation: str = DEFAULT_HIDDEN_ACTIVATION
    output_activation: str = DEFAULT_OUTPUT_ACTIVATION
    decoder_channels: Optional[list[int]] = None


def _conv_block(
    in_ch: int,
    out_ch: int,
    padding: bool = True,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    activation: str = DEFAULT_HIDDEN_ACTIVATION,
):
    if torch is None or nn is None:
        raise ModuleNotFoundError("PyTorch is required to build the U-Net model.")

    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1 if padding else 0, bias=not use_batch_norm),
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(build_activation(activation))
    layers.extend([
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1 if padding else 0, bias=not use_batch_norm),
    ])
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(build_activation(activation))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def build_unet(
    in_channels: int,
    base_channels: int = 32,
    depth: int = 4,
    padding: bool = True,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    activation: str = DEFAULT_HIDDEN_ACTIVATION,
    output_activation: str = DEFAULT_OUTPUT_ACTIVATION,
    decoder_channels: Optional[list[int]] = None,
    **kwargs,
):
    """Build a configurable U-Net with standard skip connections.

    decoder_channels is the list of output widths for the decoder blocks after each
    skip concatenation. It should have length depth - 1. For example, depth=4 uses
    decoder_channels=[128, 64, 32].
    """
    if torch is None or nn is None:
        raise ModuleNotFoundError("PyTorch is required to build the U-Net model.")

    depth = max(2, int(depth))
    base_channels = max(8, int(base_channels))

    if decoder_channels is not None:
        decoder_channels = [max(1, int(ch)) for ch in list(decoder_channels)]
        if len(decoder_channels) != depth:
            raise ValueError(f"decoder_channels must have length {depth}, got {len(decoder_channels)}")

    class SmallUNet(nn.Module):
        def __init__(self, cin: int, base_channels: int = base_channels, depth: int = depth):
            super().__init__()
            self.depth = max(2, int(depth))
            self.base_channels = max(8, int(base_channels))
            self.padding = padding
            self.use_batch_norm = use_batch_norm
            self.dropout = dropout
            self.activation = activation
            self.output_activation = output_activation

            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()

            encoder_channels = [cin] + [self.base_channels * (2 ** i) for i in range(self.depth)]
            for i in range(self.depth):
                self.encoders.append(_conv_block(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    padding,
                    use_batch_norm,
                    dropout,
                    activation,
                ))
                if i < self.depth - 1:
                    self.pools.append(nn.MaxPool2d(2))

            bottleneck_channels = self.base_channels * (2 ** self.depth)
            self.bottleneck = _conv_block(
                encoder_channels[-1],
                bottleneck_channels,
                padding,
                use_batch_norm,
                dropout,
                activation,
            )

            if decoder_channels is None:
                self.decoder_channels = [
                    self.base_channels * (2 ** (self.depth - 1 - i)) for i in range(self.depth)
                ]
            else:
                self.decoder_channels = list(decoder_channels)

            self.output = nn.Conv2d(self.decoder_channels[-1], 1, kernel_size=1)
            self.final_activation = build_activation(output_activation)

        def forward(self, x):
            skips = []
            for i, encoder in enumerate(self.encoders):
                x = encoder(x)
                skips.append(x)
                if i < self.depth - 1:
                    x = self.pools[i](x)

            x = self.bottleneck(x)

            for i in range(len(self.decoder_channels)):
                x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                skip = skips[-1 - i]
                if x.shape[2:] != skip.shape[2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                out_ch = self.decoder_channels[i]
                x = _conv_block(x.shape[1], out_ch, self.padding, self.use_batch_norm, self.dropout, self.activation)(x)

            x = self.output(x)
            if self.output_activation != "identity" and not self.training:
                x = self.final_activation(x)
            return x

    return SmallUNet(in_channels, base_channels=base_channels, depth=depth)


def build_unetpp(
    in_channels: int,
    base_channels: int = 32,
    depth: int = 4,
    padding: bool = True,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    activation: str = DEFAULT_HIDDEN_ACTIVATION,
    output_activation: str = DEFAULT_OUTPUT_ACTIVATION,
    decoder_channels: Optional[list[int]] = None,
    **kwargs,
):
    """Compatibility wrapper for U-Net++-style CLI selection.

    The project pipeline expects a single, stable encoder/decoder contract and the
    current U-Net implementation is the proven one. This keeps `--model unetpp`
    callable without changing the rest of the training stack.
    """
    return build_unet(
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
        padding=padding,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        activation=activation,
        output_activation=output_activation,
        decoder_channels=decoder_channels,
        **kwargs,
    )


SmallUNet = build_unet


def tversky_loss(logits, target, valid_mask=None, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
    """Tversky loss masked to valid pixels only; matches the logic in the legacy training script."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for Tversky loss.")

    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(1)
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.float32)
    elif valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(1)

    prob = torch.sigmoid(logits) * valid_mask
    target_masked = target * valid_mask

    tp = (prob * target_masked).sum(dim=(1, 2, 3))
    fp = (prob * (1.0 - target_masked)).sum(dim=(1, 2, 3))
    fn = ((1.0 - prob) * target_masked).sum(dim=(1, 2, 3))

    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - t).mean()


if __name__ == "__main__":
    print("U-Net module imported successfully.")
