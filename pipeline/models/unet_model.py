
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

from pipeline.models.activations import DEFAULT_HIDDEN_ACTIVATION, build_activation


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
    output_activation: str = "identity"
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
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1 if padding else 0,
            bias=not use_batch_norm,
        )
    ]

    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))

    layers.append(build_activation(activation))

    layers.append(
        nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1 if padding else 0,
            bias=not use_batch_norm,
        )
    )

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
    output_activation: str = "identity",
    decoder_channels: Optional[list[int]] = None,
    **kwargs,
):
    if torch is None or nn is None:
        raise ModuleNotFoundError("PyTorch is required to build the U-Net model.")

    depth = max(2, int(depth))
    base_channels = max(8, int(base_channels))

    if decoder_channels is None:
        decoder_channels = [
            base_channels * (2 ** (depth - 1 - i))
            for i in range(depth)
        ]
    else:
        decoder_channels = [max(1, int(ch)) for ch in decoder_channels]

    if len(decoder_channels) != depth:
        raise ValueError(
            f"decoder_channels must have length {depth}, "
            f"got {len(decoder_channels)}"
        )

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.depth = depth
            self.base_channels = base_channels

            encoder_channels = [
                base_channels * (2 ** i)
                for i in range(depth)
            ]

            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()

            current_channels = in_channels

            for out_channels in encoder_channels:
                self.encoders.append(
                    _conv_block(
                        current_channels,
                        out_channels,
                        padding,
                        use_batch_norm,
                        dropout,
                        activation,
                    )
                )

                self.pools.append(nn.MaxPool2d(kernel_size=2))
                current_channels = out_channels

            bottleneck_channels = base_channels * (2 ** depth)

            self.bottleneck = _conv_block(
                encoder_channels[-1],
                bottleneck_channels,
                padding,
                use_batch_norm,
                dropout,
                activation,
            )

            self.upsamplers = nn.ModuleList()
            self.decoders = nn.ModuleList()

            current_channels = bottleneck_channels

            for i in range(depth):
                skip_channels = encoder_channels[-1 - i]
                out_channels = decoder_channels[i]

                self.upsamplers.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    )
                )

                self.decoders.append(
                    _conv_block(
                        current_channels + skip_channels,
                        out_channels,
                        padding,
                        use_batch_norm,
                        dropout,
                        activation,
                    )
                )

                current_channels = out_channels

            self.output = nn.Conv2d(
                decoder_channels[-1],
                1,
                kernel_size=1,
            )

        def forward(self, x):
            skips = []

            for encoder, pool in zip(self.encoders, self.pools):
                x = encoder(x)
                skips.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            for upsample, decoder, skip in zip(
                self.upsamplers,
                self.decoders,
                reversed(skips),
            ):
                x = upsample(x)

                if x.shape[2:] != skip.shape[2:]:
                    x = nn.functional.interpolate(
                        x,
                        size=skip.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                x = torch.cat([x, skip], dim=1)
                x = decoder(x)

            return self.output(x)

    return SmallUNet()


def build_unetpp(
    in_channels: int,
    base_channels: int = 32,
    depth: int = 4,
    padding: bool = True,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    activation: str = DEFAULT_HIDDEN_ACTIVATION,
    output_activation: str = "identity",
    decoder_channels: Optional[list[int]] = None,
    **kwargs,
):
    return build_unet(
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
        padding=padding,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        activation=activation,
        output_activation="identity",
        decoder_channels=decoder_channels,
        **kwargs,
    )


SmallUNet = build_unet


def tversky_loss(
    logits,
    target,
    valid_mask=None,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for Tversky loss.")

    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)

    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:
        target = target.unsqueeze(1)

    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.float32)
    elif valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
    elif valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)

    prob = torch.sigmoid(logits)
    valid_mask = valid_mask.float()
    target = target.float()

    tp = (prob * target * valid_mask).sum(dim=(1, 2, 3))
    fp = (prob * (1.0 - target) * valid_mask).sum(dim=(1, 2, 3))
    fn = ((1.0 - prob) * target * valid_mask).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (
        tp + alpha * fp + beta * fn + eps
    )

    return (1.0 - tversky).mean()


if __name__ == "__main__":
    print("U-Net module imported successfully.")
