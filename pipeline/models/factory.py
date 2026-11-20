from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Callable

from pipeline.models.unet_model import build_unet, build_unetpp
from pipeline.models.nnunet_adapter import build_nnunet_model


MODEL_DEPENDENCIES = {
    "unet_resnet34": ("segmentation_models_pytorch", "segmentation-models-pytorch"),
}


def load_model_dependency(model_name: str):
    module, package = MODEL_DEPENDENCIES[model_name]
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise RuntimeError(f"Unable to import dependencies for {model_name} using {sys.executable}: {exc}") from exc
        command_prefix = "& " if os.name == "nt" else ""
        raise RuntimeError(
            f"{model_name} requires {package}. Install it with the Python running the pipeline:\n"
            f'  {command_prefix}"{sys.executable}" -m pip install {package}'
        ) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"Unable to import {package} using {sys.executable}: {exc}") from exc


def build_unet_resnet34(in_channels: int, encoder_weights: str | None = "imagenet"):
    smp = load_model_dependency("unet_resnet34")

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )

MODEL_REGISTRY: dict[str, Callable[..., object]] = {
    "unet": build_unet,
    "unet_resnet34": build_unet_resnet34,
    "unetpp": build_unetpp,
    "unet_pp": build_unetpp,
    "u_net_pp": build_unetpp,
    "u_net_plus_plus": build_unetpp,
    "u_netpp": build_unetpp,
    "u-net-pp": build_unetpp,
    "u-net-plus-plus": build_unetpp,
    "unet++": build_unetpp,
    "unet_plus_plus": build_unetpp,
    "unet_small": build_unet,
    "u-net": build_unet,
    "u_net": build_unet,
    "u-net-small": build_unet,
    "u_net_small": build_unet,
    "nnunet": build_nnunet_model,
    "nn_unet": build_nnunet_model,
    "nn-u-net": build_nnunet_model,
    "nnu-net": build_nnunet_model,
}



def normalize_model_name(name: str) -> str:
    raw = (name or "unet").strip().lower()

    unetpp_aliases = {
        "unetpp",
        "unet++",
        "unet_pp",
        "u_net_pp",
        "u_netpp",
        "u-net-pp",
        "u-net++",
        "u_net++",
        "unet_plus_plus",
        "u_net_plus_plus",
        "u-net-plus-plus",
    }
    if raw in unetpp_aliases:
        return "unetpp"
    candidate = raw.replace("-", "_").replace("+", "")
    if candidate in {"u", "u_net"}:
        return "unet"
    if candidate in {"u_small"}:
        return "unet_small"
    return candidate


def get_model(name: str, in_channels: int, model_config: dict | None = None, **kwargs):
    """Return a model instance by name and merge model-specific config values."""
    model_name = normalize_model_name(name)
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted({k.replace("_", "-") for k in MODEL_REGISTRY}))
        raise ValueError(f"Unsupported model '{name}'. Available models: {available}")

    cfg = dict(model_config or {})
    cfg.update(kwargs)
    cfg["in_channels"] = int(in_channels)
    return MODEL_REGISTRY[model_name](**cfg)
