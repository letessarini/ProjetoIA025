from __future__ import annotations

from typing import Callable

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency during import
    torch = None
    nn = None


DEFAULT_HIDDEN_ACTIVATION = "relu"
DEFAULT_OUTPUT_ACTIVATION = "sigmoid"


def _normalize_name(name: str | None) -> str:
    if name is None:
        return "identity"
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "none": "identity",
        "identity": "identity",
        "linear": "identity",
        "relu": "relu",
        "leakyrelu": "leaky_relu",
        "leaky_relu": "leaky_relu",
        "elu": "elu",
        "selu": "selu",
        "gelu": "gelu",
        "silu": "silu",
        "swish": "silu",
        "mish": "mish",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
        "softmax": "softmax",
        "logsoftmax": "logsoftmax",
    }
    return aliases.get(normalized, normalized)


def build_activation(name: str | Callable | None = DEFAULT_HIDDEN_ACTIVATION, *, inplace: bool = False, **kwargs):
    """Build a torch activation module from a name or a callable.

    Defaults:
    - hidden layers: ReLU (most common default for U-Nets)
    - output head: Sigmoid (common for binary segmentation)
    """
    if callable(name) and not isinstance(name, str):
        return name(**kwargs) if callable(name) and hasattr(name, "__call__") else name

    normalized = _normalize_name(name)
    if nn is None:
        raise ModuleNotFoundError("PyTorch is required to build activation layers.")

    registry = {
        "identity": lambda: nn.Identity(),
        "relu": lambda: nn.ReLU(inplace=inplace),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=kwargs.get("negative_slope", 0.01), inplace=inplace),
        "elu": lambda: nn.ELU(alpha=kwargs.get("alpha", 1.0), inplace=inplace),
        "selu": lambda: nn.SELU(inplace=inplace),
        "gelu": lambda: nn.GELU(),
        "silu": lambda: nn.SiLU(),
        "mish": lambda: nn.Mish(),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "softmax": lambda: nn.Softmax(dim=kwargs.get("dim", 1)),
        "logsoftmax": lambda: nn.LogSoftmax(dim=kwargs.get("dim", 1)),
    }

    if normalized not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported activation '{name}'. Available activations: {available}")

    return registry[normalized]()


ACTIVATION_FUNCTIONS = {
    "relu": "most common hidden activation in U-Nets",
    "leaky_relu": "helps with negative activations; useful when dead-ReLU is a concern",
    "elu": "smooth and negative-preserving alternative to ReLU",
    "selu": "self-normalizing activation used in some deeper networks",
    "gelu": "popular in transformer-like and modern conv designs",
    "silu": "smooth activation, common in modern architectures",
    "mish": "self-regularized non-monotonic activation",
    "tanh": "bounded activation used for smooth nonlinearities",
    "sigmoid": "common final activation for binary segmentation",
    "softmax": "multi-class output activation",
    "logsoftmax": "log-domain multi-class output activation",
    "identity": "no activation, useful for raw logits",
}


__all__ = [
    "ACTIVATION_FUNCTIONS",
    "DEFAULT_HIDDEN_ACTIVATION",
    "DEFAULT_OUTPUT_ACTIVATION",
    "build_activation",
    "_normalize_name",
]
