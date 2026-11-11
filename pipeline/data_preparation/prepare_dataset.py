from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.data_preparation.dataset_loader import find_dataset_dir, list_patch_files, load_manifest, validate_patch_file


@dataclass
class PreprocessConfig:
    """Configuration for patch-level preprocessing and augmentation."""
    normalize: bool = True
    fill_missing: bool = True
    use_manifest_fill_values: bool = True
    normalize_per_channel: bool = True
    augmentation: bool = False
    rotation_degrees: Sequence[int] = (0, 90, 180, 270)
    flip_horizontal: bool = True
    flip_vertical: bool = True
    seed: int = 20261022


DEFAULT_PREPROCESS = PreprocessConfig()


def _resolve_dataset_dir(dataset_dir: Optional[Path | str] = None) -> Path:
    return Path(dataset_dir) if dataset_dir is not None else find_dataset_dir()


def _channel_fill_values(manifest: Dict[str, Any]) -> Dict[str, float]:
    fill_values = manifest.get("preenchimento_nodata", {})
    return {key: float(value) for key, value in fill_values.items()}


def _channel_ranges(manifest: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    ranges = manifest.get("normalizacao", {})
    return {key: (float(v[0]), float(v[1])) for key, v in ranges.items()}


def normalize_patch_channels(X: np.ndarray, manifest: Optional[Dict[str, Any]] = None, per_channel: bool = True) -> np.ndarray:
    """Normalize the feature channels using manifest values if available."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected a 3D array (C, H, W), got shape {X.shape}")

    if manifest is None:
        return X

    ranges = _channel_ranges(manifest)
    if not ranges:
        return X

    out = X.copy()
    for i, channel_name in enumerate(manifest.get("canais_fisicos", [])[: X.shape[0]]):
        if channel_name not in ranges:
            continue
        low, high = ranges[channel_name]
        if high <= low:
            continue
        if per_channel:
            out[i] = (out[i] - low) / (high - low)
        else:
            out[i] = (out[i] - np.min(out[i])) / (np.max(out[i]) - np.min(out[i]) + 1e-8)
    return out


def fill_missing_channels(X: np.ndarray, manifest: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Replace nodata values with the fill values stored in the dataset manifest."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected a 3D array (C, H, W), got shape {X.shape}")

    if manifest is None:
        return X

    fill_values = _channel_fill_values(manifest)
    if not fill_values:
        return X

    out = X.copy()
    for i, channel_name in enumerate(manifest.get("canais_fisicos", [])[: X.shape[0]]):
        if channel_name in fill_values:
            out[i] = np.where(np.isclose(out[i], 0.0, atol=1e-12), float(fill_values[channel_name]), out[i])
    return out


def apply_preprocessing(X: np.ndarray, manifest: Optional[Dict[str, Any]] = None, config: Optional[PreprocessConfig] = None) -> np.ndarray:
    """Apply patch preprocessing: fill missing values and normalize channels."""
    config = config or DEFAULT_PREPROCESS
    out = X
    if config.fill_missing and manifest is not None:
        out = fill_missing_channels(out, manifest)
    if config.normalize and manifest is not None:
        out = normalize_patch_channels(out, manifest, per_channel=config.normalize_per_channel)
    return np.asarray(out, dtype=np.float32)


def augment_patch(X: np.ndarray, y: np.ndarray, valido: np.ndarray, config: Optional[PreprocessConfig] = None, rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple spatial augmentation for training: rotations and flips."""
    config = config or DEFAULT_PREPROCESS
    if not config.augmentation:
        return X.copy(), y.copy(), valido.copy()

    rng = rng or np.random.RandomState(config.seed)

    # Random choice among valid options
    choices = ["identity", "flip_h", "flip_v", "rot_90", "rot_180", "rot_270"]
    mode = choices[int(rng.randint(len(choices)))]

    if mode == "flip_h":
        X = X[:, :, ::-1]
        y = y[:, ::-1]
        valido = valido[:, ::-1]
    elif mode == "flip_v":
        X = X[:, ::-1, :]
        y = y[::-1, :]
        valido = valido[::-1, :]
    elif mode == "rot_90":
        X = np.rot90(X, axes=(1, 2))
        y = np.rot90(y, k=1)
        valido = np.rot90(valido, k=1)
    elif mode == "rot_180":
        X = np.rot90(X, k=2, axes=(1, 2))
        y = np.rot90(y, k=2)
        valido = np.rot90(valido, k=2)
    elif mode == "rot_270":
        X = np.rot90(X, k=3, axes=(1, 2))
        y = np.rot90(y, k=3)
        valido = np.rot90(valido, k=3)

    return X.astype(np.float32), y.astype(np.float32), valido.astype(np.float32)


def load_and_preprocess_patch(file_path: Path | str, manifest: Optional[Dict[str, Any]] = None, config: Optional[PreprocessConfig] = None, augment: bool = False) -> Dict[str, np.ndarray]:
    """Load a patch, validate it, apply preprocessing, and optionally augment it."""
    file_path = Path(file_path)
    info = validate_patch_file(file_path)
    with np.load(file_path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        valido = np.asarray(data["valido"], dtype=np.float32)

    X = apply_preprocessing(X, manifest=manifest, config=config)
    if augment:
        X, y, valido = augment_patch(X, y, valido, config=config)

    return {
        "path": str(file_path),
        "X": X,
        "y": y,
        "valido": valido,
        "meta": info,
    }


def _print_progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    percent = current / total
    filled = int(30 * percent)
    bar = "#" * filled + "-" * (30 - filled)
    print(f"{label}: [{bar}] {current}/{total} ({percent * 100:5.1f}%)", end="\r")
    if current >= total:
        print()


def prepare_dataset(dataset_dir: Optional[Path | str] = None, limit: Optional[int] = None, config: Optional[PreprocessConfig] = None, augment: bool = False) -> Dict[str, Any]:
    """Prepare the dataset into a list of validated, normalized patches.

    Parameters
    ----------
    dataset_dir:
        Directory containing the dataset. If omitted, it is inferred automatically.
    limit:
        Optional number of patches to load for a quick test.
    config:
        Preprocessing configuration.
    augment:
        Whether to apply light spatial augmentation.
    """
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    manifest = load_manifest(dataset_dir)
    patch_paths = list_patch_files(dataset_dir)
    if limit is not None and limit > 0:
        patch_paths = patch_paths[:limit]

    prepared = []
    print("[Stage 1/2] Loading dataset and preparing patches...")
    total = len(patch_paths)
    for idx, patch_path in enumerate(patch_paths, start=1):
        prepared.append(load_and_preprocess_patch(patch_path, manifest=manifest, config=config, augment=augment))
        _print_progress("Loading patches", idx, total)
    print("[Stage 1/2] Dataset loading complete.")

    return {
        "dataset_dir": str(dataset_dir),
        "manifest": manifest,
        "patches": prepared,
        "n_patches": len(prepared),
    }


if __name__ == "__main__":
    ds = prepare_dataset(limit=2, config=DEFAULT_PREPROCESS, augment=False)
    print(f"Dataset dir: {ds['dataset_dir']}")
    print(f"N patches: {ds['n_patches']}")
    if ds["patches"]:
        first = ds["patches"][0]
        print(f"Patch: {first['meta']['name']}")
        print(f"X shape: {first['X'].shape}")
        print(f"y shape: {first['y'].shape}")
