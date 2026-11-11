from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_dataset_dir() -> Path:
    """Find the dataset directory relative to the repository root.

    This accepts the current folder layout used in this project:
    - REPO_ROOT/DATASET_UNET
    - REPO_ROOT/DATASET_UNET/DATASET_UNET
    - REPO_ROOT/data/DATASET_UNET
    """
    candidates = [
        REPO_ROOT / "DATASET_UNET",
        REPO_ROOT / "DATASET_UNET" / "DATASET_UNET",
        REPO_ROOT / "data" / "DATASET_UNET",
    ]
    for candidate in candidates:
        manifest = candidate / "manifest.json"
        patches = candidate / "patches"
        if manifest.exists() and patches.exists():
            return candidate
    return REPO_ROOT / "DATASET_UNET"


def load_manifest(dataset_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else find_dataset_dir()
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_patch_files(dataset_dir: Optional[Path | str] = None) -> List[Path]:
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else find_dataset_dir()

    # Support both layouts:
    #   DATASET_UNET/patches
    #   DATASET_UNET/DATASET_UNET/patches
    possible_patch_dirs = [
        dataset_dir / "patches",
        dataset_dir / "DATASET_UNET" / "patches",
    ]

    for patch_dir in possible_patch_dirs:
        if patch_dir.exists():
            return sorted(patch_dir.glob("*.npz"))

    raise FileNotFoundError(f"No patches directory found under {dataset_dir}")


def validate_patch_file(file_path: Path | str) -> Dict[str, Any]:
    """Light validation for a patch file. Returns the available metadata."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Patch file not found: {file_path}")

    with np.load(file_path, allow_pickle=False) as data:
        required = {"X", "y", "valido"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Patch missing required fields: {sorted(missing)}")

        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        valido = np.asarray(data["valido"])

        if X.ndim != 3:
            raise ValueError(f"X must have 3 dimensions, got {X.shape}")
        if y.shape != valido.shape:
            raise ValueError(f"y and valido shapes do not match: {y.shape} vs {valido.shape}")
        if X.shape[1:] != y.shape:
            raise ValueError(
                f"Patch spatial dimensions do not match: X={X.shape}, y={y.shape}, valido={valido.shape}"
            )

        payload = {
            "path": str(file_path),
            "name": file_path.name,
            "shape_X": tuple(X.shape),
            "shape_y": tuple(y.shape),
            "n_channels": X.shape[0],
        }

        for key in ("date", "precip_val", "frac_valida"):
            if key in data.files:
                payload[key] = data[key].item() if data[key].ndim == 0 else data[key].tolist()

        return payload


def load_dataset(dataset_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Return dataset metadata and patch paths in a single object."""
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else find_dataset_dir()
    manifest = load_manifest(dataset_dir)
    patches = list_patch_files(dataset_dir)
    return {
        "dataset_dir": str(dataset_dir),
        "manifest": manifest,
        "patches": [str(path) for path in patches],
        "n_patches": len(patches),
    }


if __name__ == "__main__":
    dataset_dir = find_dataset_dir()
    dataset = load_dataset(dataset_dir)
    print(f"Dataset dir: {dataset['dataset_dir']}")
    print(f"Number of patches: {dataset['n_patches']}")
    if dataset["n_patches"]:
        sample = validate_patch_file(dataset["patches"][0])
        print(f"Sample patch: {sample['name']}")
        print(f"X shape: {sample['shape_X']}")
