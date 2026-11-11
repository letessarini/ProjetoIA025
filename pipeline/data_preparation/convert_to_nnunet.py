from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover - the project requires nibabel for nnU-Net conversion
    raise RuntimeError("nibabel is required to convert this dataset to nnU-Net format.") from exc

from pipeline.data_preparation.dataset_loader import find_dataset_dir, list_patch_files, load_manifest


DEFAULT_DATASET_NAME = "FloodSegmentation"


def convert_project_patches_to_nnunet(
    raw_root: Optional[Path | str] = None,
    dataset_id: int = 102,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_dir: Optional[Path | str] = None,
) -> Path:
    """Convert the project patch dataset into the nnU-Net raw folder layout.

    Each patch becomes a single training case. Each feature channel is exported as a
    separate NIfTI file in `imagesTr/<case>_0000.nii.gz`, ..., and the binary target
    mask is exported as the segmentation file in `labelsTr/<case>.nii.gz`.
    Invalid pixels are preserved as ignore labels (255), matching the project's
    `valido` mask semantics.
    """
    raw_root = Path(raw_root) if raw_root is not None else Path(__file__).resolve().parents[2] / "nnUNet_raw"
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else find_dataset_dir()
    manifest = load_manifest(dataset_dir)
    channels = list(manifest.get("canais_fisicos", []))
    patch_paths = list_patch_files(dataset_dir)

    if not patch_paths:
        raise FileNotFoundError(f"No patch files found in dataset directory: {dataset_dir}")

    train_dir = raw_root / f"Dataset{int(dataset_id)}_{dataset_name}"
    if train_dir.exists():
        shutil.rmtree(train_dir)

    images_tr = train_dir / "imagesTr"
    labels_tr = train_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    if not channels:
        raise ValueError(f"Manifest in {dataset_dir} does not define any physical channels.")

    for patch_path in patch_paths:
        with np.load(patch_path, allow_pickle=False) as data:
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            valido = np.asarray(data["valido"], dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(f"Patch {patch_path} has invalid X shape {X.shape}; expected (C, H, W).")
        if X.shape[0] != len(channels):
            raise ValueError(
                f"Patch {patch_path} has {X.shape[0]} channels, but manifest defines {len(channels)} physical channels. "
                "This usually means the project config or dataset contract is stale."
            )
        if y.shape != valido.shape or y.shape != X.shape[1:]:
            raise ValueError(f"Patch {patch_path} has mismatched target/mask shapes: X={X.shape}, y={y.shape}, valido={valido.shape}")

        case_name = patch_path.stem
        for idx in range(X.shape[0]):
            image = X[idx]
            image_nii = nib.Nifti1Image(image.astype(np.float32), affine=np.eye(4))
            nib.save(image_nii, images_tr / f"{case_name}_{idx:04d}.nii.gz")

        label_map = np.where(valido > 0.5, y, 255).astype(np.uint8)
        seg_nii = nib.Nifti1Image(label_map, affine=np.eye(4))
        nib.save(seg_nii, labels_tr / f"{case_name}.nii.gz")

    dataset_json = {
        "channel_names": {str(i): channel_name for i, channel_name in enumerate(channels)},
        "labels": {"background": 0, "flood": 1, "ignore": 255},
        "numTraining": len(patch_paths),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Flood segmentation dataset exported from the project patch files.",
        "licence": "Project-specific dataset conversion for nnU-Net.",
        "converted_by": "ProjetoIA025 pipeline",
    }
    with (train_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)

    return train_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert the project patch dataset into a nnU-Net raw dataset.")
    parser.add_argument("--raw-root", type=str, default=str(Path(__file__).resolve().parents[2] / "nnUNet_raw"), help="Output root folder for nnUNet_raw")
    parser.add_argument("--dataset-id", type=int, default=102)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-dir", type=str, default=None)
    args = parser.parse_args()

    result = convert_project_patches_to_nnunet(
        raw_root=args.raw_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
    )
    print(f"Created nnU-Net dataset at {result}")
